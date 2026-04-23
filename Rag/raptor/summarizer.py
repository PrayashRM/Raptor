# raptor/summarizer.py
"""
Hardened external API summarization for RAPTOR tree construction.

Handles ALL failure modes:
- Rate limits (429): exponential backoff + jitter
- Server errors (500/502/503): retry with backoff
- Timeouts: configurable hard timeout per request
- Auth failures: fail fast, clear error message
- Context overflow: automatic input splitting
- Silent truncation: output length validation
- Provider outages: fallback model chain
- Network failures: retry with backoff
- Complete API failure: extractive fallback (zero LLM)
- Cost tracking: token usage logged per call
- Privacy: raw text never logged

Output guarantee:
    A summary string is ALWAYS returned.
    Either LLM-generated or extractive fallback.
    The pipeline never fails due to summarization.
"""

from __future__ import annotations
import time
import random
import re
import hashlib
import litellm
from litellm import completion, token_counter

from core.logger import get_logger
from core.exceptions import SummarizationError
from core.utils import count_tokens
import config

logger = get_logger(__name__)

# Suppress LiteLLM internal verbose logging
litellm.suppress_debug_info = True

# ── Session-level cost accumulator ────────────────────────────────────────────
_session_cost = {
    "total_input_tokens":  0,
    "total_output_tokens": 0,
    "total_calls":         0,
    "fallback_calls":      0,
    "extractive_calls":    0,
    "failed_calls":        0,
}


def get_session_cost_report() -> dict:
    """Return accumulated token usage for this session."""
    return dict(_session_cost)


def _log_text_safe(text: str, label: str = "text") -> str:
    """
    Privacy-safe text logging.
    If SUMMARIZER_PRIVACY_SAFE_LOGGING is True,
    logs a hash instead of raw content.
    """
    if config.SUMMARIZER_PRIVACY_SAFE_LOGGING:
        h = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"[{label} hash={h} len={len(text)}]"
    return text[:100] + "..." if len(text) > 100 else text


def _compute_backoff(
    attempt: int,
    base: float,
    cap: float,
    jitter: bool = True,
) -> float:
    """
    Exponential backoff with optional jitter.
    Formula: min(cap, base * 2^attempt) + random(0, 1)
    Jitter prevents thundering herd on retry.
    """
    delay = min(cap, base * (2 ** attempt))
    if jitter:
        delay += random.uniform(0, 1)
    return delay


def _extract_retry_after(error_str: str) -> int:
    """
    Extract server-specified retry delay from error message.
    Returns extracted seconds or 0 if not found.
    """
    patterns = [
        r'retryDelay["\s:]+(\d+)',
        r'retry.after["\s:]+(\d+)',
        r'retry in (\d+)',
        r'Retry-After["\s:]+(\d+)',
        r'wait (\d+) second',
    ]
    for pattern in patterns:
        match = re.search(pattern, error_str, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def _is_retryable(error: Exception) -> tuple[bool, int]:
    """
    Determine if an error is retryable and how long to wait.

    Returns:
        (is_retryable: bool, wait_seconds: int)
    """
    error_str = str(error)

    # Rate limit — always retryable, use server wait time
    if isinstance(error, litellm.RateLimitError):
        server_wait = _extract_retry_after(error_str)
        wait = server_wait + 2 if server_wait > 0 else 30
        return True, wait

    # Service unavailable / overloaded — retryable
    if isinstance(error, litellm.ServiceUnavailableError):
        return True, 15

    # Timeout — retryable
    if isinstance(error, litellm.Timeout):
        return True, 5

    # API connection error — retryable
    if isinstance(error, litellm.APIConnectionError):
        return True, 10

    # Internal server errors (500/502/503/504) — retryable
    if isinstance(error, litellm.APIError):
        code = getattr(error, 'status_code', 0)
        if code in (500, 502, 503, 504):
            return True, 10

    # Auth failure — NOT retryable (wrong key, expired)
    if isinstance(error, litellm.AuthenticationError):
        return False, 0

    # Context window exceeded — NOT retryable (need different strategy)
    if isinstance(error, litellm.ContextWindowExceededError):
        return False, 0

    # Unknown errors — retry once with short wait
    return True, 5


def check_llm_available():
    """
    Verify primary LLM model is accessible.
    Tests primary model first, then validates fallbacks exist.
    Fails fast with actionable error message.
    """
    model = config.SUMMARIZER_PRIMARY_MODEL
    logger.info(f"Checking LLM availability: {model}")

    try:
        response = litellm.completion(
            model       = model,
            messages    = [{"role": "user", "content": "Reply with OK."}],
            max_tokens  = 5,
            temperature = 0,
            timeout     = 15,
        )
        reply = response.choices[0].message.content.strip()
        logger.info(
            f"Primary LLM available: {model} "
            f"(response: '{reply}')"
        )

    except litellm.AuthenticationError as e:
        raise SummarizationError(
            f"\n{'='*60}\n"
            f"API AUTHENTICATION FAILED\n"
            f"{'='*60}\n"
            f"Model: {model}\n"
            f"Error: {e}\n\n"
            f"Fix:\n"
            f"  Check API key in .env file\n"
            f"  Gemini: GEMINI_API_KEY\n"
            f"  Groq:   GROQ_API_KEY\n"
            f"{'='*60}"
        ) from e

    except Exception as e:
        logger.warning(
            f"Primary model check failed: {e}. "
            f"Fallbacks configured: {config.SUMMARIZER_FALLBACK_MODELS}"
        )
        # Non-fatal — fallbacks may work even if primary is down

    # Validate fallback models are configured
    valid_fallbacks = [
        m for m in config.SUMMARIZER_FALLBACK_MODELS if m
    ]
    logger.info(
        f"Fallback chain: {valid_fallbacks}"
    )


def _call_single_model(
    model: str,
    messages: list[dict],
    attempt: int = 0,
) -> str:
    """
    Single LLM call to one specific model.
    Handles retries for transient errors on this model.
    Raises SummarizationError if model fails all retries.
    """
    for retry in range(config.SUMMARIZER_MAX_RETRIES):
        try:
            response = litellm.completion(
                model       = model,
                messages    = messages,
                max_tokens  = config.SUMMARIZER_MAX_TOKENS,
                temperature = config.SUMMARIZER_TEMPERATURE,
                timeout     = config.SUMMARIZER_REQUEST_TIMEOUT,
            )

            result = response.choices[0].message.content
            if result is None or not result.strip():
                raise SummarizationError(
                    f"Model {model} returned empty response"
                )

            # Track token usage
            if config.SUMMARIZER_TRACK_COST:
                usage = getattr(response, 'usage', None)
                if usage:
                    _session_cost["total_input_tokens"] += getattr(
                        usage, 'prompt_tokens', 0
                    )
                    _session_cost["total_output_tokens"] += getattr(
                        usage, 'completion_tokens', 0
                    )
            _session_cost["total_calls"] += 1

            # Validate output was not silently truncated
            result_tokens = count_tokens(result.strip())
            if result_tokens >= config.SUMMARIZER_MAX_TOKENS - 10:
                logger.warning(
                    f"Output may be truncated: "
                    f"{result_tokens} tokens "
                    f"(max={config.SUMMARIZER_MAX_TOKENS})"
                )

            return result.strip()

        except (litellm.AuthenticationError,
                litellm.ContextWindowExceededError) as e:
            # Non-retryable on this model
            raise SummarizationError(
                f"Non-retryable error on {model}: {e}"
            ) from e

        except Exception as e:
            retryable, wait = _is_retryable(e)

            if not retryable:
                raise SummarizationError(
                    f"Non-retryable error on {model}: {e}"
                ) from e

            if retry < config.SUMMARIZER_MAX_RETRIES - 1:
                # Use server-specified wait if available,
                # otherwise exponential backoff
                if wait == 0:
                    wait = _compute_backoff(
                        retry,
                        config.SUMMARIZER_RETRY_BASE_DELAY,
                        config.SUMMARIZER_RETRY_MAX_DELAY,
                    )
                logger.warning(
                    f"Model {model} error (retry {retry+1}/"
                    f"{config.SUMMARIZER_MAX_RETRIES}): "
                    f"{type(e).__name__}. "
                    f"Waiting {wait:.1f}s..."
                )
                time.sleep(wait)
            else:
                raise SummarizationError(
                    f"Model {model} failed after "
                    f"{config.SUMMARIZER_MAX_RETRIES} retries: {e}"
                ) from e

    raise SummarizationError(
        f"Model {model} exhausted all retries"
    )


def _call_llm_with_fallbacks(messages: list[dict]) -> str:
    """
    Try primary model first, then fallback chain.
    Guaranteed to return a string or raise only if ALL
    models fail (extractive fallback handles that case).

    Model chain:
        primary → fallback_1 → fallback_2 → fallback_3
    """
    all_models = [
        config.SUMMARIZER_PRIMARY_MODEL
    ] + [
        m for m in config.SUMMARIZER_FALLBACK_MODELS if m
    ]

    last_error = None

    for model_idx, model in enumerate(all_models):
        if not model:
            continue

        is_fallback = model_idx > 0
        if is_fallback:
            _session_cost["fallback_calls"] += 1
            logger.warning(
                f"Primary failed. Trying fallback "
                f"{model_idx}/{len(all_models)-1}: {model}"
            )

        try:
            result = _call_single_model(model, messages)
            if is_fallback:
                logger.info(
                    f"Fallback model succeeded: {model}"
                )
            return result

        except SummarizationError as e:
            last_error = e
            logger.warning(
                f"Model {model} failed: {e}"
            )
            continue

    # All models failed
    _session_cost["failed_calls"] += 1
    raise SummarizationError(
        f"All models in fallback chain failed. "
        f"Last error: {last_error}"
    )


def _build_messages(system_content: str, user_content: str) -> list[dict]:
    """Build standardized message list for LLM call."""
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _call_llm(prompt: str) -> str:
    """
    Main LLM call entry point.
    Tries primary + fallbacks.
    Raises SummarizationError only if all models fail.
    Extractive fallback is handled at tree_builder level.
    """
    messages = _build_messages(
        system_content = (
            "You are a precise technical summarizer for research papers. "
            "Follow instructions exactly. "
            "Never introduce information not present in the source text. "
            "Be concise and factually accurate."
        ),
        user_content = prompt,
    )

    logger.debug(
        f"LLM call: {_log_text_safe(prompt, 'prompt')}"
    )

    return _call_llm_with_fallbacks(messages)


# ── Input management ───────────────────────────────────────────────────────────

def _dynamic_word_limit(input_text: str) -> int:
    """
    Calculate word limit as 30% of input token count.
    Min 100, max 350 words.
    """
    input_tokens = count_tokens(input_text)
    limit        = int(input_tokens * 0.30)
    return max(100, min(350, limit))


def _split_if_too_long(
    text: str,
    max_tokens: int = None,
) -> list[str]:
    """
    Split long text into safe-sized parts for LLM input.
    Splits at paragraph boundaries.
    Never cuts mid-paragraph.
    """
    if max_tokens is None:
        max_tokens = config.SUMMARIZER_MAX_PART_TOKENS

    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = text.split("\n\n")
    parts      = []
    current    = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # Single paragraph exceeds limit — split by sentences
        if para_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                s_tokens = count_tokens(sentence)
                if current_tokens + s_tokens > max_tokens and current:
                    parts.append("\n\n".join(current))
                    current        = [sentence]
                    current_tokens = s_tokens
                else:
                    current.append(sentence)
                    current_tokens += s_tokens
            continue

        if current_tokens + para_tokens > max_tokens and current:
            parts.append("\n\n".join(current))
            current        = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        parts.append("\n\n".join(current))

    logger.debug(
        f"Split input ({count_tokens(text)} tokens) "
        f"into {len(parts)} parts "
        f"(max {max_tokens} tokens each)"
    )
    return parts


def _validate_output_not_truncated(
    result: str,
    prompt: str,
) -> bool:
    """
    Check if output was silently truncated mid-sentence.
    Truncated outputs end without punctuation.
    """
    stripped = result.strip()
    if not stripped:
        return False
    # Check if ends with sentence-ending punctuation
    ends_properly = stripped[-1] in '.!?")'
    if not ends_properly:
        logger.warning(
            f"Output may be truncated "
            f"(ends with: '...{stripped[-20:]}')"
        )
    return ends_properly


# ── Summarization functions ────────────────────────────────────────────────────

def summarize_l1_section(
    section_title: str,
    section_hierarchy: list[str],
    paper_title: str,
    chunk_texts: list[str],
) -> str:
    """
    Generate L1 section summary.
    Handles long sections by splitting into parts.
    """
    combined_text = "\n\n".join(chunk_texts)
    word_limit    = _dynamic_word_limit(combined_text)

    logger.debug(
        f"L1 summarize: section='{section_title}' "
        f"input_tokens={count_tokens(combined_text)} "
        f"word_limit={word_limit}"
    )

    parts = _split_if_too_long(
        combined_text,
        max_tokens=config.SUMMARIZER_MAX_PART_TOKENS,
    )

    if len(parts) > 1:
        logger.info(
            f"Section '{section_title}' split into "
            f"{len(parts)} parts for summarization"
        )
        part_summaries = []
        per_part_limit = max(80, word_limit // len(parts))

        for part_idx, part in enumerate(parts):
            logger.debug(
                f"Summarizing part {part_idx+1}/{len(parts)}"
            )
            summary = _summarize_single_chunk(
                section_title, paper_title, part, per_part_limit
            )
            part_summaries.append(summary)

        # Combine part summaries into final L1 summary
        combined_parts = "\n\n".join(part_summaries)
        return _combine_partial_summaries(
            section_title, paper_title,
            combined_parts, word_limit
        )

    return _summarize_single_chunk(
        section_title, paper_title, combined_text, word_limit
    )


def _summarize_single_chunk(
    section_title: str,
    paper_title: str,
    text: str,
    word_limit: int,
) -> str:
    """Summarize a single chunk of text."""
    prompt = f"""Paper: {paper_title}
Section: {section_title}

Source text:
{text}

Write a technical summary that:
1. States the main contribution or finding of this section
2. Names every specific method, metric, architecture component, or result mentioned
3. Preserves ALL numerical values exactly as they appear in the source
4. Uses the same technical terminology as the source
5. Does NOT introduce any information not present in the source text

Length: {word_limit} words maximum.

Summary:"""

    return _call_llm(prompt)


def _combine_partial_summaries(
    section_title: str,
    paper_title: str,
    partial_summaries: str,
    word_limit: int,
) -> str:
    """Combine multiple part summaries into one coherent summary."""
    prompt = f"""Paper: {paper_title}
Section: {section_title}

Partial summaries to combine:
{partial_summaries}

Write one unified summary that:
1. Captures all key technical content from the partial summaries
2. Removes redundancy
3. Preserves ALL numerical values exactly
4. Does NOT add any new information

Length: {word_limit} words maximum.

Combined summary:"""

    return _call_llm(prompt)


def summarize_l2_group(
    paper_title: str,
    l1_summary_texts: list[str],
    section_titles: list[str],
) -> str:
    """Generate L2 group summary from related L1 summaries."""
    combined = "\n\n---\n\n".join([
        f"Section: {title}\n{summary}"
        for title, summary in zip(section_titles, l1_summary_texts)
    ])
    word_limit = _dynamic_word_limit(combined)

    # Check if combined L1 summaries exceed input limit
    parts = _split_if_too_long(
        combined,
        max_tokens=config.SUMMARIZER_MAX_PART_TOKENS,
    )

    if len(parts) > 1:
        logger.info(
            f"L2 input too long, summarizing in "
            f"{len(parts)} parts"
        )
        part_results = []
        for part in parts:
            prompt = f"""Paper: {paper_title}

Section summaries:
{part}

Summarize the key themes and findings:"""
            part_results.append(_call_llm(prompt))

        combined = "\n\n".join(part_results)

    prompt = f"""Paper: {paper_title}

Section summaries:
{combined}

Write a summary that:
1. Captures the overarching theme connecting these sections
2. Preserves key technical details and results from each section
3. Explains how these sections relate to each other
4. Contains ONLY information from the summaries above
5. Preserves ALL numerical values exactly

Length: {word_limit} words maximum.

Summary:"""

    return _call_llm(prompt)


def summarize_l3_root(
    paper_title: str,
    l2_summary_texts: list[str],
) -> str:
    """
    Generate L3 root summary from all L2 summaries.
    Always built. Never skipped.
    """
    combined   = "\n\n---\n\n".join(l2_summary_texts)
    word_limit = 350

    # Split if needed (rare but possible for large papers)
    if count_tokens(combined) > config.SUMMARIZER_MAX_PART_TOKENS:
        parts = _split_if_too_long(combined)
        intermediates = []
        for part in parts:
            prompt = f"""Paper: {paper_title}

Summarize these section summaries concisely:
{part}"""
            intermediates.append(_call_llm(prompt))
        combined = "\n\n".join(intermediates)

    prompt = f"""Paper: {paper_title}

Section group summaries:
{combined}

Write a comprehensive paper summary that:
1. States the core problem the paper solves
2. Describes the proposed approach at a technical level
3. Lists the main results with specific numbers preserved exactly
4. States the key conclusions and contributions
5. Contains ONLY information from the summaries above

Length: 250-{word_limit} words.

Paper summary:"""

    return _call_llm(prompt)
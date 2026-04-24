# raptor/summarizer.py
"""
Async summarization at chunk level for RAPTOR tree.

Features:
- Async parallel calls where provider allows
- Sequential fallback when rate limited
- Multi-key rotation per provider
- Full retry with exponential backoff + jitter
- Provider-aware concurrency limits
- Rate limit proactive throttling
- All API failure modes handled gracefully
- Extractive fallback guarantees output always

Async strategy:
- Different providers: always parallel (different rate buckets)
- Same provider, multiple keys: parallel (different quotas)
- Same provider, same key: sequential (share rate limit)
- Free tier detected: force sequential to avoid quota burn
"""

from __future__ import annotations
import os
import re
import time
import random
import asyncio
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Optional
import litellm

from core.logger import get_logger
from core.exceptions import SummarizationError
from core.utils import count_tokens
import config

logger = get_logger(__name__)
litellm.suppress_debug_info = True


# ── Session statistics ─────────────────────────────────────────────────────────
_session_stats = {
    "total_calls":        0,
    "successful_calls":   0,
    "fallback_calls":     0,
    "extractive_calls":   0,
    "failed_calls":       0,
    "total_input_tokens": 0,
    "total_output_tokens":0,
    "rate_limit_hits":    0,
    "total_wait_seconds": 0.0,
}
_stats_lock = threading.Lock()


def _update_stats(**kwargs):
    with _stats_lock:
        for key, value in kwargs.items():
            if key in _session_stats:
                _session_stats[key] += value


def get_session_stats() -> dict:
    with _stats_lock:
        return dict(_session_stats)


# ── Provider key rotation ──────────────────────────────────────────────────────
_key_rotation = {}
_key_rotation_lock = threading.Lock()


def _get_next_key(provider: str) -> str:
    """Thread-safe round-robin key rotation per provider."""
    with _key_rotation_lock:
        if provider == "gemini":
            keys = config.SUMMARIZER_GEMINI_KEYS
        elif provider == "groq":
            keys = config.SUMMARIZER_GROQ_KEYS
        else:
            return ""

        if not keys:
            return ""

        idx = _key_rotation.get(provider, 0) % len(keys)
        _key_rotation[provider] = idx + 1
        return keys[idx]


def _detect_provider(model: str) -> str:
    """Detect provider from model string."""
    model_lower = model.lower()
    if "gemini" in model_lower:
        return "gemini"
    if "groq" in model_lower:
        return "groq"
    if "openrouter" in model_lower:
        return "openrouter"
    if "gpt" in model_lower or "openai" in model_lower:
        return "openai"
    if "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic"
    if "ollama" in model_lower:
        return "ollama"
    return "unknown"


def _set_provider_key(model: str, override_key: str = ""):
    """Set the correct API key env var for a model."""
    if override_key:
        provider = _detect_provider(model)
        key_map = {
            "gemini":    "GEMINI_API_KEY",
            "groq":      "GROQ_API_KEY",
            "openrouter":"OPENROUTER_API_KEY",
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = key_map.get(provider)
        if env_var:
            os.environ[env_var] = override_key
        return

    provider = _detect_provider(model)
    key = _get_next_key(provider)
    if key:
        key_map = {
            "gemini": "GEMINI_API_KEY",
            "groq":   "GROQ_API_KEY",
        }
        env_var = key_map.get(provider)
        if env_var:
            os.environ[env_var] = key

    if provider == "openrouter" and config.SUMMARIZER_OPENROUTER_KEY:
        os.environ["OPENROUTER_API_KEY"] = \
            config.SUMMARIZER_OPENROUTER_KEY


def _model_has_valid_key(model: str) -> bool:
    """Check if required API key exists for model."""
    provider = _detect_provider(model)
    checks = {
        "gemini":     bool(config.SUMMARIZER_GEMINI_KEYS),
        "groq":       bool(config.SUMMARIZER_GROQ_KEYS),
        "openrouter": bool(config.SUMMARIZER_OPENROUTER_KEY),
        "openai":     bool(os.getenv("OPENAI_API_KEY", "")),
        "anthropic":  bool(os.getenv("ANTHROPIC_API_KEY", "")),
        "ollama":     True,
        "unknown":    True,
    }
    return checks.get(provider, True)


# ── Error classification ───────────────────────────────────────────────────────
@dataclass
class ErrorInfo:
    is_retryable:     bool
    wait_seconds:     int
    is_daily_quota:   bool = False
    is_auth_failure:  bool = False
    is_context_limit: bool = False


def _classify_error(error: Exception) -> ErrorInfo:
    """
    Classify an API error for retry strategy.
    Returns ErrorInfo with full retry metadata.
    """
    error_str = str(error)

    # Auth failure — never retryable
    if isinstance(error, litellm.AuthenticationError):
        return ErrorInfo(False, 0, is_auth_failure=True)

    # Bad request with auth markers — never retryable
    if isinstance(error, litellm.BadRequestError):
        auth_markers = [
            "401", "invalid_api_key", "invalid api key",
            "authentication", "unauthorized", "api_key_invalid"
        ]
        if any(m in error_str.lower() for m in auth_markers):
            return ErrorInfo(False, 0, is_auth_failure=True)
        return ErrorInfo(False, 0)

    # Context window exceeded — not retryable
    if isinstance(error, litellm.ContextWindowExceededError):
        return ErrorInfo(False, 0, is_context_limit=True)

    # Rate limit — check if daily quota (not retryable today)
    if isinstance(error, litellm.RateLimitError):
        daily_markers = [
            "PerDay", "per_day", "daily", "DAILY",
            "GenerateRequestsPerDayPerProject",
        ]
        is_daily = any(m in error_str for m in daily_markers)
        if is_daily:
            return ErrorInfo(False, 0, is_daily_quota=True)

        # Per-minute rate limit — retryable with server wait
        server_wait = _extract_retry_after(error_str)
        wait = server_wait + 2 if server_wait > 0 else 30
        return ErrorInfo(True, wait)

    # Service unavailable — retryable
    if isinstance(error, litellm.ServiceUnavailableError):
        return ErrorInfo(True, 15)

    # Timeout — retryable
    if isinstance(error, litellm.Timeout):
        return ErrorInfo(True, 5)

    # Connection error — retryable
    if isinstance(error, litellm.APIConnectionError):
        return ErrorInfo(True, 10)

    # Server errors — retryable
    if isinstance(error, litellm.APIError):
        code = getattr(error, 'status_code', 0)
        if code in (500, 502, 503, 504):
            return ErrorInfo(True, 10)
        return ErrorInfo(False, 0)

    # Unknown — try once more
    return ErrorInfo(True, 5)


def _extract_retry_after(error_str: str) -> int:
    """Extract server-specified retry delay from error."""
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


def _compute_backoff(attempt: int) -> float:
    """Exponential backoff with jitter."""
    base  = config.SUMMARIZER_RETRY_BASE_DELAY
    cap   = config.SUMMARIZER_RETRY_MAX_DELAY
    delay = min(cap, base * (2 ** attempt))
    return delay + random.uniform(0, min(delay * 0.1, 5))


# ── Provider concurrency control ───────────────────────────────────────────────
# Semaphores limit concurrent calls per provider
# Prevents overwhelming free tier quotas
_provider_semaphores: dict[str, asyncio.Semaphore] = {}
_semaphore_lock = threading.Lock()


def _get_provider_semaphore(provider: str) -> asyncio.Semaphore:
    """
    Get or create a semaphore for a provider.
    Controls max concurrent requests to that provider.
    Must be called from async context.
    """
    if provider not in _provider_semaphores:
        limit = config.SUMMARIZER_MAX_CONCURRENT_SAME_PROVIDER
        _provider_semaphores[provider] = asyncio.Semaphore(limit)
    return _provider_semaphores[provider]


# ── Core async LLM call ────────────────────────────────────────────────────────

async def _async_call_single_model(
    model: str,
    messages: list[dict],
) -> str:
    """
    Async LLM call to one model with full retry logic.
    Runs in executor to avoid blocking event loop
    (litellm sync calls wrapped in run_in_executor).
    """
    provider = _detect_provider(model)
    semaphore = _get_provider_semaphore(provider)

    async with semaphore:
        for retry in range(config.SUMMARIZER_MAX_RETRIES):
            try:
                _set_provider_key(model)

                # Proactive rate limit throttle
                if config.SUMMARIZER_MIN_CALL_GAP_SECONDS > 0:
                    await asyncio.sleep(
                        config.SUMMARIZER_MIN_CALL_GAP_SECONDS
                    )

                # Run sync litellm in thread executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: litellm.completion(
                        model       = model,
                        messages    = messages,
                        max_tokens  = config.SUMMARIZER_MAX_TOKENS,
                        temperature = config.SUMMARIZER_TEMPERATURE,
                        timeout     = config.SUMMARIZER_REQUEST_TIMEOUT,
                    )
                )

                result = response.choices[0].message.content
                if not result or not result.strip():
                    raise SummarizationError(
                        f"Empty response from {model}"
                    )

                # Track usage
                usage = getattr(response, 'usage', None)
                if usage:
                    _update_stats(
                        total_calls        = 1,
                        successful_calls   = 1,
                        total_input_tokens = getattr(
                            usage, 'prompt_tokens', 0
                        ),
                        total_output_tokens = getattr(
                            usage, 'completion_tokens', 0
                        ),
                    )

                return result.strip()

            except Exception as e:
                info = _classify_error(e)

                if not info.is_retryable:
                    _update_stats(failed_calls=1)
                    raise SummarizationError(
                        f"Non-retryable error on {model}: "
                        f"{type(e).__name__}: {e}"
                    ) from e

                if retry < config.SUMMARIZER_MAX_RETRIES - 1:
                    wait = info.wait_seconds if info.wait_seconds > 0 \
                           else _compute_backoff(retry)
                    _update_stats(
                        rate_limit_hits   = 1,
                        total_wait_seconds = wait,
                    )
                    logger.warning(
                        f"Summarizer {model} retry "
                        f"{retry+1}/{config.SUMMARIZER_MAX_RETRIES} "
                        f"({type(e).__name__}) "
                        f"waiting {wait:.1f}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    _update_stats(failed_calls=1)
                    raise SummarizationError(
                        f"Model {model} failed after "
                        f"{config.SUMMARIZER_MAX_RETRIES} retries: {e}"
                    ) from e

    raise SummarizationError(f"Model {model} exhausted retries")


async def _async_call_with_fallbacks(
    messages: list[dict],
) -> str:
    """
    Try primary model, then fallback chain asynchronously.
    
    Async strategy:
    - All models in chain tried sequentially (one at a time)
    - Within each model, calls are async (non-blocking wait)
    - This is correct: fallbacks are sequential by design
    - Parallelism happens at the CLUSTER level (multiple
      clusters summarized concurrently), not model level
    """
    all_models = [
        config.SUMMARIZER_PRIMARY_MODEL
    ] + [
        m for m in config.SUMMARIZER_FALLBACK_MODELS if m
    ]

    # Filter models without valid keys
    valid_models = [
        m for m in all_models
        if m and _model_has_valid_key(m)
    ]

    if not valid_models:
        raise SummarizationError(
            "No models with valid API keys configured. "
            "Check SUMMARIZER_GEMINI_KEYS or SUMMARIZER_GROQ_KEYS."
        )

    last_error = None

    for model_idx, model in enumerate(valid_models):
        is_fallback = model_idx > 0
        if is_fallback:
            _update_stats(fallback_calls=1)
            logger.warning(
                f"Summarizer fallback {model_idx}: {model}"
            )

        try:
            result = await _async_call_single_model(
                model    = model,
                messages = messages,
            )
            if is_fallback:
                logger.info(f"Summarizer fallback succeeded: {model}")
            return result

        except SummarizationError as e:
            last_error = e
            logger.warning(f"Summarizer model {model} failed: {e}")
            continue

    raise SummarizationError(
        f"All {len(valid_models)} summarizer models failed. "
        f"Last: {last_error}"
    )


def _sync_call_with_fallbacks(messages: list[dict]) -> str:
    """
    Synchronous wrapper for _async_call_with_fallbacks.
    Used when called from sync context (tree_builder).
    Creates new event loop if needed.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context — use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                _async_call_with_fallbacks(messages), loop
            )
            return future.result(
                timeout=config.SUMMARIZER_REQUEST_TIMEOUT * 10
            )
        else:
            return loop.run_until_complete(
                _async_call_with_fallbacks(messages)
            )
    except RuntimeError:
        # No event loop — create one
        return asyncio.run(_async_call_with_fallbacks(messages))


# ── Input management ───────────────────────────────────────────────────────────

def _dynamic_word_limit(input_text: str) -> int:
    """30% compression, min 100, max 350 words."""
    tokens = count_tokens(input_text)
    return max(100, min(350, int(tokens * 0.30)))


def _split_if_too_long(
    text: str,
    max_tokens: int = None,
) -> list[str]:
    """
    Split text at paragraph boundaries.
    Never cuts mid-sentence.
    """
    if max_tokens is None:
        max_tokens = config.SUMMARIZER_MAX_PART_TOKENS

    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs     = text.split("\n\n")
    parts          = []
    current        = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # Single paragraph too long — split by sentences
        if para_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                s_tokens = count_tokens(sentence)
                if current_tokens + s_tokens > max_tokens \
                        and current:
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
        f"into {len(parts)} parts"
    )
    return parts


# ── LLM call entry point ───────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """
    Synchronous LLM call entry point.
    Called by summarize_* functions.
    Handles both sync and async contexts.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise technical summarizer for "
                "research papers. Follow instructions exactly. "
                "Never introduce information not present in "
                "the source text. Be factually accurate."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return _sync_call_with_fallbacks(messages)


async def _async_call_llm(prompt: str) -> str:
    """
    Async LLM call entry point.
    Called directly from async pipeline for parallel execution.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise technical summarizer for "
                "research papers. Follow instructions exactly. "
                "Never introduce information not present in "
                "the source text. Be factually accurate."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return await _async_call_with_fallbacks(messages)


# ── LLM availability check ─────────────────────────────────────────────────────

def check_llm_available():
    """
    Verify at least one summarization model is accessible.
    Called once at pipeline startup.
    Fails fast with clear error if nothing works.
    """
    logger.info(
        f"Checking summarizer: {config.SUMMARIZER_PRIMARY_MODEL}"
    )

    valid_models = [
        m for m in [
            config.SUMMARIZER_PRIMARY_MODEL
        ] + config.SUMMARIZER_FALLBACK_MODELS
        if m and _model_has_valid_key(m)
    ]

    if not valid_models:
        raise SummarizationError(
            f"\n{'='*60}\n"
            f"NO SUMMARIZER MODELS CONFIGURED\n"
            f"{'='*60}\n"
            f"No valid API keys found for any model.\n\n"
            f"Fix:\n"
            f"  Set SUMMARIZER_GEMINI_API_KEY in .env\n"
            f"  OR set SUMMARIZER_GROQ_API_KEY in .env\n"
            f"{'='*60}"
        )

    # Test primary model
    try:
        _set_provider_key(config.SUMMARIZER_PRIMARY_MODEL)
        response = litellm.completion(
            model       = config.SUMMARIZER_PRIMARY_MODEL,
            messages    = [{"role": "user", "content": "Reply OK."}],
            max_tokens  = 5,
            temperature = 0,
            timeout     = 15,
        )
        reply = response.choices[0].message.content.strip()
        logger.info(
            f"Summarizer ready: {config.SUMMARIZER_PRIMARY_MODEL} "
            f"(response: '{reply}')"
        )
    except litellm.AuthenticationError as e:
        raise SummarizationError(
            f"\n{'='*60}\n"
            f"SUMMARIZER AUTH FAILED\n"
            f"{'='*60}\n"
            f"Model: {config.SUMMARIZER_PRIMARY_MODEL}\n"
            f"Error: {e}\n\n"
            f"Fix: Check SUMMARIZER_GEMINI_API_KEY in .env\n"
            f"{'='*60}"
        ) from e
    except Exception as e:
        logger.warning(
            f"Primary summarizer check failed: {e}. "
            f"Fallbacks available: "
            f"{config.SUMMARIZER_FALLBACK_MODELS}"
        )


# ── Parallel summarization ─────────────────────────────────────────────────────

async def summarize_clusters_parallel(
    cluster_tasks: list[dict],
) -> list[str]:
    """
    Summarize multiple clusters in parallel.

    Each task is a dict with keys matching
    summarize_l1_section() parameters.

    Async parallelism strategy:
    - Tasks assigned to different providers run truly parallel
    - Tasks assigned to same provider are limited by semaphore
    - Free tier: semaphore=1 → effectively sequential per provider
    - Paid tier: semaphore=N → N parallel calls per provider

    Args:
        cluster_tasks: List of dicts with summarization parameters

    Returns:
        List of summary strings, same order as input tasks
    """
    if not cluster_tasks:
        return []

    total_concurrent = min(
        len(cluster_tasks),
        config.SUMMARIZER_MAX_CONCURRENT_TOTAL,
    )

    logger.info(
        f"Parallel summarization: {len(cluster_tasks)} clusters, "
        f"max_concurrent={total_concurrent}"
    )

    # Global semaphore limits total concurrent calls
    global_semaphore = asyncio.Semaphore(total_concurrent)

    async def run_task(task: dict, idx: int) -> tuple[int, str]:
        """Run one summarization task with global semaphore."""
        async with global_semaphore:
            try:
                prompt = _build_l1_prompt(
                    section_title = task["section_title"],
                    paper_title   = task["paper_title"],
                    text          = "\n\n".join(task["chunk_texts"]),
                    word_limit    = task["word_limit"],
                )
                result = await _async_call_llm(prompt)
                logger.info(
                    f"Cluster {idx} summarized: "
                    f"'{task['section_title'][:30]}' "
                    f"({count_tokens(result)} tokens)"
                )
                return idx, result
            except SummarizationError as e:
                logger.warning(
                    f"Cluster {idx} LLM failed: {e}. "
                    f"Using extractive fallback."
                )
                fallback = extractive_fallback(task["chunk_texts"])
                return idx, fallback

    # Launch all tasks concurrently
    coroutines = [
        run_task(task, idx)
        for idx, task in enumerate(cluster_tasks)
    ]
    results_with_idx = await asyncio.gather(*coroutines)

    # Sort by original index to preserve order
    results_with_idx.sort(key=lambda x: x[0])
    return [r for _, r in results_with_idx]


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_l1_prompt(
    section_title: str,
    paper_title: str,
    text: str,
    word_limit: int,
) -> str:
    return f"""Paper: {paper_title}
Section: {section_title}

Source text:
{text}

Write a technical summary that:
1. States the main contribution or finding of this section
2. Names every specific method, metric, result mentioned
3. Preserves ALL numerical values exactly as they appear
4. Uses the same technical terminology as the source
5. Does NOT introduce any information not in the source

Length: {word_limit} words maximum.

Summary:"""


def _build_combine_prompt(
    section_title: str,
    paper_title: str,
    partial_summaries: str,
    word_limit: int,
) -> str:
    return f"""Paper: {paper_title}
Section: {section_title}

Partial summaries to combine:
{partial_summaries}

Write one unified summary that:
1. Captures all key technical content
2. Removes redundancy
3. Preserves ALL numerical values exactly
4. Adds no new information

Length: {word_limit} words maximum.

Combined summary:"""


# ── Public summarization functions ─────────────────────────────────────────────

def summarize_l1_section(
    section_title: str,
    section_hierarchy: list[str],
    paper_title: str,
    chunk_texts: list[str],
) -> str:
    """
    Generate L1 section summary (synchronous entry point).
    Called by tree_builder for individual cluster summarization.
    """
    combined   = "\n\n".join(chunk_texts)
    word_limit = _dynamic_word_limit(combined)

    parts = _split_if_too_long(
        combined,
        max_tokens=config.SUMMARIZER_MAX_PART_TOKENS,
    )

    if len(parts) == 1:
        prompt = _build_l1_prompt(
            section_title, paper_title, combined, word_limit
        )
        return _call_llm(prompt)

    # Long section: summarize parts then combine
    logger.debug(
        f"L1 '{section_title}': {len(parts)} parts"
    )
    per_limit = max(80, word_limit // len(parts))
    part_summaries = []

    for part_idx, part in enumerate(parts):
        logger.debug(
            f"Summarizing part {part_idx+1}/{len(parts)}"
        )
        prompt = _build_l1_prompt(
            section_title, paper_title, part, per_limit
        )
        part_summaries.append(_call_llm(prompt))

    combined_parts = "\n\n".join(part_summaries)
    combine_prompt = _build_combine_prompt(
        section_title, paper_title, combined_parts, word_limit
    )
    return _call_llm(combine_prompt)


def summarize_l2_group(
    paper_title: str,
    l1_summary_texts: list[str],
    section_titles: list[str],
) -> str:
    """Generate L2 group summary."""
    combined = "\n\n---\n\n".join([
        f"Section: {t}\n{s}"
        for t, s in zip(section_titles, l1_summary_texts)
    ])
    word_limit = _dynamic_word_limit(combined)

    parts = _split_if_too_long(combined)
    if len(parts) > 1:
        intermediates = []
        for part in parts:
            prompt = (
                f"Paper: {paper_title}\n\n"
                f"Summarize these section summaries:\n{part}"
            )
            intermediates.append(_call_llm(prompt))
        combined = "\n\n".join(intermediates)

    prompt = f"""Paper: {paper_title}

Section summaries:
{combined}

Write a summary that:
1. Captures the overarching theme connecting these sections
2. Preserves key technical details and results
3. Contains ONLY information from the summaries above
4. Preserves ALL numerical values exactly

Length: {word_limit} words maximum.

Summary:"""

    return _call_llm(prompt)


def summarize_l3_root(
    paper_title: str,
    l2_summary_texts: list[str],
) -> str:
    """Generate L3 root summary. Always built."""
    combined = "\n\n---\n\n".join(l2_summary_texts)

    if count_tokens(combined) > config.SUMMARIZER_MAX_PART_TOKENS:
        parts = _split_if_too_long(combined)
        intermediates = []
        for part in parts:
            prompt = (
                f"Paper: {paper_title}\n\n"
                f"Summarize:\n{part}"
            )
            intermediates.append(_call_llm(prompt))
        combined = "\n\n".join(intermediates)

    prompt = f"""Paper: {paper_title}

Section group summaries:
{combined}

Write a comprehensive paper summary that:
1. States the core problem the paper solves
2. Describes the proposed approach technically
3. Lists main results with exact numbers preserved
4. States key conclusions and contributions
5. Contains ONLY information from the summaries above

Length: 250-350 words.

Paper summary:"""

    return _call_llm(prompt)


# ── Extractive fallback ────────────────────────────────────────────────────────

def extractive_fallback(chunk_texts: list[str]) -> str:
    """
    Pure extractive summary when ALL LLM calls fail.
    Zero hallucination risk. Always succeeds.
    Takes first + last sentence of each chunk.
    """
    _update_stats(extractive_calls=1)
    sentences = []

    for text in chunk_texts:
        if not text:
            continue
        parts = [s.strip() for s in text.split(".") if s.strip()]
        if len(parts) >= 2:
            sentences.append(parts[0] + ".")
            sentences.append(parts[-1] + ".")
        elif parts:
            sentences.append(parts[0] + ".")

    result = " ".join(sentences)
    logger.warning(
        f"Extractive fallback used "
        f"({count_tokens(result)} tokens)"
    )
    return result
# generation/llm_client.py
"""
Hardened LLM client for final answer generation.
Supports BYOK + system default keys + multi-key rotation.
Full fallback chain with retry logic.
"""

from __future__ import annotations
import os
import time
import random
import re
import hashlib
import litellm

from core.logger import get_logger
from core.exceptions import GenerationError
import config

logger         = get_logger(__name__)
litellm.suppress_debug_info = True

# ── Key rotation state ─────────────────────────────────────────────────────────
_key_rotation = {
    "gemini": 0,
    "groq":   0,
}


def _get_next_key(provider: str) -> str | None:
    """
    Round-robin key rotation for a provider.
    Returns next available key or None if no keys configured.
    """
    if provider == "gemini":
        keys = config.GEMINI_API_KEYS
    elif provider == "groq":
        keys = config.GROQ_API_KEYS
    else:
        return None

    if not keys:
        return None

    idx = _key_rotation[provider] % len(keys)
    _key_rotation[provider] = (idx + 1) % len(keys)
    return keys[idx]


def _set_api_key_for_model(model: str, override_key: str = ""):
    """
    Set the correct API key environment variable for a model.
    Handles key rotation for multi-key providers.
    """
    if override_key:
        # BYOK: use provided key directly
        if "gemini" in model:
            os.environ["GEMINI_API_KEY"] = override_key
        elif "gpt" in model or "openai" in model:
            os.environ["OPENAI_API_KEY"] = override_key
        elif "claude" in model or "anthropic" in model:
            os.environ["ANTHROPIC_API_KEY"] = override_key
        elif "groq" in model:
            os.environ["GROQ_API_KEY"] = override_key
        return

    # System keys with rotation
    if "gemini" in model:
        key = _get_next_key("gemini")
        if key:
            os.environ["GEMINI_API_KEY"] = key
    elif "groq" in model:
        key = _get_next_key("groq")
        if key:
            os.environ["GROQ_API_KEY"] = key
    elif "openrouter" in model:
        if config.OPENROUTER_API_KEY:
            os.environ["OPENROUTER_API_KEY"] = config.OPENROUTER_API_KEY


def _extract_retry_after(error_str: str) -> int:
    """Extract server-specified retry delay from error message."""
    patterns = [
        r'retryDelay["\s:]+(\d+)',
        r'retry.after["\s:]+(\d+)',
        r'retry in (\d+)',
        r'Retry-After["\s:]+(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, error_str, re.IGNORECASE)
        if match:
            return int(match.group(1)) + 2
    return 0


def _compute_backoff(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with jitter."""
    delay = min(cap, base * (2 ** attempt))
    return delay + random.uniform(0, 1)


def _is_retryable(error: Exception) -> tuple[bool, int]:
    """Determine if error is retryable and wait time."""
    error_str = str(error)

    if isinstance(error, litellm.RateLimitError):
        if "PerDay" in error_str or "per_day" in error_str.lower():
            return False, 0   # Daily limit — skip model
        server_wait = _extract_retry_after(error_str)
        return True, (server_wait if server_wait > 0 else 30)

    if isinstance(error, litellm.ServiceUnavailableError):
        return True, 15

    if isinstance(error, litellm.Timeout):
        return True, 5

    if isinstance(error, litellm.APIConnectionError):
        return True, 10

    if isinstance(error, litellm.AuthenticationError):
        return False, 0   # Wrong key — skip model

    if isinstance(error, litellm.BadRequestError):
        error_str_lower = error_str.lower()
        if any(x in error_str_lower for x in [
            "401", "invalid_api_key", "invalid api key",
            "authentication", "unauthorized"
        ]):
            return False, 0
        return False, 0

    if isinstance(error, litellm.ContextWindowExceededError):
        return False, 0   # Input too long — skip model

    if isinstance(error, litellm.APIError):
        code = getattr(error, 'status_code', 0)
        if code in (500, 502, 503, 504):
            return True, 10
        return False, 0

    return True, 5


def _call_single_model(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    override_key: str = "",
) -> str:
    """
    Call one specific model with retry logic.
    Raises GenerationError if all retries exhausted.
    """
    _set_api_key_for_model(model, override_key)

    for retry in range(config.GENERATION_MAX_RETRIES):
        try:
            response = litellm.completion(
                model       = model,
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = temperature,
                timeout     = config.GENERATION_REQUEST_TIMEOUT,
            )

            result = response.choices[0].message.content
            if not result or not result.strip():
                raise GenerationError(f"Model {model} returned empty response")

            logger.debug(
                f"Generation success: model={model} "
                f"tokens={getattr(response.usage, 'completion_tokens', '?')}"
            )
            return result.strip()

        except litellm.ContextWindowExceededError as e:
            raise GenerationError(
                f"Context too long for {model}: {e}"
            ) from e

        except litellm.AuthenticationError as e:
            raise GenerationError(
                f"Auth failed for {model}: {e}"
            ) from e

        except Exception as e:
            retryable, wait = _is_retryable(e)

            if not retryable:
                raise GenerationError(
                    f"Non-retryable error on {model}: "
                    f"{type(e).__name__}: {e}"
                ) from e

            if retry < config.GENERATION_MAX_RETRIES - 1:
                if wait == 0:
                    wait = _compute_backoff(
                        retry,
                        config.GENERATION_RETRY_BASE,
                        config.GENERATION_RETRY_MAX,
                    )
                logger.warning(
                    f"Generation retry {retry+1}/"
                    f"{config.GENERATION_MAX_RETRIES} "
                    f"model={model} "
                    f"error={type(e).__name__} "
                    f"wait={wait:.1f}s"
                )
                time.sleep(wait)
            else:
                raise GenerationError(
                    f"Model {model} failed after "
                    f"{config.GENERATION_MAX_RETRIES} retries: {e}"
                ) from e

    raise GenerationError(f"Model {model} exhausted retries")


def call_llm(
    messages: list[dict],
    max_tokens: int = None,
    temperature: float = None,
    byok_model: str = "",
    byok_key: str = "",
) -> str:
    """
    Main LLM call with full fallback chain.

    Priority:
    1. BYOK model + key (user provided)
    2. System primary model + key rotation
    3. Fallback chain models

    Returns generated text string.
    Raises GenerationError only if ALL models fail.
    """
    max_tokens  = max_tokens or config.GENERATION_MAX_TOKENS
    temperature = temperature or config.GENERATION_TEMPERATURE

    # Build model chain
    if byok_model and byok_key:
        # BYOK: user's model goes first
        model_chain = [(byok_model, byok_key)]
        # Add system fallbacks without BYOK key
        model_chain += [
            (m, "") for m in [config.GENERATION_PRIMARY_MODEL]
            + config.GENERATION_FALLBACK_MODELS
        ]
    else:
        # System keys only
        model_chain = [
            (m, "") for m in [config.GENERATION_PRIMARY_MODEL]
            + config.GENERATION_FALLBACK_MODELS
        ]

    last_error = None

    for model_idx, (model, key) in enumerate(model_chain):
        if not model:
            continue

        is_fallback = model_idx > 0
        if is_fallback:
            logger.warning(
                f"Generation fallback {model_idx}: {model}"
            )

        try:
            result = _call_single_model(
                model       = model,
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = temperature,
                override_key = key,
            )

            if is_fallback:
                logger.info(f"Fallback succeeded: {model}")
            return result

        except GenerationError as e:
            last_error = e
            logger.warning(f"Model {model} failed: {e}")
            continue

    raise GenerationError(
        f"All models in fallback chain failed. "
        f"Last error: {last_error}"
    )
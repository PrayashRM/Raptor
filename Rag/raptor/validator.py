# raptor/validator.py
"""
Validates generated summaries before they enter the index.

Two checks:
1. Compression: summary must be meaningfully shorter than input
2. Hallucination: summary must not contain numbers absent from source

On failure: retry once with stricter prompt.
On second failure: extractive fallback (no LLM, no hallucination risk).
"""

from __future__ import annotations
from core.logger import get_logger
from core.utils import count_tokens, extract_numbers
import config

logger = get_logger(__name__)


def validate_summary(
    summary: str,
    source_text: str,
    chunk_id_for_logging: str = "",
) -> tuple[bool, str]:
    """
    Validate a generated summary.

    Returns:
        (True, reason)  if valid
        (False, reason) if invalid
    """
    if not summary or not summary.strip():
        return False, "EMPTY_SUMMARY"

    input_tokens   = count_tokens(source_text)
    summary_tokens = count_tokens(summary)

    # ── Check 1: Compression ──────────────────────────────────────────────
    if input_tokens > 50:  # only check if input is substantial
        compression_ratio = summary_tokens / max(input_tokens, 1)
        if compression_ratio > config.SUMMARY_COMPRESSION_MAX:
            return False, (
                f"FAILED_COMPRESSION: ratio={compression_ratio:.2f} "
                f"(max={config.SUMMARY_COMPRESSION_MAX})"
            )

    # ── Check 2: Number hallucination ─────────────────────────────────────
    source_numbers  = extract_numbers(source_text)
    summary_numbers = extract_numbers(summary)
    hallucinated    = summary_numbers - source_numbers

    # Allow small numbers (1,2,3..10) as they appear everywhere
    # Only flag numbers > 10 that dont appear in source
    significant_hallucinations = {
        n for n in hallucinated
        if float(n) > 10
    }

    if significant_hallucinations:
        return False, (
            f"FAILED_HALLUCINATION: numbers in summary not in source: "
            f"{significant_hallucinations}"
        )

    return True, "PASSED"


def extractive_fallback(chunk_texts: list[str]) -> str:
    """
    Extractive summary when LLM validation fails twice.
    Takes first and last sentence of each chunk.
    Zero hallucination risk. Always succeeds.
    """
    sentences = []
    for text in chunk_texts:
        if not text:
            continue
        parts = [s.strip() for s in text.split(".") if s.strip()]
        if len(parts) >= 2:
            sentences.append(parts[0] + ".")
            sentences.append(parts[-1] + ".")
        elif len(parts) == 1:
            sentences.append(parts[0] + ".")

    result = " ".join(sentences)
    logger.warning(
        f"Using extractive fallback summary "
        f"({count_tokens(result)} tokens)"
    )
    return result
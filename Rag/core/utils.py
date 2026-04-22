# core/utils.py
import re
import hashlib
from typing import Optional
from core.logger import get_logger

logger = get_logger(__name__)


def count_tokens(text: str) -> int:
    """
    Fast approximation: 1 token ≈ 4 characters.
    Accurate enough for chunking decisions.
    Not used for model context window management
    where tiktoken would be more precise.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def hash_text(text: str) -> str:
    """
    SHA256 hash of text for content-based deduplication.
    Used on text_for_embedding field only.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """
    Minimal cleaning for text_for_embedding.
    Removes control characters and normalizes whitespace.
    Does NOT remove punctuation or lowercase —
    SPLADE and nomic handle that internally.
    """
    if not text:
        return ""
    # Remove control characters except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize multiple whitespace to single space
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple newlines to double newline max
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_numbers(text: str) -> set:
    """
    Extract all numeric values from text.
    Used for hallucination detection in summary validation.
    Returns set of strings to allow exact match comparison.
    """
    pattern = r"\b\d+\.?\d*\b"
    return set(re.findall(pattern, text))


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences for soft-split boundary detection.
    Simple regex approach — accurate enough for academic text.
    """
    # Split on period/exclamation/question followed by space and capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def find_sentence_boundary(text: str, target_tokens: int) -> int:
    """
    Find character index of nearest sentence boundary
    at or before target_tokens worth of text.

    Returns:
        len(text)  — if entire text fits within target
        index > 0  — character index to split at
        -1         — no good boundary found within text
    """
    target_chars = target_tokens * 4

    # Entire text fits within target, no split needed
    if len(text) <= target_chars:
        return len(text)

    # Search in a window around the target
    search_end    = min(len(text), target_chars + 300)
    search_region = text[:search_end]
    sentences     = split_into_sentences(search_region)

    if not sentences:
        return -1

    accumulated   = 0
    best_boundary = -1

    for sentence in sentences:
        sentence_len = len(sentence)
        next_pos     = accumulated + sentence_len

        if next_pos <= target_chars + 150:
            best_boundary = next_pos
            accumulated   = next_pos + 1  # +1 for space, only for tracking
                                           # NOT added to best_boundary
        else:
            break

    # Clamp to actual text length, never exceed it
    if best_boundary > len(text):
        best_boundary = len(text)

    return best_boundary


def normalize_section_title(title: str) -> str:
    """
    Normalize section title for lookup in treatment maps.
    Strips numbering and lowercases for matching.

    Example:
        "6.2 Model Variations" -> "model variations"
        "3. Methods" -> "methods"
    """
    # Remove leading numbers and dots
    title = re.sub(r"^\d+[\.\d]*\s*", "", title)
    return title.strip().lower()
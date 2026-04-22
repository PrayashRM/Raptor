# ingestion/section_filter.py
"""
Determines treatment for every element based on its section.
Assigns: skip | metadata_only | process_high | process_medium | process_low
"""

from core.models import ParsedElement
from core.logger import get_logger
from core.utils import normalize_section_title
import config

logger = get_logger(__name__)

# Inline section markers that may appear without a formal header
INLINE_SKIP_MARKERS = {
    "acknowledgements",
    "acknowledgments",
    "acknowledgement",
    "acknowledgment",
}


def get_section_treatment(element: ParsedElement) -> str:
    """
    Returns treatment string for a given element based on section title.

    Returns:
        "skip"           — discard entirely
        "metadata_only"  — store at document level, never embed
        "process_high"   — embed, index, boost
        "process_medium" — embed, index, normal weight
        "process_low"    — embed, index, slight penalty
    """
    raw_title = element.section.title
    normalized = normalize_section_title(raw_title)

    # Check for inline acknowledgements buried inside another section
    # (common pattern: appears at end of Conclusion without its own header)
    if element.type == "text" and element.content.raw:
        first_word = element.content.raw.strip().split()[0].lower() \
            if element.content.raw.strip() else ""
        first_two = " ".join(
            element.content.raw.strip().lower().split()[:2]
        )
        if first_word in INLINE_SKIP_MARKERS \
                or first_two in INLINE_SKIP_MARKERS:
            logger.debug(
                f"Inline acknowledgement detected in {element.element_id}, "
                f"marking skip"
            )
            return "skip"

    # References — metadata only (store list, never embed)
    if normalized in {"references", "bibliography"}:
        return "metadata_only"

    # Hard skips
    for skip_section in config.SECTIONS_TO_SKIP:
        if normalized == skip_section or normalized.startswith(skip_section):
            return "skip"

    # High importance
    for high_section in config.SECTIONS_HIGH:
        if normalized == high_section or normalized.startswith(high_section):
            return "process_high"

    # Low importance
    for low_section in config.SECTIONS_LOW:
        if normalized == low_section or normalized.startswith(low_section):
            return "process_low"

    # Default: medium
    # Better to embed something useless than skip something useful
    logger.debug(
        f"Unknown section '{raw_title}' normalized to '{normalized}', "
        f"defaulting to process_medium"
    )
    return "process_medium"


def should_embed(treatment: str) -> bool:
    return treatment in {"process_high", "process_medium", "process_low"}


def get_importance_from_treatment(treatment: str) -> str:
    mapping = {
        "process_high":   "high",
        "process_medium": "medium",
        "process_low":    "low",
        "skip":           "low",
        "metadata_only":  "low",
    }
    return mapping.get(treatment, "medium")


def get_boost_from_treatment(treatment: str) -> float:
    mapping = {
        "process_high":   1.10,
        "process_medium": 1.00,
        "process_low":    0.95,
        "skip":           0.0,
        "metadata_only":  0.0,
    }
    return mapping.get(treatment, 1.00)


def get_query_affinities(element: ParsedElement, treatment: str) -> list[str]:
    """
    Assign query affinity tags based on element content type and level.
    These are used at retrieval time for mode routing.
    """
    affinities = []

    if element.type == "table":
        affinities.append("table")
        affinities.append("specific")
    elif element.type == "equation":
        affinities.append("equation")
        affinities.append("specific")
    elif element.type == "figure":
        affinities.append("figure")
    else:
        # Text: broad vs specific depends on section
        if treatment == "process_high":
            normalized = normalize_section_title(element.section.title)
            broad_sections = {
                "abstract", "introduction", "conclusion",
                "discussion", "related work", "background"
            }
            if any(b in normalized for b in broad_sections):
                affinities.append("broad")
            else:
                affinities.append("specific")
        else:
            affinities.append("broad")

    return affinities if affinities else ["broad"]
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


# ingestion/section_filter.py
"""
Determines treatment for every element based on its section.
"""

from core.models import ParsedElement
from core.logger import get_logger
from core.utils import normalize_section_title
import config
import re

logger = get_logger(__name__)

# Inline section markers that appear without a formal header
INLINE_SKIP_MARKERS = {
    "acknowledgements",
    "acknowledgments",
    "acknowledgement",
    "acknowledgment",
}

# Footnote detection patterns
# Footnotes start with a number marker at the beginning
FOOTNOTE_START_PATTERN = re.compile(
    r'^\s*\d+\s+[A-Z]',  # "5 We used values..."
)

# Footnote anchor pattern in text (superscript reference)
FOOTNOTE_ANCHOR_PATTERN = re.compile(
    r'\b\d+\s*$'  # ends with a number (footnote marker)
)


def detect_element_subtype(element: ParsedElement) -> str:
    """
    Detect if an element is actually a footnote
    misclassified as text by the parser.

    Returns:
        "footnote" if detected as footnote
        "text"     if normal text
        "acknowledgement" if inline acknowledgement
    """
    if not element.content.raw:
        return "text"

    raw = element.content.raw.strip()

    # Check for inline acknowledgement
    first_word  = raw.split()[0].lower() if raw.split() else ""
    first_two   = " ".join(raw.lower().split()[:2])
    if first_word in INLINE_SKIP_MARKERS \
            or first_two in INLINE_SKIP_MARKERS:
        return "acknowledgement"

    # Check for footnote pattern
    # Footnotes: start with digit + space + capital letter
    # e.g., "5 We used values of 2.8, 3.7..."
    if FOOTNOTE_START_PATTERN.match(raw):
        # Additional check: token count is small (footnotes are short)
        token_count = element.content.token_count or 0
        if token_count < 60:
            return "footnote"

    return "text"


def get_section_treatment(element: ParsedElement) -> str:
    """
    Returns treatment string for a given element.

    Returns:
        "skip"           — discard entirely
        "metadata_only"  — store at document level, never embed
        "process_high"   — embed, index, boost
        "process_medium" — embed, index, normal weight
        "process_low"    — embed, index, slight penalty
    """
    raw_title  = element.section.title
    normalized = normalize_section_title(raw_title)

    # Detect misclassified element subtypes
    subtype = detect_element_subtype(element)

    if subtype == "acknowledgement":
        logger.debug(
            f"Inline acknowledgement detected in "
            f"{element.element_id}, marking skip"
        )
        return "skip"

    if subtype == "footnote":
        # Footnotes are processed but merged with anchor
        # They get the same treatment as their parent section
        logger.debug(
            f"Footnote detected in {element.element_id}"
        )
        # Fall through to section-based treatment
        # The chunker will handle merging with adjacent text

    # References — metadata only
    if normalized in {"references", "bibliography"}:
        return "metadata_only"

    # Hard skips
    for skip_section in config.SECTIONS_TO_SKIP:
        if normalized == skip_section \
                or normalized.startswith(skip_section):
            return "skip"

    # High importance
    for high_section in config.SECTIONS_HIGH:
        if normalized == high_section \
                or normalized.startswith(high_section):
            return "process_high"

    # Low importance
    for low_section in config.SECTIONS_LOW:
        if normalized == low_section \
                or normalized.startswith(low_section):
            return "process_low"

    # Default: medium
    logger.debug(
        f"Unknown section '{raw_title}' -> "
        f"'{normalized}', defaulting to process_medium"
    )
    return "process_medium"


def should_embed(treatment: str) -> bool:
    return treatment in {
        "process_high", "process_medium", "process_low"
    }


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


def get_query_affinities(
    element: ParsedElement,
    treatment: str,
) -> list[str]:
    """
    Assign query affinity tags based on element content type.
    Used at retrieval time for mode routing.
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
        if treatment == "process_high":
            normalized = normalize_section_title(
                element.section.title
            )
            broad_sections = {
                "abstract", "introduction", "conclusion",
                "discussion", "related work", "background",
                "contributions",
            }
            if any(b in normalized for b in broad_sections):
                affinities.append("broad")
            else:
                affinities.append("specific")
        else:
            affinities.append("broad")

    return affinities if affinities else ["broad"]

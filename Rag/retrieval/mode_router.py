# retrieval/mode_router.py
"""
Retrieval mode selection based on query signals.

Four modes:
  MODE_A: Specific fact    → leaf nodes only, sparse-heavy
  MODE_B: Broad concept    → summaries first, dense-heavy
  MODE_C: Balanced default → all levels, equal weights
  MODE_D: Modality         → filtered by has_table/equation/figure

No LLM needed. Pure rule-based decision.
Deterministic and debuggable.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from retrieval.signal_extractor import QuerySignals
from core.logger import get_logger

logger = get_logger(__name__)


class RetrievalMode(Enum):
    A = "specific_leaf_focus"
    B = "broad_summary_first"
    C = "balanced_default"
    D = "modality_specific"


@dataclass
class RetrievalConfig:
    """Complete retrieval configuration for one query."""
    mode:           RetrievalMode
    dense_weight:   float
    sparse_weight:  float
    top_k:          int
    level_filter:   list[int] | None   # None = all levels
    modality_filter: dict | None       # None = no filter
    boost_key_sections: bool
    rerank_top_n:   int


def route_query(signals: QuerySignals) -> RetrievalConfig:
    """
    Select retrieval mode and configuration based on signals.

    Priority order for conflict resolution:
    1. Modality signal (most specific, always wins)
    2. Broad dominates (cross-section or broad_score >> specific)
    3. Specific dominates
    4. Balanced default (fallback)
    """

    # ── Priority 1: Modality always wins ──────────────────────────────────
    if signals.table_signal or signals.equation_signal \
            or signals.figure_signal:

        modality_filter = {}
        if signals.table_signal:
            modality_filter["has_table"] = True
        if signals.equation_signal:
            modality_filter["has_equation"] = True
        if signals.figure_signal:
            modality_filter["has_figure"] = True

        config = RetrievalConfig(
            mode             = RetrievalMode.D,
            dense_weight     = 0.50,
            sparse_weight    = 0.50,
            top_k            = 8,
            level_filter     = None,
            modality_filter  = modality_filter,
            boost_key_sections = True,
            rerank_top_n     = 4,
        )
        logger.debug(f"Mode D selected: {modality_filter}")
        return config

    # ── Priority 2: Broad / cross-section ─────────────────────────────────
    if signals.cross_section or (
        signals.broad_score > signals.specific_score + 1
    ):
        config = RetrievalConfig(
            mode             = RetrievalMode.B,
            dense_weight     = 0.65,
            sparse_weight    = 0.35,
            top_k            = 6,
            level_filter     = [1, 2, 3],  # summaries only first
            modality_filter  = None,
            boost_key_sections = True,
            rerank_top_n     = 4,
        )
        logger.debug("Mode B selected: broad/cross-section query")
        return config

    # ── Priority 3: Specific fact ──────────────────────────────────────────
    if signals.specific_score > signals.broad_score + 1 \
            or signals.has_number:
        config = RetrievalConfig(
            mode             = RetrievalMode.A,
            dense_weight     = 0.40,
            sparse_weight    = 0.60,
            top_k            = 8,
            level_filter     = [0],        # leaf nodes only
            modality_filter  = None,
            boost_key_sections = False,
            rerank_top_n     = 4,
        )
        logger.debug("Mode A selected: specific fact query")
        return config

    # ── Priority 4: Balanced default ──────────────────────────────────────
    config = RetrievalConfig(
        mode             = RetrievalMode.C,
        dense_weight     = 0.50,
        sparse_weight    = 0.50,
        top_k            = 12,
        level_filter     = None,           # all levels
        modality_filter  = None,
        boost_key_sections = True,
        rerank_top_n     = 5,
    )
    logger.debug("Mode C selected: balanced default")
    return config
# retrieval/signal_extractor.py
"""
Rule-based query signal extraction.
Zero LLM calls. Zero ML. Pure string matching.
Runs in <5ms on any hardware.

Extracts three signals from query text:
1. Specificity: specific fact vs broad concept
2. Modality:    table / equation / figure referenced
3. Scope:       cross-section / whole-paper query
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QuerySignals:
    """All signals extracted from a query string."""
    raw_query:        str
    specific_score:   int
    broad_score:      int
    table_signal:     bool
    equation_signal:  bool
    figure_signal:    bool
    cross_section:    bool
    has_number:       bool


# ── Signal indicator lists ─────────────────────────────────────────────────────

SPECIFIC_INDICATORS = [
    # Number hunting
    r'\b\d+\.?\d*\b',
    'how many', 'what is the value', 'exact', 'specific',
    'what was', 'which number', 'what score', 'what rate',
    'how much', 'how long', 'how large', 'how fast',
    # Result hunting
    'table', 'figure', 'result', 'score', 'metric',
    'benchmark', 'performance', 'accuracy', 'bleu',
    'perplexity', 'loss', 'precision', 'recall', 'f1',
    'wmt', 'newstest', 'dataset',
    # Implementation detail hunting
    'learning rate', 'batch size', 'epoch', 'step',
    'dropout', 'regularization', 'weight decay', 'warmup', 'optimizer', 'adam', 'gpu',
    'parameter', 'layer', 'head', 'dimension', 'size', 'weight decay',
    'p_drop', 'label smoothing', 'checkpoint',
    'warmup steps', 'beta', 'epsilon',
]

BROAD_INDICATORS = [
    'why', 'how does', 'what is the purpose',
    'explain', 'describe', 'overview', 'concept',
    'motivation', 'advantage', 'benefit', 'intuition',
    'difference between', 'what makes', 'what approach',
    'summarize', 'what is', 'how does this work',
    'what problem', 'contribution', 'novel', 'propose',
    'main idea', 'key insight', 'compare', 'versus',
    'what problem', 'what challenge', 'what limitation',
    'solve', 'address', 'tackle', 'overcome', '', '', '',
    'what does', 'how is', 'in what way',
    'what role', 'what purpose', 'what effect', 'what impact',
]

TABLE_INDICATORS = [
    'table', 'comparison', 'versus', 'vs',
    'compared to', 'results show', 'in table',
    'performance of', 'benchmark', 'leaderboard',
    'outperforms', 'beats', 'achieves',
]

EQUATION_INDICATORS = [
    'formula', 'equation', 'calculate', 'compute',
    'loss function', 'learning rate formula', 'objective',
    'mathematically', 'expression', 'defined as',
    'softmax', 'attention formula', 'function',
]

FIGURE_INDICATORS = [
    'figure', 'diagram', 'architecture diagram',
    'visualization', 'plot', 'graph', 'shown in',
    'illustrated', 'depicted', 'image', 'picture',
]

CROSS_SECTION_INDICATORS = [
    'throughout', 'overall', 'in general', 'the paper',
    'main contribution', 'key idea', 'approach',
    'the method', 'the model', 'the system', 'proposed',
    'summarize the paper', 'what does this paper',
    'entire paper', 'whole paper', 'this work',
]

NUMBER_PATTERN = re.compile(r'\b\d+\.?\d*\b')


def extract_signals(query: str) -> QuerySignals:
    """
    Extract all retrieval signals from a query string.

    Args:
        query: Raw user query string

    Returns:
        QuerySignals dataclass with all extracted signals
    """
    query_lower = query.lower().strip()

    # ── Specificity score ──────────────────────────────────────────────────
    specific_score = 0
    for indicator in SPECIFIC_INDICATORS:
        if indicator.startswith(r'\b'):
            # Regex pattern
            if re.search(indicator, query_lower):
                specific_score += 1
        elif indicator in query_lower:
            specific_score += 1

    # ── Broad score ────────────────────────────────────────────────────────
    broad_score = 0
    for indicator in BROAD_INDICATORS:
        if indicator in query_lower:
            broad_score += 1

    # ── Modality signals ───────────────────────────────────────────────────
    table_signal    = any(ind in query_lower for ind in TABLE_INDICATORS)
    equation_signal = any(ind in query_lower for ind in EQUATION_INDICATORS)
    figure_signal   = any(ind in query_lower for ind in FIGURE_INDICATORS)

    # ── Cross-section signal ───────────────────────────────────────────────
    cross_section = any(ind in query_lower for ind in CROSS_SECTION_INDICATORS)

    # ── Number detection ───────────────────────────────────────────────────
    has_number = bool(NUMBER_PATTERN.search(query_lower))

    signals = QuerySignals(
        raw_query       = query,
        specific_score  = specific_score,
        broad_score     = broad_score,
        table_signal    = table_signal,
        equation_signal = equation_signal,
        figure_signal   = figure_signal,
        cross_section   = cross_section,
        has_number      = has_number,
    )

    logger.debug(
        f"Query signals: specific={specific_score} "
        f"broad={broad_score} "
        f"table={table_signal} eq={equation_signal} "
        f"figure={figure_signal} cross={cross_section}"
    )

    return signals
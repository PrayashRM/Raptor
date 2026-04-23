# generation/context_builder.py
"""
Builds the final LLM context from retrieved chunks.
Handles token budget, priority ordering, and
formatting of text, tables, equations, and figures.
"""

from __future__ import annotations
from dataclasses import dataclass
from core.logger import get_logger
from core.utils import count_tokens
import config

logger = get_logger(__name__)


@dataclass
class BuiltContext:
    """Result of context building."""
    context_text:     str         # formatted context for LLM
    section_list:     list[str]   # section names in context (for citation)
    total_tokens:     int
    chunks_included:  int
    chunks_dropped:   int
    has_tables:       bool
    has_equations:    bool
    has_figures:      bool
    image_paths:      list[str]   # for VLM if needed
    equation_latex:   list[str]   # for explicit equation display


def build_context(
    chunks: list[dict],
    model: str = "",
    system_prompt_tokens: int = 300,
    query_tokens: int = 50,
) -> BuiltContext:
    """
    Build formatted LLM context from retrieved chunks.

    Args:
        chunks:               Retrieved chunks sorted by reading order
        model:                Target model (for context limit lookup)
        system_prompt_tokens: Reserved for system prompt
        query_tokens:         Reserved for user query

    Returns:
        BuiltContext with formatted text and metadata
    """
    # ── Determine token budget ─────────────────────────────────────────────
    model_key   = model if model in config.GENERATION_MAX_CONTEXT_TOKENS \
                  else "default"
    max_context = config.GENERATION_MAX_CONTEXT_TOKENS[model_key]

    generation_reserve = config.GENERATION_CONTEXT_RESERVE
    available_tokens   = (
        max_context
        - system_prompt_tokens
        - query_tokens
        - generation_reserve
    )

    logger.debug(
        f"Context budget: {available_tokens} tokens available "
        f"(model={model_key} max={max_context})"
    )

    # ── Priority ordering ──────────────────────────────────────────────────
    # 1. Direct retrieval hits with high rerank score
    # 2. Key section chunks
    # 3. Traversal-fetched parents/children
    def priority_key(chunk: dict) -> tuple:
        rerank = chunk.get("_rerank_score", 0.0)
        is_key = chunk.get("is_key_section", False)
        level  = chunk.get("level", 0)
        fetched_as = chunk.get("_fetched_as", "direct")
        is_direct  = fetched_as == "direct"
        return (
            is_direct,      # direct hits first
            is_key,         # key sections boosted
            rerank,         # then by rerank score
            level == 0,     # prefer leaves (specific)
        )

    sorted_chunks = sorted(chunks, key=priority_key, reverse=True)

    # ── Fill context within budget ─────────────────────────────────────────
    included       = []
    used_tokens    = 0
    dropped        = 0
    section_set    = set()
    image_paths    = []
    equation_latex = []
    has_tables     = False
    has_equations  = False
    has_figures    = False

    for chunk in sorted_chunks:
        chunk_tokens = chunk.get("token_count", 0)
        if chunk_tokens == 0:
            text         = chunk.get("text_for_display", "")
            chunk_tokens = count_tokens(text)

        if used_tokens + chunk_tokens > available_tokens:
            dropped += 1
            logger.debug(
                f"Dropped chunk {chunk.get('chunk_id')} "
                f"({chunk_tokens} tokens, budget exhausted)"
            )
            continue

        included.append(chunk)
        used_tokens += chunk_tokens

        # Track metadata
        section = chunk.get("section_title", "")
        if section:
            section_set.add(section)

        # Collect multimodal paths
        for path in chunk.get("table_image_paths", []):
            if path and path not in image_paths:
                image_paths.append(path)
                has_tables = True

        for path in chunk.get("equation_image_paths", []):
            if path and path not in image_paths:
                image_paths.append(path)
                has_equations = True

        for path in chunk.get("figure_image_paths", []):
            if path and path not in image_paths:
                image_paths.append(path)
                has_figures = True

        for latex in chunk.get("equation_latex", []):
            if latex and latex not in equation_latex:
                equation_latex.append(latex)

        if chunk.get("has_table"):
            has_tables = True
        if chunk.get("has_equation"):
            has_equations = True
        if chunk.get("has_figure"):
            has_figures = True

    # Sort included chunks by reading order for coherence
    included.sort(
        key=lambda x: x.get("chunk_sequence_index", 0)
    )

    # ── Format context text ────────────────────────────────────────────────
    context_text = _format_context(included, equation_latex)

    section_list = sorted(section_set)

    logger.info(
        f"Context built: {len(included)} chunks / "
        f"{used_tokens} tokens / "
        f"{dropped} dropped"
    )

    return BuiltContext(
        context_text    = context_text,
        section_list    = section_list,
        total_tokens    = used_tokens,
        chunks_included = len(included),
        chunks_dropped  = dropped,
        has_tables      = has_tables,
        has_equations   = has_equations,
        has_figures     = has_figures,
        image_paths     = image_paths,
        equation_latex  = equation_latex,
    )


def _format_context(
    chunks: list[dict],
    equation_latex: list[str],
) -> str:
    """
    Format chunks into readable context for LLM.
    Preserves markdown tables and equations.
    Adds section labels for citation.
    """
    parts = []

    for chunk in chunks:
        level   = chunk.get("level", 0)
        section = chunk.get("section_title", "Unknown Section")
        text    = chunk.get("text_for_display", "")

        if not text:
            continue

        # Level label for LLM awareness
        level_label = {
            0: "SOURCE",
            1: "SECTION SUMMARY",
            2: "GROUP SUMMARY",
            3: "PAPER OVERVIEW",
        }.get(level, "SOURCE")

        # Format chunk block
        block = (
            f"--- {level_label}: {section} ---\n"
            f"{text}"
        )

        # Append equation latex explicitly if present
        chunk_latex = chunk.get("equation_latex", [])
        if chunk_latex:
            latex_block = "\n".join(
                f"Equation: $${eq}$$" for eq in chunk_latex
            )
            block = f"{block}\n{latex_block}"

        parts.append(block)

    context = "\n\n".join(parts)

    return context
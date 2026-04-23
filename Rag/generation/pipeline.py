# generation/pipeline.py
"""
Phase 6 orchestrator: retrieval results -> final answer.

Flow:
1. Build context from retrieved chunks (token budget)
2. Build prompt messages (system + user)
3. Call LLM with fallback chain (BYOK or system keys)
4. Return structured GenerationResult
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.logger import get_logger
from core.exceptions import GenerationError
from core.utils import count_tokens
from retrieval.pipeline import RetrievalResult, retrieve
from generation.context_builder import build_context, BuiltContext
from generation.prompt_builder import build_prompt_messages
from generation.llm_client import call_llm
import config

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Complete result from Phase 6 generation."""
    query:            str
    paper_id:         str
    paper_title:      str
    answer:           str
    retrieval_mode:   str
    chunks_retrieved: int
    chunks_in_context: int
    context_tokens:   int
    model_used:       str
    has_tables:       bool
    has_equations:    bool
    has_figures:      bool
    image_paths:      list[str]
    equation_latex:   list[str]
    section_list:     list[str]
    generated_at:     str
    error:            str = ""


def generate_answer(
    query: str,
    paper_id: str,
    paper_title: str = "",
    byok_model: str = "",
    byok_key: str = "",
) -> GenerationResult:
    """
    Full Phase 6 pipeline: query -> answer.

    Args:
        query:       User question
        paper_id:    Paper to query against
        paper_title: Human readable paper title
        byok_model:  User's own model (optional)
        byok_key:    User's own API key (optional)

    Returns:
        GenerationResult with answer and full metadata
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"Generation start: query='{query[:60]}' "
        f"paper='{paper_id}'"
    )

    # ── Step 1: Retrieve context ───────────────────────────────────────────
    try:
        retrieval_result: RetrievalResult = retrieve(
            query    = query,
            paper_id = paper_id,
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return _error_result(
            query        = query,
            paper_id     = paper_id,
            paper_title  = paper_title,
            generated_at = generated_at,
            error        = f"Retrieval failed: {e}",
        )

    if not retrieval_result.final_chunks:
        logger.warning("No chunks retrieved, returning no-context answer")
        return GenerationResult(
            query             = query,
            paper_id          = paper_id,
            paper_title       = paper_title,
            answer            = (
                "I could not find relevant information in this paper "
                "to answer your question. Please try rephrasing."
            ),
            retrieval_mode    = retrieval_result.mode,
            chunks_retrieved  = 0,
            chunks_in_context = 0,
            context_tokens    = 0,
            model_used        = "",
            has_tables        = False,
            has_equations     = False,
            has_figures       = False,
            image_paths       = [],
            equation_latex    = [],
            section_list      = [],
            generated_at      = generated_at,
        )

    # ── Step 2: Build context ──────────────────────────────────────────────
    # Determine active model for context limit calculation
    active_model = byok_model or config.GENERATION_PRIMARY_MODEL

    try:
        context: BuiltContext = build_context(
            chunks               = retrieval_result.final_chunks,
            model                = active_model,
            system_prompt_tokens = count_tokens(
                "You are a precise research paper question-answering assistant."
            ),
            query_tokens         = count_tokens(query),
        )
    except Exception as e:
        logger.error(f"Context building failed: {e}")
        return _error_result(
            query, paper_id, paper_title, generated_at,
            f"Context building failed: {e}"
        )

    # ── Step 3: Build prompt ───────────────────────────────────────────────
    messages = build_prompt_messages(
        query       = query,
        context     = context,
        signals     = retrieval_result.signals,
        paper_title = paper_title or paper_id,
    )

    # ── Step 4: Generate answer ────────────────────────────────────────────
    model_used = ""
    try:
        answer = call_llm(
            messages    = messages,
            byok_model  = byok_model,
            byok_key    = byok_key,
        )
        # Determine which model was actually used
        model_used = byok_model or config.GENERATION_PRIMARY_MODEL

    except GenerationError as e:
        logger.error(f"All LLM models failed: {e}")
        # Last resort: return context directly without generation
        answer = (
            "Generation failed due to API issues. "
            "Here is the raw relevant context:\n\n"
            + context.context_text[:2000]
        )
        model_used = "fallback_raw_context"

    # ── Step 5: Build result ───────────────────────────────────────────────
    result = GenerationResult(
        query             = query,
        paper_id          = paper_id,
        paper_title       = paper_title or paper_id,
        answer            = answer,
        retrieval_mode    = retrieval_result.mode,
        chunks_retrieved  = retrieval_result.total_candidates,
        chunks_in_context = context.chunks_included,
        context_tokens    = context.total_tokens,
        model_used        = model_used,
        has_tables        = context.has_tables,
        has_equations     = context.has_equations,
        has_figures       = context.has_figures,
        image_paths       = context.image_paths,
        equation_latex    = context.equation_latex,
        section_list      = context.section_list,
        generated_at      = generated_at,
    )

    logger.info(
        f"Generation complete: "
        f"mode={result.retrieval_mode} "
        f"chunks={result.chunks_in_context} "
        f"tokens={result.context_tokens} "
        f"model={result.model_used}"
    )

    return result


def _error_result(
    query: str,
    paper_id: str,
    paper_title: str,
    generated_at: str,
    error: str,
) -> GenerationResult:
    """Build a failed GenerationResult."""
    return GenerationResult(
        query             = query,
        paper_id          = paper_id,
        paper_title       = paper_title,
        answer            = (
            f"An error occurred while processing your query. "
            f"Please try again."
        ),
        retrieval_mode    = "error",
        chunks_retrieved  = 0,
        chunks_in_context = 0,
        context_tokens    = 0,
        model_used        = "",
        has_tables        = False,
        has_equations     = False,
        has_figures       = False,
        image_paths       = [],
        equation_latex    = [],
        section_list      = [],
        generated_at      = generated_at,
        error             = error,
    )
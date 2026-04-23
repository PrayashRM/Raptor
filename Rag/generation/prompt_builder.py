# generation/prompt_builder.py
"""
Builds system and user prompts for final answer generation.
Section-cited answers. Preserves numerical values.
Handles different query types with appropriate instructions.
"""

from __future__ import annotations
from retrieval.signal_extractor import QuerySignals
from generation.context_builder import BuiltContext
from core.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a precise research paper question-answering assistant.
You answer questions strictly from the provided context sections.

RULES:
1. Answer ONLY from the provided context. Never use external knowledge.
2. Cite the section name when stating a fact.
   Use ONLY section names from the "Available sections" list.
3. Preserve ALL numerical values exactly as they appear in the context.
4. If the answer is not in the context, say:
   "This information is not available in the provided sections."
   Do NOT guess or fabricate.
5. If a table is referenced in your answer, note it explicitly.
6. If an equation is relevant, include it using LaTeX formatting.
7. Format your answer in clear markdown.
8. Be concise but complete. Do not pad your answer."""


def build_prompt_messages(
    query: str,
    context: BuiltContext,
    signals: QuerySignals,
    paper_title: str,
) -> list[dict]:
    """
    Build the complete message list for LLM generation.

    Returns:
        List of message dicts: [system, user]
    """
    # Build user message
    user_message = _build_user_message(
        query        = query,
        context      = context,
        signals      = signals,
        paper_title  = paper_title,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


def _build_user_message(
    query: str,
    context: BuiltContext,
    signals: QuerySignals,
    paper_title: str,
) -> str:
    """Build the user turn message with context injected."""

    # Section list for citation grounding
    section_list_str = "\n".join(
        f"  - {s}" for s in context.section_list
    ) if context.section_list else "  - (no sections available)"

    # Modality notices
    modality_notices = []
    if context.has_tables:
        modality_notices.append(
            "Note: This context contains tables. "
            "Reference them explicitly if relevant."
        )
    if context.has_equations:
        modality_notices.append(
            "Note: This context contains equations in LaTeX format."
        )
    if context.has_figures:
        modality_notices.append(
            "Note: This context references figures/diagrams."
        )

    modality_block = (
        "\n".join(modality_notices) + "\n"
        if modality_notices else ""
    )

    # Format-specific instructions based on query type
    format_instructions = _get_format_instructions(signals)

    user_message = f"""Paper: {paper_title}

Available sections in context:
{section_list_str}

{modality_block}
[CONTEXT START]

{context.context_text}

[CONTEXT END]

{format_instructions}

Question: {query}

Answer:"""

    return user_message


def _get_format_instructions(signals: QuerySignals) -> str:
    """
    Query-type specific formatting instructions.
    Helps LLM produce better structured answers.
    """
    if signals.table_signal:
        return (
            "Instructions: If you found relevant tables, "
            "reproduce the key data in your answer. "
            "Cite the table number and section."
        )
    elif signals.equation_signal:
        return (
            "Instructions: Include the relevant equation in LaTeX format. "
            "Explain each variable. "
            "Cite the section where it appears."
        )
    elif signals.cross_section or signals.broad_score > 2:
        return (
            "Instructions: Provide a structured answer covering "
            "the main points. Use headers if multiple aspects are covered. "
            "Cite sections for each major claim."
        )
    else:
        return (
            "Instructions: Answer directly and precisely. "
            "Cite the section for your answer. "
            "Preserve all numerical values exactly."
        )
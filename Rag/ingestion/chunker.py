# ingestion/chunker.py
"""
Merges parsed elements into coherent chunks ready for embedding.

Algorithm:
1. Filter elements by section treatment
2. Walk elements in order
3. Apply action-based merging logic
4. Enforce token budget with sentence-boundary soft splits
5. Output list of ChunkMetadata objects with combined text fields
"""

from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Optional

from core.models import (
    ParsedElement, ParsedDocument, ChunkMetadata, DocumentMeta
)
from core.logger import get_logger
from core.utils import (
    count_tokens, hash_text, clean_text,
    find_sentence_boundary, normalize_section_title
)
from core.exceptions import ChunkingError
from ingestion.section_filter import (
    get_section_treatment, should_embed,
    get_importance_from_treatment, get_boost_from_treatment,
    get_query_affinities
)
import config

logger = get_logger(__name__)


class ElementBuffer:
    """
    Accumulates elements until a flush condition is met.
    Tracks all metadata needed to produce a ChunkMetadata on flush.
    """

    def __init__(self, paper_id: str, paper_title: str, language: str):
        self.paper_id    = paper_id
        self.paper_title = paper_title
        self.language    = language
        self._reset()

    def _reset(self):
        self.elements:              list[ParsedElement] = []
        self.text_for_embedding_parts: list[str] = []
        self.text_for_display_parts:   list[str] = []
        self.token_count:           int = 0
        self.cites_tables:          set[str] = set()
        self.cites_figures:         set[str] = set()
        self.cites_equations:       set[str] = set()
        self.referenced_by:         set[str] = set()
        self.modalities:            set[str] = set()
        self.table_image_paths:     list[str] = []
        self.equation_image_paths:  list[str] = []
        self.figure_image_paths:    list[str] = []
        self.equation_latex:        list[str] = []
        self.has_table:             bool = False
        self.has_equation:          bool = False
        self.has_figure:            bool = False

    def is_empty(self) -> bool:
        return len(self.elements) == 0

    def add(self, element: ParsedElement, treatment: str):
        """Add element to buffer, accumulating all fields."""
        self.elements.append(element)

        emb = element.phase2_ready.text_for_embedding or ""
        disp = element.phase2_ready.text_for_display or \
               element.content.raw or ""

        if emb:
            self.text_for_embedding_parts.append(clean_text(emb))
        if disp:
            self.text_for_display_parts.append(disp)

        self.token_count += count_tokens(emb)

        # Accumulate cross references
        if element.references:
            self.cites_tables.update(element.references.cites_tables)
            self.cites_figures.update(element.references.cites_figures)
            self.cites_equations.update(element.references.cites_equations)
            self.referenced_by.update(
                element.references.referenced_by_elements
            )

        # Track modalities
        if element.type == "table":
            self.has_table = True
            self.modalities.add("table")
            if element.phase2_ready.image_path_for_vlm:
                self.table_image_paths.append(
                    element.phase2_ready.image_path_for_vlm
                )
        elif element.type == "equation":
            self.has_equation = True
            self.modalities.add("equation")
            if element.phase2_ready.image_path_for_vlm:
                self.equation_image_paths.append(
                    element.phase2_ready.image_path_for_vlm
                )
            if element.phase2_ready.latex_for_llm:
                self.equation_latex.append(
                    element.phase2_ready.latex_for_llm
                )
            elif element.content.latex_clean:
                self.equation_latex.append(element.content.latex_clean)
        elif element.type == "figure":
            self.has_figure = True
            self.modalities.add("figure")
            if element.phase2_ready.image_path_for_vlm:
                self.figure_image_paths.append(
                    element.phase2_ready.image_path_for_vlm
                )
        else:
            self.modalities.add("text")

    def flush(
        self,
        chunk_sequence_index: int,
        total_elements_in_paper: int,
        soft_split: bool = False,
    ) -> Optional[ChunkMetadata]:
        """
        Produce a ChunkMetadata from accumulated elements.
        Returns None if buffer is empty or text is empty.
        """
        if self.is_empty():
            return None

        text_for_embedding = "\n\n".join(self.text_for_embedding_parts).strip()
        text_for_display   = "\n\n".join(self.text_for_display_parts).strip()

        if not text_for_embedding:
            self._reset()
            return None

        # Token count on final combined text
        final_tokens = count_tokens(text_for_embedding)

        # Skip if below minimum (unless it's a self-contained chunk like table)
        if final_tokens < config.CHUNK_MIN_TOKENS and not soft_split:
            # Do not reset — let caller decide to merge forward
            return None

        first_elem = self.elements[0]
        last_elem  = self.elements[-1]
        section    = first_elem.section
        treatment  = get_section_treatment(first_elem)

        # Build chunk_id
        paper_short = self.paper_id.replace(" ", "_")[:30]
        chunk_id = (
            f"{paper_short}_L0_{chunk_sequence_index:04d}"
        )

        # Determine chunk_type
        modality_list = sorted(self.modalities)
        if len(modality_list) == 1:
            chunk_type = modality_list[0]
        else:
            chunk_type = "mixed"

        # Position in paper (0.0 → 1.0)
        position = (
            first_elem.location.order / max(total_elements_in_paper, 1)
        )

        metadata = ChunkMetadata(
            # IDENTITY
            chunk_id        = chunk_id,
            paper_id        = self.paper_id,
            paper_title     = self.paper_title,
            content_hash    = hash_text(text_for_embedding),
            embedding_model = config.EMBEDDING_MODEL,
            embedding_dim   = config.EMBEDDING_DIM,
            ingested_at     = datetime.now(timezone.utc).isoformat(),
            language        = self.language,

            # POSITION
            section_title     = section.title,
            section_hierarchy = section.hierarchy,
            is_key_section    = section.is_key_section,
            page_range        = [
                first_elem.location.page_no,
                last_elem.location.page_no
            ],
            order_range       = [
                first_elem.location.order,
                last_elem.location.order
            ],
            chunk_sequence_index = chunk_sequence_index,
            position_in_paper    = round(position, 4),

            # CONTENT TYPE
            level         = 0,
            depth         = 0,
            chunk_type    = chunk_type,
            modalities    = modality_list,
            has_table     = self.has_table,
            has_equation  = self.has_equation,
            has_figure    = self.has_figure,
            table_image_paths    = self.table_image_paths,
            equation_image_paths = self.equation_image_paths,
            figure_image_paths   = self.figure_image_paths,
            equation_latex       = self.equation_latex,

            # CONTENT
            text_for_embedding = text_for_embedding,
            text_for_display   = text_for_display,
            token_count        = final_tokens,

            # RAPTOR (empty at ingestion, filled in Phase 3)
            parent_ids           = [],
            children_ids         = [],
            summary_of           = [],
            cluster_ids          = [],
            cluster_probabilities = {},
            sibling_ids          = [],

            # RETRIEVAL SIGNALS
            query_affinities  = get_query_affinities(first_elem, treatment),
            section_importance = get_importance_from_treatment(treatment),
            retrieval_boost    = get_boost_from_treatment(treatment),

            # CROSS REFERENCES
            cites_tables    = sorted(self.cites_tables),
            cites_figures   = sorted(self.cites_figures),
            cites_equations = sorted(self.cites_equations),
            referenced_by_elements = sorted(self.referenced_by),

            # FLAGS
            soft_split         = soft_split,
            source_element_ids = [e.element_id for e in self.elements],
        )

        self._reset()
        return metadata


def _handle_no_section_headers(
    elements: list[ParsedElement],
    paper_id: str,
    paper_title: str,
    language: str,
) -> list[ParsedElement]:
    """
    Fallback for papers with no section headers.
    Injects synthetic section boundaries every 400 tokens.
    """
    logger.warning(
        f"Paper '{paper_id}' has no section headers detected. "
        f"Applying token-based synthetic section split fallback."
    )
    accumulated_tokens = 0
    part_index = 1
    synthetic_title = f"part_{part_index}"

    for elem in elements:
        # Inject synthetic section into element
        elem.section.title = synthetic_title
        elem.section.hierarchy = [synthetic_title]
        elem.section.is_key_section = False

        tokens = count_tokens(
            elem.phase2_ready.text_for_embedding or ""
        )
        accumulated_tokens += tokens

        if accumulated_tokens >= 400:
            accumulated_tokens = 0
            part_index += 1
            synthetic_title = f"part_{part_index}"

    return elements


def build_chunks(document: ParsedDocument) -> list[ChunkMetadata]:
    """
    Main chunking function.
    Walks elements in order and produces merged chunks.

    Returns list of ChunkMetadata ready for embedding.
    """
    paper_id    = document.document.paper_id
    paper_title = document.document.paper_title
    language    = document.document.language
    elements    = document.elements
    total_elems = len(elements)

    logger.info(
        f"Starting chunking for '{paper_id}' "
        f"({total_elems} elements)"
    )

    # ── Detect and handle no-section-header edge case ─────────────────────
    header_count = sum(
        1 for e in elements if e.type == "section_header"
    )
    if header_count == 0:
        elements = _handle_no_section_headers(
            elements, paper_id, paper_title, language
        )

    # ── Collect reference metadata before main loop ───────────────────────
    # References are metadata_only — extract to list, don't embed
    references_list = []
    for elem in elements:
        treatment = get_section_treatment(elem)
        if treatment == "metadata_only" and elem.content.raw:
            references_list.append(elem.content.raw)

    # ── Main chunking loop ─────────────────────────────────────────────────
    chunks:     list[ChunkMetadata] = []
    buffer      = ElementBuffer(paper_id, paper_title, language)
    seq_index   = 0
    pending_min: Optional[ChunkMetadata] = None
    # pending_min: holds a chunk that was too small to emit alone
    # it gets merged into the next chunk that comes along

    def emit_chunk(soft_split: bool = False) -> None:
        """Flush buffer and emit chunk, handling pending_min merges."""
        nonlocal seq_index, pending_min

        candidate = buffer.flush(seq_index, total_elems, soft_split)

        if candidate is None:
            return

        if candidate.token_count < config.CHUNK_MIN_TOKENS:
            # Too small — hold as pending, merge into next
            pending_min = candidate
            logger.debug(
                f"Chunk {candidate.chunk_id} too small "
                f"({candidate.token_count} tokens), holding for merge"
            )
            return

        # Merge pending_min into this chunk if exists
        if pending_min is not None:
            candidate = _merge_two_chunks(pending_min, candidate, seq_index)
            pending_min = None

        chunks.append(candidate)
        seq_index += 1
        logger.debug(
            f"Emitted chunk {candidate.chunk_id} "
            f"({candidate.token_count} tokens, "
            f"section: {candidate.section_title})"
        )

    for i, element in enumerate(elements):
        treatment = get_section_treatment(element)
        action    = element.phase2_ready.action

        # ── SKIP ──────────────────────────────────────────────────────────
        if treatment in ("skip", "metadata_only"):
            logger.debug(
                f"Skipping element {element.element_id} "
                f"(treatment: {treatment})"
            )
            continue

        # ── HARD BOUNDARY (section header) ────────────────────────────────
        if action == "hard_chunk_boundary":
            if not buffer.is_empty():
                emit_chunk()
            # Section header itself is never added to buffer
            continue

        # ── SELF-CONTAINED CHUNK (tables, large equations) ─────────────────
        if action == "self_contained_chunk":
            # Flush current buffer first
            if not buffer.is_empty():
                emit_chunk()

            # Emit table/equation as its own chunk immediately
            buffer.add(element, treatment)
            emit_chunk()
            continue

        # ── MERGE WITH ADJACENT TEXT ───────────────────────────────────────
        if action in ("merge_with_adjacent_text", "merge_with_anchor_element"):

            elem_text       = element.phase2_ready.text_for_embedding or ""
            incoming_tokens = count_tokens(elem_text)

            # ── CASE 1: Single element exceeds hard limit ──────────────────────
            # Guard: never split tables, equations, figures
            # They must stay intact regardless of token count
            is_atomic = element.type in ("table", "equation", "figure")

            if incoming_tokens > config.CHUNK_HARD_LIMIT and not is_atomic:
                # Flush whatever is in buffer first
                if not buffer.is_empty():
                    emit_chunk()

                # Split large text element into sub-chunks
                remaining_text = elem_text
                while remaining_text:
                    remaining_tokens = count_tokens(remaining_text)

                    if remaining_tokens <= config.CHUNK_HARD_LIMIT:
                        buffer.text_for_embedding_parts.append(
                            clean_text(remaining_text)
                        )
                        buffer.text_for_display_parts.append(
                            element.phase2_ready.text_for_display or remaining_text
                        )
                        buffer.token_count += remaining_tokens
                        buffer.elements.append(element)
                        if element.references:
                            buffer.cites_tables.update(
                                element.references.cites_tables
                            )
                        buffer.modalities.add(element.type)
                        break

                    split_char = find_sentence_boundary(
                        remaining_text, config.CHUNK_TARGET_TOKENS
                    )

                    if split_char <= 0 or split_char >= len(remaining_text):
                        split_char = config.CHUNK_TARGET_TOKENS * 4

                    first_part     = remaining_text[:split_char].strip()
                    remaining_text = remaining_text[split_char:].strip()

                    if first_part:
                        buffer.text_for_embedding_parts.append(
                            clean_text(first_part)
                        )
                        buffer.text_for_display_parts.append(first_part)
                        buffer.token_count += count_tokens(first_part)
                        buffer.elements.append(element)
                        buffer.modalities.add(element.type)
                        emit_chunk(soft_split=True)

                continue

            # ── CASE 1B: Atomic element exceeds hard limit (table/equation) ────
            # Accept it as-is, emit as its own chunk
            if incoming_tokens > config.CHUNK_HARD_LIMIT and is_atomic:
                if not buffer.is_empty():
                    emit_chunk()
                buffer.add(element, treatment)
                emit_chunk()
                continue

            # ── CASE 2: Adding element would exceed hard limit ─────────────────
            projected = buffer.token_count + incoming_tokens

            if projected > config.CHUNK_HARD_LIMIT and not buffer.is_empty():
                combined   = "\n\n".join(buffer.text_for_embedding_parts)
                split_char = find_sentence_boundary(
                    combined, config.CHUNK_TARGET_TOKENS
                )

                if split_char > 0 and split_char < len(combined):
                    first_part  = combined[:split_char].strip()
                    second_part = combined[split_char:].strip()

                    buffer.text_for_embedding_parts = [first_part]
                    buffer.text_for_display_parts   = [first_part]
                    buffer.token_count = count_tokens(first_part)
                    emit_chunk(soft_split=True)

                    if second_part:
                        buffer.text_for_embedding_parts.append(
                            clean_text(second_part)
                        )
                        buffer.text_for_display_parts.append(second_part)
                        buffer.token_count += count_tokens(second_part)
                else:
                    emit_chunk(soft_split=True)

            # ── CASE 3: Normal add ─────────────────────────────────────────────
            buffer.add(element, treatment)

            if buffer.token_count >= config.CHUNK_TARGET_TOKENS:
                next_elem = elements[i + 1] if i + 1 < len(elements) else None
                next_is_boundary = (
                    next_elem is not None and
                    next_elem.phase2_ready.action == "hard_chunk_boundary"
                )
                if not next_is_boundary:
                    emit_chunk()

            continue

            # Check if we hit target after adding
            if buffer.token_count >= config.CHUNK_TARGET_TOKENS:
                # Check next element — if it's a boundary, let it flush
                next_elem = elements[i + 1] if i + 1 < len(elements) else None
                next_is_boundary = (
                    next_elem is not None and
                    next_elem.phase2_ready.action == "hard_chunk_boundary"
                )
                if not next_is_boundary:
                    emit_chunk()
            continue

        # ── UNKNOWN ACTION ─────────────────────────────────────────────────
        logger.warning(
            f"Unknown action '{action}' on element "
            f"{element.element_id}, treating as merge"
        )
        buffer.add(element, treatment)

    # ── Flush remaining buffer ─────────────────────────────────────────────
    if not buffer.is_empty():
        emit_chunk()

    # ── Handle any remaining pending_min ──────────────────────────────────
    if pending_min is not None:
        # Last chunk was too small and nothing came after
        # Emit it anyway — better than losing content
        chunks.append(pending_min)
        logger.debug(
            f"Emitting pending small chunk {pending_min.chunk_id} "
            f"at end of document"
        )

    logger.info(
        f"Chunking complete for '{paper_id}': "
        f"{len(chunks)} chunks produced"
    )

    return chunks, references_list


def _inject_raw_text(
    buffer: ElementBuffer,
    text: str,
    element: ParsedElement,
    treatment: str,
):
    """Helper to inject a raw text string into buffer as a pseudo-element."""
    # We can't add a raw string to buffer.add() which expects ParsedElement
    # Instead directly append to the parts lists
    buffer.text_for_embedding_parts.append(clean_text(text))
    buffer.text_for_display_parts.append(text)
    buffer.token_count += count_tokens(text)


def _merge_two_chunks(
    small: ChunkMetadata,
    big: ChunkMetadata,
    new_seq_index: int,
) -> ChunkMetadata:
    """
    Merge a small pending chunk into the next chunk.
    Small chunk goes first (preserves reading order).
    """
    merged_text_emb  = (
        small.text_for_embedding + "\n\n" + big.text_for_embedding
    )
    merged_text_disp = (
        small.text_for_display + "\n\n" + big.text_for_display
    )

    big.chunk_id             = big.chunk_id  # keep the bigger chunk's ID
    big.text_for_embedding   = merged_text_emb
    big.text_for_display     = merged_text_disp
    big.token_count          = count_tokens(merged_text_emb)
    big.content_hash         = hash_text(merged_text_emb)
    big.order_range[0]       = small.order_range[0]
    big.page_range[0]        = small.page_range[0]
    big.source_element_ids   = (
        small.source_element_ids + big.source_element_ids
    )
    big.cites_tables    = sorted(set(small.cites_tables + big.cites_tables))
    big.cites_figures   = sorted(set(small.cites_figures + big.cites_figures))
    big.cites_equations = sorted(
        set(small.cites_equations + big.cites_equations)
    )
    big.modalities = sorted(set(small.modalities + big.modalities))
    big.has_table    = small.has_table    or big.has_table
    big.has_equation = small.has_equation or big.has_equation
    big.has_figure   = small.has_figure   or big.has_figure

    return big
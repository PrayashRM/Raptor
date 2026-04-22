# core/models.py
from __future__ import annotations
from typing import Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


# ── Parser Element Models (Input) ─────────────────────────────────────────────

class ElementLocation(BaseModel):
    page_no: int
    order: int
    bbox: Optional[dict] = None


class ElementSection(BaseModel):
    title: str
    hierarchy: list[str]
    is_key_section: bool
    section_index: int
    section_importance: Optional[str] = "medium"
    retrieval_boost: Optional[float] = 1.0
    heading_level: Optional[int] = None
    is_section_boundary: Optional[bool] = False


class ElementContent(BaseModel):
    raw: Optional[str] = None
    token_count: Optional[int] = 0
    caption: Optional[str] = None
    header_row: Optional[list[str]] = None
    rows: Optional[list[dict]] = None
    has_formula_cells: Optional[bool] = False
    image_path: Optional[str] = None
    latex_raw: Optional[str] = None
    latex_clean: Optional[str] = None
    latex_confidence: Optional[float] = None
    natural_language_description: Optional[str] = None
    equation_number: Optional[str] = None
    extracted_variables: Optional[Union[dict, list]] = None


class ElementReferences(BaseModel):
    cites_equations: list[str] = Field(default_factory=list)
    cites_tables: list[str] = Field(default_factory=list)
    cites_figures: list[str] = Field(default_factory=list)
    referenced_by_elements: list[str] = Field(default_factory=list)


class ElementSurroundingContext(BaseModel):
    preceding_element_id: Optional[str] = None
    following_element_id: Optional[str] = None
    preceding_text: Optional[str] = None
    following_text: Optional[str] = None
    context_window_text: Optional[str] = None


class ElementPhase2Ready(BaseModel):
    action: str
    text_for_embedding: Optional[str] = None
    text_for_display: Optional[str] = None
    embedding_strategy: Optional[str] = None
    embedding_notes: Optional[str] = None
    image_path_for_vlm: Optional[str] = None
    latex_for_llm: Optional[str] = None
    note: Optional[str] = None


class ElementAdjacency(BaseModel):
    prev_element_id: Optional[str] = None
    next_element_id: Optional[str] = None
    prev_type: Optional[str] = None
    next_type: Optional[str] = None


class ParsedElement(BaseModel):
    element_id: str
    self_ref: Optional[str] = None
    type: str
    subtype: Optional[str] = None
    location: ElementLocation
    section: ElementSection
    content: ElementContent
    references: Optional[ElementReferences] = None
    surrounding_context: Optional[ElementSurroundingContext] = None
    phase2_ready: ElementPhase2Ready
    adjacency: ElementAdjacency


class DocumentMeta(BaseModel):
    paper_id: str
    paper_title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    total_pages: Optional[int] = None
    total_elements: Optional[int] = None
    language: str = "en"
    parsed_at: Optional[str] = None
    parser_version: Optional[str] = None
    source_pdf: Optional[str] = None
    assets_dir: Optional[str] = None


class ParsedDocument(BaseModel):
    document: DocumentMeta
    elements: list[ParsedElement]


# ── Chunk Models (Phase 2 Output) ─────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    # IDENTITY
    chunk_id: str
    paper_id: str
    paper_title: str
    content_hash: str
    embedding_model: str
    embedding_dim: int
    ingested_at: str
    language: str = "en"

    # POSITION
    section_title: str
    section_hierarchy: list[str]
    is_key_section: bool
    page_range: list[int]
    order_range: list[int]
    chunk_sequence_index: int
    position_in_paper: float

    # CONTENT TYPE
    level: int = 0
    depth: int = 0
    chunk_type: str
    modalities: list[str]
    has_table: bool = False
    has_equation: bool = False
    has_figure: bool = False
    table_image_paths: list[str] = Field(default_factory=list)
    equation_image_paths: list[str] = Field(default_factory=list)
    figure_image_paths: list[str] = Field(default_factory=list)
    equation_latex: list[str] = Field(default_factory=list)

    # CONTENT
    text_for_embedding: str
    text_for_display: str
    token_count: int

    # RAPTOR TREE LINKS (populated in Phase 3)
    parent_ids: list[str] = Field(default_factory=list)
    children_ids: list[str] = Field(default_factory=list)
    summary_of: list[str] = Field(default_factory=list)
    cluster_ids: list[int] = Field(default_factory=list)
    cluster_probabilities: dict[str, float] = Field(default_factory=dict)
    sibling_ids: list[str] = Field(default_factory=list)

    # RETRIEVAL SIGNALS
    query_affinities: list[str] = Field(default_factory=list)
    section_importance: str = "medium"
    retrieval_boost: float = 1.0

    # CROSS REFERENCES
    cites_tables: list[str] = Field(default_factory=list)
    cites_figures: list[str] = Field(default_factory=list)
    cites_equations: list[str] = Field(default_factory=list)
    referenced_by_elements: list[str] = Field(default_factory=list)

    # SOFT SPLIT FLAG
    soft_split: bool = False
    source_element_ids: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    metadata: ChunkMetadata
    dense_vector: Optional[list[float]] = None
    sparse_indices: Optional[list[int]] = None
    sparse_values: Optional[list[float]] = None


# ── Ingestion Report ──────────────────────────────────────────────────────────

class IngestionReport(BaseModel):
    paper_id: str
    total_elements_input: int
    elements_skipped: int
    elements_processed: int
    total_chunks_created: int
    soft_splits: int
    self_contained_chunks: int
    merged_chunks: int
    min_chunk_tokens: int
    max_chunk_tokens: int
    avg_chunk_tokens: float
    sections_processed: list[str]
    sections_skipped: list[str]
    ingestion_duration_seconds: float
    status: str
    errors: list[str] = Field(default_factory=list)
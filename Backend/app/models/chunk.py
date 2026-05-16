"""
app/models/chunk.py

RAG index chunks — the core of the vector search system.

Each paper is split into chunks during the RAG pipeline.
Each chunk is embedded into a 1536-dimensional vector
(text-embedding-3-small) and stored here alongside the
raw text and position metadata.

The HNSW index on the embedding column enables approximate
nearest-neighbour search — the core of RAG retrieval.

pgvector:
    The VECTOR(1536) column type and the HNSW index require
    the pgvector extension which is pre-installed in the
    pgvector/pgvector:pg16 Docker image.
    Extension is enabled in scripts/init-db.sql.
"""

from typing import TYPE_CHECKING, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, SmallInteger, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.paper import Paper


class Chunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A single text chunk from a paper's RAG index.

    chunk_index:
        Zero-based position of this chunk within the paper.
        Used to reconstruct reading order if needed.

    text:
        The verbatim text content of this chunk.
        Returned as chunk_text in SourceChunk API responses.

    page_number:
        Page in the original document where this chunk starts.
        Null if the parser could not determine page boundaries
        (common for DOCX and TXT files).

    section:
        Section heading under which this chunk appears.
        Extracted by the hybrid chunker from headings.
        Null if no section structure was detected.

    embedding:
        1536-dimensional dense vector produced by
        text-embedding-3-small. Used for cosine similarity
        search at query time.
        VECTOR(1536) requires pgvector extension.

    HNSW Index:
        Hierarchical Navigable Small World index.
        Faster query performance than IVFFlat at the cost of
        higher memory usage. Correct choice for this app where
        retrieval latency directly affects chat response time.
        Created in the Alembic migration, not here, because
        SQLAlchemy does not natively support HNSW index syntax.
    """

    __tablename__ = "chunks"

    __table_args__ = (
        # Composite index for ordered chunk retrieval within a paper
        Index(
            "idx_chunks_paper_chunk_order",
            "paper_id",
            "chunk_index",
        ),
    )
    # NOTE: The HNSW vector index is created in the Alembic migration
    # using raw SQL because SQLAlchemy does not support HNSW syntax:
    # CREATE INDEX idx_chunks_embedding_hnsw
    # ON chunks USING hnsw (embedding vector_cosine_ops)
    # WITH (m = 16, ef_construction = 64);

    # ------------------------------------------------------------------
    # Foreign keys
    # ------------------------------------------------------------------
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Paper this chunk belongs to.",
    )

    # ------------------------------------------------------------------
    # Position metadata
    # ------------------------------------------------------------------
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment=(
            "Zero-based position of this chunk within the paper. "
            "Chunks are ordered by this field for context window assembly."
        ),
    )

    page_number: Mapped[Optional[int]] = mapped_column(
        SmallInteger,
        nullable=True,
        default=None,
        comment=(
            "Page number in the original document where this chunk starts. "
            "Null for DOCX/TXT files where page boundaries are unavailable."
        ),
    )

    section: Mapped[Optional[str]] = mapped_column(
        nullable=True,
        default=None,
        comment=(
            "Section heading under which this chunk appears. "
            "Extracted by the hybrid chunker. "
            "Null if no section structure was detected in the document."
        ),
    )

    # ------------------------------------------------------------------
    # Text content
    # ------------------------------------------------------------------
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "Verbatim text content of this chunk. "
            "Returned as chunk_text in SourceChunk API responses."
        ),
    )

    # ------------------------------------------------------------------
    # Vector embedding
    # ------------------------------------------------------------------
    embedding: Mapped[List[float]] = mapped_column(
        Vector(1536),
        nullable=False,
        comment=(
            "1536-dimensional dense vector produced by text-embedding-3-small. "
            "Used for cosine similarity search at query time. "
            "Requires pgvector extension (pre-installed in pgvector/pgvector:pg16)."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    paper: Mapped["Paper"] = relationship(
        "Paper",
        back_populates="chunks",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<Chunk "
            f"id={self.id!r} "
            f"paper_id={self.paper_id!r} "
            f"chunk_index={self.chunk_index!r} "
            f"page={self.page_number!r} "
            f"section={self.section!r}>"
        )
"""
app/models/pipeline_trace.py

Per-paper RAG pipeline execution trace.

One record per pipeline run per paper. Stores the step-by-step
execution details (duration, chunk count, embedding model, etc.)
as a JSONB array rather than normalised rows.

Why JSONB for steps:
    The pipeline has exactly 4 fixed steps (parse, chunk, embed, index).
    Storing them as JSONB on one record means the entire trace
    is readable in a single row fetch with no joins. Normalising
    into a separate pipeline_trace_steps table would add a join
    for zero benefit at this scale.

Access control:
    Readable by the user who first uploaded/ingested the paper
    (uploaded_by field on papers table) and by admin users.
    Enforced in paper_service.get_pipeline_trace().
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.paper import Paper
    from app.models.user import User


class PipelineTrace(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Execution trace of a RAG pipeline run for a paper.

    steps (JSONB):
        Array of PipelineStep objects. Each step:
        {
            "step": "parse" | "chunk" | "embed" | "index",
            "status": "pending" | "running" | "done" | "failed",
            "duration_ms": int | null,
            "extraction_method": "native_text" | "ocr" | null,
            "chunk_count": int | null,
            "chunk_strategy": str | null,
            "model": str | null,
            "vector_count": int | null,
            "vector_store": str | null,
            "error": str | null
        }
        Written incrementally by the ARQ worker as each step completes.

    total_duration_ms:
        Sum of all step durations. Null until pipeline completes.
        Used by stats_service for avg_rag_build_time_ms calculation.

    triggered_by:
        User ID of the user whose upload/ingest triggered this run.
        SET NULL if that user is deleted (paper record is retained).
    """

    __tablename__ = "pipeline_traces"

    __table_args__ = (
        # Primary access pattern: trace for a specific paper
        Index(
            "idx_pipeline_traces_paper_id",
            "paper_id",
        ),
        # Stats query: recent build times for avg calculation
        Index(
            "idx_pipeline_traces_triggered_at",
            "triggered_at",
        ),
    )

    # ------------------------------------------------------------------
    # Foreign keys
    # ------------------------------------------------------------------
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        comment="Paper this pipeline trace belongs to.",
    )

    triggered_by: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment=(
            "User who initiated the ingest that triggered this pipeline run. "
            "SET NULL if the user is deleted."
        ),
    )

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Timestamp when the pipeline job was enqueued.",
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment=(
            "Timestamp when the pipeline job finished (success or failure). "
            "Null while pipeline is still running."
        ),
    )

    total_duration_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        default=None,
        comment=(
            "Total wall-clock duration across all steps in milliseconds. "
            "Null until pipeline completes. "
            "Used by stats_service for avg_rag_build_time_ms."
        ),
    )

    # ------------------------------------------------------------------
    # Step details
    # ------------------------------------------------------------------
    steps: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment=(
            "Array of PipelineStep objects. Written incrementally by the "
            "ARQ worker as each step (parse, chunk, embed, index) completes. "
            "See PipelineStep schema for full shape documentation."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    paper: Mapped["Paper"] = relationship(
        "Paper",
        back_populates="pipeline_trace",
        lazy="noload",
    )

    triggered_by_user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="triggered_pipeline_traces",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<PipelineTrace "
            f"id={self.id!r} "
            f"paper_id={self.paper_id!r} "
            f"total_duration_ms={self.total_duration_ms!r} "
            f"completed_at={self.completed_at!r}>"
        )
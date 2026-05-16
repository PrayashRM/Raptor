"""
app/models/paper.py

Core paper model. Represents a research paper on the platform.

Papers are GLOBAL records — shared across all users. Deduplication
by DOI (external papers) and SHA-256 content hash (uploads) ensures
each paper exists once on the platform.

The user-to-paper relationship is through saved_papers (library
membership). A user can save/ingest a paper but never "owns" the
global record (except uploaded_by for access control on pipeline trace).

RAG lifecycle:
    rag_status: 'none' → 'processing' → 'ready' | 'failed'
    rag_lock:   Prevents duplicate pipeline jobs from being enqueued.

Full-text search:
    search_vector is a tsvector column populated by a PostgreSQL
    trigger from title + authors (JSON) + abstract.
    GIN index enables fast full-text search for the dashboard ?q= param.
    The trigger is created in the Alembic migration, not in this model.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import Boolean, Index , ForeignKey, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.chat_message import ChatMessage
    from app.models.chat_session import ChatSession
    from app.models.chunk import Chunk
    from app.models.collection_paper import CollectionPaper
    from app.models.notification import Notification
    from app.models.pipeline_trace import PipelineTrace
    from app.models.saved_paper import SavedPaper
    from app.models.tag import Tag
    from app.models.user import User


class Paper(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A research paper record on the platform.

    Deduplication keys:
        doi          → for external papers (discovery/ingest)
        content_hash → for uploaded papers (SHA-256 of file bytes)

    Source:
        'upload'       → user uploaded a file directly
        'external_api' → ingested from discovery (DOI or URL)

    RAG lifecycle:
        rag_status: none → processing → ready | failed
        rag_lock:   True while a pipeline job is running
                    Prevents duplicate jobs from being enqueued

    File storage:
        file_key: R2 object key (e.g. "papers/{id}/original.pdf")
        Null for bookmarked external papers not yet ingested.
        Pre-signed URL generated fresh on every GET /papers/{id}.

    Full-text search:
        search_vector: TSVECTOR populated by DB trigger from
        title + authors + abstract. GIN-indexed.
    """

    __tablename__ = "papers"

    __table_args__ = (
        # Deduplication: DOI (partial — only non-null DOIs)
        Index(
            "idx_papers_doi",
            "doi",
            unique=True,
            postgresql_where="doi IS NOT NULL",
        ),
        # Deduplication: content hash (partial — only non-null hashes)
        Index(
            "idx_papers_content_hash",
            "content_hash",
            unique=True,
            postgresql_where="content_hash IS NOT NULL",
        ),
        # Dashboard filter: by rag_status
        Index(
            "idx_papers_rag_status",
            "rag_status",
        ),
        # Admin queries: papers by uploader
        Index(
            "idx_papers_uploaded_by",
            "uploaded_by",
        ),
        # Full-text search GIN index
        Index(
            "idx_papers_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        # Dashboard sort: by year
        Index(
            "idx_papers_year",
            "year",
        ),
    )

    # ------------------------------------------------------------------
    # Deduplication keys
    # ------------------------------------------------------------------
    doi: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        unique=True,
        comment=(
            "DOI string (without resolver prefix). "
            "Primary deduplication key for external papers. "
            "Null for papers uploaded without a DOI."
        ),
    )

    content_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        unique=True,
        comment=(
            "SHA-256 hex digest of the raw file bytes. "
            "Primary deduplication key for uploaded papers. "
            "Null for external papers deduplicated by DOI."
        ),
    )

    # ------------------------------------------------------------------
    # Source tracking
    # ------------------------------------------------------------------
    source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment=(
            "How this paper entered the platform. "
            "'upload' = direct file upload. "
            "'external_api' = ingested from discovery (DOI/URL)."
        ),
    )

    # ------------------------------------------------------------------
    # Metadata (auto-extracted, manually overridable)
    # ------------------------------------------------------------------
    title: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "Paper title. Auto-extracted during parse step. "
            "Null if extraction failed. "
            "Overridable via PATCH /papers/{id}/metadata."
        ),
    )

    authors: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment=(
            "Array of Author objects: "
            "[{name, affiliation, orcid}]. "
            "Auto-extracted. Null if extraction failed."
        ),
    )

    abstract: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Paper abstract. Auto-extracted. Null if extraction failed.",
    )

    year: Mapped[Optional[int]] = mapped_column(
        SmallInteger,
        nullable=True,
        comment="Publication year. Auto-extracted. Null if extraction failed.",
    )

    journal: Mapped[Optional[str]] = mapped_column(
        String(300),
        nullable=True,
        comment="Journal or conference name.",
    )

    venue: Mapped[Optional[str]] = mapped_column(
        String(300),
        nullable=True,
        comment="Venue (may differ from journal for conference papers).",
    )

    url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Link to the original paper at source (arXiv, DOI resolver).",
    )

    # ------------------------------------------------------------------
    # File storage
    # ------------------------------------------------------------------
    file_key: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "R2/S3 object key for the stored file. "
            "e.g. 'papers/{paper_id}/original.pdf'. "
            "Null for bookmarked papers not yet ingested."
        ),
    )

    # ------------------------------------------------------------------
    # RAG pipeline state
    # ------------------------------------------------------------------
    rag_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="none",
        comment=(
            "RAG pipeline lifecycle status. "
            "'none' = not started. "
            "'processing' = pipeline running. "
            "'ready' = RAG index built, chat available. "
            "'failed' = pipeline error."
        ),
    )

    rag_lock: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment=(
            "Prevents duplicate pipeline jobs from being enqueued. "
            "Set True when a job is dispatched, False on completion/failure."
        ),
    )

    extraction_method: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        default=None,
        comment=(
            "How text was extracted from the document. "
            "'native_text' = PDF with selectable text. "
            "'ocr' = scanned/image PDF processed with OCR. "
            "Null if pipeline has not run."
        ),
    )

    # ------------------------------------------------------------------
    # Ownership
    # ------------------------------------------------------------------
    uploaded_by: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment=(
            "User ID of the user who first introduced this paper to the platform. "
            "SET NULL if that user is deleted (paper is retained). "
            "Used for access control on pipeline-trace endpoint."
        ),
    )

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------
    search_vector: Mapped[Optional[str]] = mapped_column(
        TSVECTOR,
        nullable=True,
        comment=(
            "tsvector for full-text search. "
            "Populated by a PostgreSQL trigger from title + authors + abstract. "
            "GIN-indexed for fast search. "
            "Trigger created in Alembic migration."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    uploader: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="uploaded_papers",
        foreign_keys=[uploaded_by],
        lazy="noload",
    )

    saved_papers: Mapped[List["SavedPaper"]] = relationship(
        "SavedPaper",
        back_populates="paper",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="paper",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        "ChatSession",
        back_populates="paper",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    chat_messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="paper",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    tags: Mapped[List["Tag"]] = relationship(
        "Tag",
        back_populates="paper",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    collection_papers: Mapped[List["CollectionPaper"]] = relationship(
        "CollectionPaper",
        back_populates="paper",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    notifications: Mapped[List["Notification"]] = relationship(
        "Notification",
        back_populates="paper",
        lazy="noload",
    )

    pipeline_trace: Mapped[Optional["PipelineTrace"]] = relationship(
        "PipelineTrace",
        back_populates="paper",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<Paper "
            f"id={self.id!r} "
            f"doi={self.doi!r} "
            f"title={self.title!r} "
            f"rag_status={self.rag_status!r}>"
        )
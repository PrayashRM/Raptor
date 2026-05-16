"""
app/models/saved_paper.py

Join table: user ↔ paper with library membership status.

This table represents a user's personal library. Every paper a user
interacts with (upload, ingest, save, bookmark) creates or updates
a record here.

Status:
    'bookmarked' → saved for later, no RAG pipeline triggered
    'ingested'   → RAG pipeline triggered (or already ready)
    Ingest always implies saved. Bookmark does not imply ingested.

Status upgrade:
    bookmarked → ingested (when user triggers ingest)
    Never downgrade: ingested → bookmarked is not allowed.
    Enforced in saved_paper_repo.upsert() via GREATEST or application logic.

last_chatted_at:
    Denormalised timestamp updated by chat_service on every new message.
    Drives the last_chatted_at sort field on the dashboard.
    Avoids a JOIN to chat_sessions for every dashboard query.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.paper import Paper
    from app.models.user import User


class SavedPaper(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Library membership record: user has this paper in their library.

    saved_at:
        When the user first added this paper to their library.
        Not updated on subsequent interactions (stable sort key).

    status:
        'bookmarked' → saved for later, no RAG pipeline triggered.
        'ingested'   → RAG pipeline triggered or complete.
        Only upgrades: bookmarked → ingested. Never downgraded.

    last_chatted_at:
        Denormalised. Updated on every new chat message.
        Null if the user has never chatted with this paper.
        Used directly in dashboard sort without joining chat_sessions.
    """

    __tablename__ = "saved_papers"

    __table_args__ = (
        # One record per (user, paper) — enforced at DB level
        UniqueConstraint(
            "user_id",
            "paper_id",
            name="ux_saved_papers_user_paper",
        ),
        # Dashboard query: all papers for a user, sorted by saved_at
        Index(
            "idx_saved_papers_user_saved",
            "user_id",
            "saved_at",
        ),
        # Dashboard filter: by status
        Index(
            "idx_saved_papers_status",
            "status",
        ),
        # Subscriber query: all users who have a given paper
        Index(
            "idx_saved_papers_paper_id",
            "paper_id",
        ),
    )

    # ------------------------------------------------------------------
    # Foreign keys
    # ------------------------------------------------------------------
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="User who saved this paper to their library.",
    )

    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        comment="Paper that is in the user's library.",
    )

    # ------------------------------------------------------------------
    # Library state
    # ------------------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="bookmarked",
        comment=(
            "Library membership status. "
            "'bookmarked' = saved for later, no RAG triggered. "
            "'ingested' = RAG pipeline triggered or complete. "
            "Only upgrades: bookmarked → ingested."
        ),
    )

    saved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        comment=(
            "When the user first added this paper to their library. "
            "Stable — not updated on subsequent interactions. "
            "Default sort key for the dashboard."
        ),
    )

    last_chatted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment=(
            "Timestamp of the user's most recent chat message on this paper. "
            "Denormalised from chat_messages for dashboard sort performance. "
            "Null if the user has never chatted with this paper."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user: Mapped["User"] = relationship(
        "User",
        back_populates="saved_papers",
        lazy="noload",
    )

    paper: Mapped["Paper"] = relationship(
        "Paper",
        back_populates="saved_papers",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<SavedPaper "
            f"id={self.id!r} "
            f"user_id={self.user_id!r} "
            f"paper_id={self.paper_id!r} "
            f"status={self.status!r}>"
        )
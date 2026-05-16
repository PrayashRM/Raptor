"""
app/models/chat_session.py

One record per (user_id, paper_id) pair.
Represents the single rolling conversation session between
a user and a paper. Never duplicated — get_or_create pattern.

Relationships:
    user     → User (many sessions per user)
    paper    → Paper (many sessions per paper)
    messages → ChatMessage (all messages in this session)
"""

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.chat_message import ChatMessage
    from app.models.paper import Paper
    from app.models.user import User


class ChatSession(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Single rolling conversation session per (user_id, paper_id) pair.

    rolling_summary:
        Condensed summary of older messages maintained by the
        summarisation ARQ task. Prepended to LLM context on each
        new message to preserve conversational coherence without
        unbounded token growth.

    cleared_at:
        Set by clear_history(). Messages created before this
        timestamp are excluded from history queries. The session
        record itself is never deleted — cleared_at acts as a
        soft boundary. Physical deletion of old messages is handled
        by the cleanup ARQ task after 30 days.

    last_message_at:
        Denormalised timestamp updated on every new message.
        Drives the last_chatted_at sort field on the dashboard.
    """

    __tablename__ = "chat_sessions"

    __table_args__ = (
        # One session per (user, paper) pair — enforced at DB level
        UniqueConstraint(
            "user_id",
            "paper_id",
            name="ux_chat_sessions_user_paper",
        ),
    )

    # ------------------------------------------------------------------
    # Foreign keys
    # ------------------------------------------------------------------
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who owns this conversation session.",
    )

    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Paper this session is about.",
    )

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------
    rolling_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment=(
            "Condensed summary of older conversation turns. "
            "Updated every SUMMARY_EVERY_K_TURNS by the summarisation task. "
            "Prepended to LLM context as a system message to preserve "
            "coherence without unbounded token growth."
        ),
    )

    cleared_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment=(
            "Timestamp of the last clear_history() operation. "
            "Messages with created_at <= cleared_at are excluded "
            "from history queries. Null means no clear has occurred."
        ),
    )

    last_message_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        index=True,
        comment=(
            "Denormalised timestamp of the most recent message in this session. "
            "Updated on every new message insert. "
            "Drives last_chatted_at on dashboard and saved_papers."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user: Mapped["User"] = relationship(
        "User",
        back_populates="chat_sessions",
        lazy="noload",
    )

    paper: Mapped["Paper"] = relationship(
        "Paper",
        back_populates="chat_sessions",
        lazy="noload",
    )

    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="noload",
        order_by="ChatMessage.created_at.asc()",
    )

    def __repr__(self) -> str:
        return (
            f"<ChatSession "
            f"id={self.id!r} "
            f"user_id={self.user_id!r} "
            f"paper_id={self.paper_id!r} "
            f"last_message_at={self.last_message_at!r}>"
        )
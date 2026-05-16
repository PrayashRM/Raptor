"""
app/models/notification.py

In-app notification records.

Delivery:
    Notifications are pushed in real time via WebSocket when
    the user is online (ws_manager.send_notification).
    If the socket is not open, the DB record is the fallback —
    loaded on next GET /notifications.

Types:
    rag_ready  → RAG pipeline completed for a paper in library
    rag_failed → RAG pipeline failed for a paper in library
    system     → Platform-level announcement or maintenance notice
"""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, ForeignKey, Index, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.paper import Paper
    from app.models.user import User


class Notification(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A single in-app notification for a user.

    paper_id:
        References the paper that triggered this notification.
        Null for system notifications.
        SET NULL on paper deletion — notification record is retained
        but the paper link is cleared.

    read:
        Whether the user has acknowledged this notification.
        Toggled by PATCH /notifications/{id}/read.
        Bulk-toggled by PATCH /notifications/read-all.

    The unread_count in the API response is a separate COUNT query
    across ALL notifications for the user, not just the current page.
    """

    __tablename__ = "notifications"

    __table_args__ = (
        # Primary access pattern: user's notifications, newest first
        Index(
            "idx_notifications_user_created",
            "user_id",
            "created_at",
        ),
        # Filter pattern: unread notifications for bell badge count
        Index(
            "idx_notifications_user_unread",
            "user_id",
            "read",
        ),
    )

    # ------------------------------------------------------------------
    # Foreign keys
    # ------------------------------------------------------------------
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="User this notification belongs to.",
    )

    paper_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("papers.id", ondelete="SET NULL"),
        nullable=True,
        default=None,
        index=True,
        comment=(
            "Paper that triggered this notification. "
            "Null for system notifications. "
            "SET NULL on paper deletion."
        ),
    )

    # ------------------------------------------------------------------
    # Notification content
    # ------------------------------------------------------------------
    type: Mapped[str] = mapped_column(
        nullable=False,
        comment=(
            "Notification type. One of: "
            "'rag_ready', 'rag_failed', 'system'."
        ),
    )

    title: Mapped[str] = mapped_column(
        nullable=False,
        comment="Short notification title shown in the bell dropdown.",
    )

    body: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full notification body text.",
    )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    read: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment=(
            "Whether the user has acknowledged this notification. "
            "False = unread (contributes to bell badge count). "
            "True = read."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user: Mapped["User"] = relationship(
        "User",
        back_populates="notifications",
        lazy="noload",
    )

    paper: Mapped[Optional["Paper"]] = relationship(
        "Paper",
        back_populates="notifications",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<Notification "
            f"id={self.id!r} "
            f"user_id={self.user_id!r} "
            f"type={self.type!r} "
            f"read={self.read!r}>"
        )
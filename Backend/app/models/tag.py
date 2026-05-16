"""
app/models/tag.py

User-defined tags on papers. Tags are personal — each user
manages their own tags independently on any paper.

Tag labels are normalised to lowercase + stripped before storage
(enforced by tag_service, not a DB constraint).

Deduplication:
    The unique constraint (user_id, paper_id, label) prevents
    duplicate tags at the DB level. add_tag is idempotent —
    INSERT ... ON CONFLICT DO NOTHING.
"""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.paper import Paper
    from app.models.user import User


class Tag(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A single user-defined tag applied to a paper.

    Scope:
        Tags are personal. User A's tags on Paper X are invisible
        to User B. There is no global/shared tagging system in v1.

    Label normalisation:
        Enforced in tag_service.add_tag():
            label = label.strip().lower()[:50]
        Stored pre-normalised. No normalisation at DB level.

    Dashboard filtering:
        GET /dashboard/papers?tag=transformer
        Translated to:
            EXISTS (
                SELECT 1 FROM tags
                WHERE user_id = :user_id
                AND paper_id = papers.id
                AND label = :tag
            )
    """

    __tablename__ = "tags"

    __table_args__ = (
        # DB-level uniqueness: one tag label per (user, paper)
        UniqueConstraint(
            "user_id",
            "paper_id",
            "label",
            name="ux_tags_user_paper_label",
        ),
        # Access pattern: all tags for a user on a paper
        Index(
            "idx_tags_user_paper",
            "user_id",
            "paper_id",
        ),
        # Access pattern: dashboard filter by label
        Index(
            "idx_tags_label",
            "label",
        ),
    )

    # ------------------------------------------------------------------
    # Foreign keys
    # ------------------------------------------------------------------
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="User who created this tag.",
    )

    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        comment="Paper this tag is applied to.",
    )

    # ------------------------------------------------------------------
    # Tag content
    # ------------------------------------------------------------------
    label: Mapped[str] = mapped_column(
        nullable=False,
        comment=(
            "Tag label. Pre-normalised to lowercase and trimmed "
            "by tag_service before storage. Max 50 characters."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user: Mapped["User"] = relationship(
        "User",
        back_populates="tags",
        lazy="noload",
    )

    paper: Mapped["Paper"] = relationship(
        "Paper",
        back_populates="tags",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<Tag "
            f"id={self.id!r} "
            f"user_id={self.user_id!r} "
            f"paper_id={self.paper_id!r} "
            f"label={self.label!r}>"
        )
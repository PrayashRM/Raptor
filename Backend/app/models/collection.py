"""
app/models/collection.py

Personal paper collections (folders) owned by a user.

Visibility:
    v1 supports 'private' only.
    'public' with shareable link is reserved for v2.

paper_count:
    Denormalised count updated on add/remove paper operations.
    Avoids a COUNT query on every collection list request.
    Kept consistent by collection_service (not a DB trigger).
"""

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.collection_paper import CollectionPaper
    from app.models.user import User


class Collection(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A personal paper collection (folder) owned by a single user.

    Collections are private in v1. Public sharing (shareable link,
    GET /collections/{id}/public) is reserved for v2 and will be
    implemented by adding visibility='public' + share_token column.

    paper_count:
        Denormalised. Updated by collection_service on every
        add_paper / remove_paper / delete_collection operation.
        Used directly in Collection API responses without a COUNT join.
    """

    __tablename__ = "collections"

    __table_args__ = (
        # Primary access pattern: user's collections sorted by updated_at
        Index(
            "idx_collections_user_updated",
            "user_id",
            "updated_at",
        ),
    )

    # ------------------------------------------------------------------
    # Foreign keys
    # ------------------------------------------------------------------
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who owns this collection.",
    )

    # ------------------------------------------------------------------
    # Collection metadata
    # ------------------------------------------------------------------
    name: Mapped[str] = mapped_column(
        nullable=False,
        comment="Collection name. Mutable via PATCH /collections/{id}.",
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="Optional longer description of the collection.",
    )

    visibility: Mapped[str] = mapped_column(
        nullable=False,
        default="private",
        comment=(
            "Collection visibility. "
            "'private' in v1 (only owner can access). "
            "'public' reserved for v2 (shareable link)."
        ),
    )

    # ------------------------------------------------------------------
    # Denormalised counts
    # ------------------------------------------------------------------
    paper_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment=(
            "Denormalised count of papers in this collection. "
            "Updated by collection_service on add/remove/delete. "
            "Avoids COUNT query on every list request."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user: Mapped["User"] = relationship(
        "User",
        back_populates="collections",
        lazy="noload",
    )

    collection_papers: Mapped[List["CollectionPaper"]] = relationship(
        "CollectionPaper",
        back_populates="collection",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<Collection "
            f"id={self.id!r} "
            f"user_id={self.user_id!r} "
            f"name={self.name!r} "
            f"paper_count={self.paper_count!r}>"
        )
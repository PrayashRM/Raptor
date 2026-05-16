"""
app/models/collection_paper.py

Join table between collections and papers.
Represents membership of a paper in a collection.

No surrogate primary key — uses composite PK (collection_id, paper_id).
This enforces uniqueness at the DB level and is the most efficient
structure for a pure join table with no extra payload beyond added_at.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.collection import Collection
    from app.models.paper import Paper


class CollectionPaper(Base):
    """
    Membership record: a paper belonging to a collection.

    Composite primary key (collection_id, paper_id) enforces
    uniqueness — a paper can only appear once in a collection.
    The add_paper operation is idempotent via INSERT ... ON CONFLICT DO NOTHING.

    Cascade behaviour:
        If the collection is deleted → this record is deleted
        (cascade from Collection.collection_papers relationship).
        If the paper is deleted → this record is deleted
        (ondelete="CASCADE" on paper_id FK).
        Papers and collections themselves are never touched.
    """

    __tablename__ = "collection_papers"

    __table_args__ = (
        # Reverse lookup: "which collections contain this paper?"
        Index(
            "idx_collection_papers_paper_id",
            "paper_id",
        ),
    )

    # ------------------------------------------------------------------
    # Composite primary key
    # ------------------------------------------------------------------
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"),
        primary_key=True,
        comment="Collection this membership record belongs to.",
    )

    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        primary_key=True,
        comment="Paper that is a member of the collection.",
    )

    # ------------------------------------------------------------------
    # Membership metadata
    # ------------------------------------------------------------------
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Timestamp when this paper was added to the collection.",
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    collection: Mapped["Collection"] = relationship(
        "Collection",
        back_populates="collection_papers",
        lazy="noload",
    )

    paper: Mapped["Paper"] = relationship(
        "Paper",
        back_populates="collection_papers",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<CollectionPaper "
            f"collection_id={self.collection_id!r} "
            f"paper_id={self.paper_id!r} "
            f"added_at={self.added_at!r}>"
        )
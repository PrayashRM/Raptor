"""
app/models/refresh_token.py

Stored refresh token records for JWT rotation and revocation.

Security model:
    Raw refresh tokens are NEVER stored in the database.
    Only the SHA-256 hash of the raw token is stored.
    This means a database breach does not expose usable tokens.

Token rotation:
    Every use of a refresh token immediately invalidates it and
    issues a new one. The old hash is deleted, a new hash is inserted.

Replay detection:
    If a previously used (deleted) refresh token is presented again,
    it means either:
        a) The token was stolen and used by an attacker, OR
        b) A legitimate client somehow replayed an old token.
    In either case, the entire token family is revoked (all sessions
    for that user are logged out) as a security measure.

family_id:
    Groups all rotation descendants of an original login into a
    single family. On replay detection, DELETE WHERE family_id = :id
    logs out all active sessions for this user.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.user import User


class RefreshToken(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A stored refresh token record.

    One record = one active session on one device.
    Multiple records per user = multiple active sessions.

    token_hash:
        SHA-256 hex digest of the raw refresh token JWT.
        Lookup is by hash: SELECT * FROM refresh_tokens WHERE token_hash = :hash.
        Never store or log the raw token.

    family_id:
        UUID assigned at login and inherited through all rotations.
        Used for family-level revocation on replay detection.
        All tokens in the same family share the same family_id.

    expires_at:
        Hard expiry of this token. Tokens past this timestamp are
        rejected even if the DB record still exists.
        DB records past expires_at are cleaned up by the cleanup ARQ task.
    """

    __tablename__ = "refresh_tokens"

    __table_args__ = (
        # Primary lookup: find token by hash (login/refresh flow)
        Index(
            "idx_refresh_tokens_token_hash",
            "token_hash",
            unique=True,
        ),
        # Revocation: revoke all tokens for a user (logout-all)
        Index(
            "idx_refresh_tokens_user_id",
            "user_id",
        ),
        # Family revocation: replay detection cleanup
        Index(
            "idx_refresh_tokens_family_id",
            "family_id",
        ),
        # Cleanup task: find and delete expired tokens
        Index(
            "idx_refresh_tokens_expires_at",
            "expires_at",
        ),
    )

    # ------------------------------------------------------------------
    # Foreign keys
    # ------------------------------------------------------------------
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="User this refresh token belongs to.",
    )

    # ------------------------------------------------------------------
    # Token fields
    # ------------------------------------------------------------------
    token_hash: Mapped[str] = mapped_column(
        nullable=False,
        unique=True,
        comment=(
            "SHA-256 hex digest of the raw refresh token JWT. "
            "The raw token is never stored. "
            "Lookup: SELECT ... WHERE token_hash = sha256(raw_token)."
        ),
    )

    family_id: Mapped[str] = mapped_column(
        nullable=False,
        comment=(
            "UUID grouping all rotation descendants of one login event. "
            "Assigned at login, inherited through all rotations. "
            "On replay detection: DELETE WHERE family_id = :id "
            "revokes all sessions in this family."
        ),
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment=(
            "Hard expiry timestamp. Tokens past this are rejected. "
            "Default: 30 days from creation (REFRESH_TOKEN_EXPIRE_DAYS). "
            "Expired records deleted by cleanup ARQ task."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user: Mapped["User"] = relationship(
        "User",
        back_populates="refresh_tokens",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<RefreshToken "
            f"id={self.id!r} "
            f"user_id={self.user_id!r} "
            f"family_id={self.family_id!r} "
            f"expires_at={self.expires_at!r}>"
        )
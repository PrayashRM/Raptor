"""
app/models/verification_token.py

Single-use time-limited tokens for:
    - Email address verification (type='email_verify')
    - Password reset (type='password_reset')

Security:
    Raw tokens are NEVER stored.
    Only SHA-256 hash is stored (same pattern as refresh_token.py).
    Token is marked used_at on consumption — subsequent use is rejected.

Expiry:
    email_verify  → 24 hours from creation
    password_reset → 1 hour from creation
    Both enforced in auth_service by checking expires_at.
    Expired records deleted by cleanup ARQ task.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.user import User


class VerificationToken(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A single-use verification token for email verification
    or password reset flows.

    type:
        'email_verify'   → used in POST /auth/verify-email flow
        'password_reset' → used in POST /auth/reset-password flow

    token_hash:
        SHA-256 hex digest of the raw token string.
        Raw token is sent in the email link and never stored here.

    used_at:
        Set to now() when the token is successfully consumed.
        Any token with used_at IS NOT NULL is rejected as already used.
        Null means the token is unused and potentially valid.

    expires_at:
        Hard expiry. Checked in auth_service before accepting token.
        email_verify: 24 hours. password_reset: 1 hour.
    """

    __tablename__ = "verification_tokens"

    __table_args__ = (
        # Primary lookup: find token by hash
        Index(
            "idx_verification_tokens_hash",
            "token_hash",
            unique=True,
        ),
        # Cleanup task: find expired tokens to delete
        Index(
            "idx_verification_tokens_expires_at",
            "expires_at",
        ),
        # User lookup: revoke all tokens for a user on password change
        Index(
            "idx_verification_tokens_user_id",
            "user_id",
        ),
    )

    # ------------------------------------------------------------------
    # Foreign keys
    # ------------------------------------------------------------------
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="User this token was issued for.",
    )

    # ------------------------------------------------------------------
    # Token fields
    # ------------------------------------------------------------------
    token_hash: Mapped[str] = mapped_column(
        nullable=False,
        unique=True,
        comment=(
            "SHA-256 hex digest of the raw token string sent in the email link. "
            "The raw token is never stored in the database."
        ),
    )

    type: Mapped[str] = mapped_column(
        nullable=False,
        comment=(
            "Token purpose. One of: "
            "'email_verify' (24h expiry), "
            "'password_reset' (1h expiry)."
        ),
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment=(
            "Hard expiry timestamp. "
            "email_verify: 24 hours from creation. "
            "password_reset: 1 hour from creation. "
            "Checked in auth_service before accepting the token."
        ),
    )

    used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment=(
            "Set to now() when the token is successfully consumed. "
            "Any token with used_at IS NOT NULL is rejected as already used. "
            "Null = unused and potentially valid."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user: Mapped["User"] = relationship(
        "User",
        back_populates="verification_tokens",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<VerificationToken "
            f"id={self.id!r} "
            f"user_id={self.user_id!r} "
            f"type={self.type!r} "
            f"used_at={self.used_at!r} "
            f"expires_at={self.expires_at!r}>"
        )
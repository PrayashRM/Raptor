"""
app/models/base.py

SQLAlchemy 2.0 declarative base and shared mixins.

All models inherit from:
    Base                → SQLAlchemy DeclarativeBase
    UUIDPrimaryKeyMixin → UUID v7 primary key (monotonic, sortable)
    TimestampMixin      → created_at + updated_at (auto-managed)

UUID v7:
    UUID v7 is time-ordered — IDs generated later have higher values.
    This makes them ideal for primary keys because:
    1. B-tree index inserts are always at the end (no page splits)
    2. IDs are naturally sortable by creation time
    3. Cursor-based pagination works by comparing UUIDs directly

    We use python-ulid to generate ULID-compatible UUID v7 values.
    These are stored as VARCHAR in PostgreSQL (not native UUID type)
    because SQLAlchemy + asyncpg handle VARCHAR more predictably
    for ULID-style strings.

    If you prefer native UUID columns, change the type annotation
    and generation function accordingly.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)
from ulid import ULID


def generate_ulid() -> str:
    """
    Generate a new ULID string.

    ULIDs are:
        - 26 characters, Crockford Base32 encoded
        - Time-ordered (monotonically increasing)
        - Unique without coordination
        - URL-safe
        - Sortable as strings

    Used as the default primary key for all models.
    """
    return str(ULID())


class Base(DeclarativeBase):
    """
    SQLAlchemy 2.0 declarative base.

    All models inherit from this class.
    Alembic env.py imports Base.metadata to detect all tables.

    type_annotation_map:
        Maps Python types to SQL column types globally.
        Individual models can override these per-column.
    """

    type_annotation_map = {
        str: String(255),
        datetime: DateTime(timezone=True),
    }


class UUIDPrimaryKeyMixin:
    """
    Mixin providing a ULID primary key column.

    Column: id
        Type:    VARCHAR (stores ULID string, e.g. "01J9XK234PQRSTUVWXYZ")
        Default: generated via generate_ulid() in Python
        Primary key, not nullable, indexed automatically

    Usage:
        class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
            __tablename__ = "users"
            name: Mapped[str] = mapped_column(...)
    """

    id: Mapped[str] = mapped_column(
        String(26),
        primary_key=True,
        default=generate_ulid,
        nullable=False,
        comment="ULID primary key. Time-ordered, globally unique.",
    )


class TimestampMixin:
    """
    Mixin providing created_at and updated_at columns.

    created_at:
        Set automatically on INSERT (server_default=func.now()).
        Never modified after creation.

    updated_at:
        Set on INSERT (server_default=func.now()).
        Updated automatically on every UPDATE (onupdate=func.now()).
        Also set explicitly in Python as a fallback.

    Both columns use TIMESTAMPTZ (timestamp with time zone).
    All timestamps are stored and returned in UTC.
    The PostgreSQL database timezone is set to UTC in init-db.sql.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        comment="Row creation timestamp (UTC). Set once, never modified.",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        default=lambda: datetime.now(timezone.utc),
        comment="Last modification timestamp (UTC). Updated on every write.",
    )
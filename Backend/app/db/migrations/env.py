"""
app/db/migrations/env.py

Alembic migration environment.

This file controls HOW Alembic runs migrations. It:
    1. Reads DATABASE_URL_SYNC from the environment
    2. Imports all models via app.models (so metadata is complete)
    3. Configures Alembic to compare against our model metadata
    4. Runs migrations either offline (SQL script) or online (live DB)

IMPORTANT:
    Alembic migrations run with a SYNCHRONOUS database driver
    (psycopg2 via DATABASE_URL_SYNC), not the async driver (asyncpg).
    This is because Alembic's migration runner is synchronous.
    The async engine in app/db/session.py is for the FastAPI runtime only.

    DATABASE_URL_SYNC format: postgresql://user:pass@host:port/db
    DATABASE_URL format:      postgresql+asyncpg://user:pass@host:port/db
"""


# Root determination for imports
import sys
import os
from pathlib import Path

# ------------------------------------------------------------------
# CRITICAL: Add project root to sys.path
# Inside Docker: WORKDIR=/app, so project root IS /app
# This file lives at /app/app/db/migrations/env.py
# We need /app on sys.path so "from app.models import Base" works
# ------------------------------------------------------------------
# This finds /app by going up 4 levels from this file:
# env.py        → migrations/
# migrations/   → db/
# db/           → app/       (the package)
# app/          → /app       (the project root — THIS is what we need)
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent.parent.parent
# Verify we found the right directory
assert (_project_root / "app" / "__init__.py").exists(), (
    f"Project root detection failed. "
    f"Detected: {_project_root}. "
    f"Expected to find app/__init__.py there. "
    f"sys.path: {sys.path}"
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))



# ------------------------------------------------------------------
# importing app modules
# ------------------------------------------------------------------
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text
from alembic.autogenerate import render
from pgvector.sqlalchemy import Vector


# ---------------------------------------------------------------------------
# Import ALL models so that Base.metadata contains every table.
# This import triggers app/models/__init__.py which imports every model.
# Without this, Alembic cannot detect tables for --autogenerate.
# ---------------------------------------------------------------------------
from app.models import Base

# ---------------------------------------------------------------------------
# Alembic Config object — provides access to alembic.ini values
# ---------------------------------------------------------------------------
config = context.config

# ---------------------------------------------------------------------------
# Override sqlalchemy.url from environment variable
# This prevents hardcoding credentials in alembic.ini
# ---------------------------------------------------------------------------
database_url = os.environ.get("DATABASE_URL_SYNC")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)
else:
    raise RuntimeError(
        "DATABASE_URL_SYNC environment variable is not set. "
        "Alembic requires a synchronous PostgreSQL URL. "
        "Format: postgresql://user:pass@host:port/db"
    )

# ---------------------------------------------------------------------------
# Set up Python logging from alembic.ini [loggers] section
# ---------------------------------------------------------------------------
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Target metadata — Alembic compares this against the live database
# to determine what migrations need to be generated/applied
# ---------------------------------------------------------------------------
target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Tables to EXCLUDE from autogenerate
# (e.g. if you have tables managed outside of Alembic)
# ---------------------------------------------------------------------------
EXCLUDE_TABLES: set[str] = set()


def include_object(
    object,  # noqa: A002 — Alembic's API uses 'object' as param name
    name: str,
    type_: str,
    reflected: bool,
    compare_to,
) -> bool:
    """
    Filter function for autogenerate.
    Returns True to include the object in the migration, False to skip.
    """
    if type_ == "table" and name in EXCLUDE_TABLES:
        return False
    return True


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Generates SQL script output without connecting to the database.
    Useful for:
        - Reviewing migration SQL before applying
        - Generating SQL for DBA review in strict environments
        - CI/CD pipelines that apply SQL separately

    Usage:
        alembic upgrade head --sql > migration.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Connects to the live database and applies migrations directly.
    This is the normal mode used in development and deployment.

    After connecting, enables required PostgreSQL extensions
    (vector, pg_trgm, uuid-ossp, btree_gin) if they don't exist.
    These are idempotent — safe to run on every migration.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # ------------------------------------------------------------------
        # Ensure required extensions are enabled
        # These are also in scripts/init-db.sql but running here guarantees
        # they exist even if the DB was created outside of Docker
        # (e.g. AWS RDS where init-db.sql doesn't run)
        # ------------------------------------------------------------------
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        connection.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gin"))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Entry point — Alembic calls this
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
"""Alembic environment.

The URL comes from `DATABASE_URL` (via Pydantic settings), never from
`alembic.ini`, so no DSN is ever committed and the same migrations run against
SQLite locally and PostgreSQL in deployment.

The URL is normalised to a sync driver: Alembic drives DDL synchronously, and
the app's `postgresql+asyncpg://` / `sqlite+aiosqlite://` URLs would otherwise
fail here.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.models.tables import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _sync_url() -> str:
    """Resolve the database URL and force a synchronous driver."""
    url = get_settings().database_url
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Migrations need a target database, e.g.\n"
            "  DATABASE_URL=postgresql://support:support@localhost:5432/support_db\n"
            "  DATABASE_URL=sqlite:///./support.db"
        )
    # Normalise to an installed *synchronous* driver.
    #
    # For PostgreSQL that must be psycopg 3 (`postgresql+psycopg`). A bare
    # `postgresql://` makes SQLAlchemy default to psycopg2, which this project
    # does not install -- so the driver has to be named explicitly rather than
    # stripped.
    url = url.replace("+aiosqlite", "")
    if url.startswith("postgres"):
        for async_driver in ("+asyncpg", "+psycopg_async"):
            url = url.replace(async_driver, "")
        if "+" not in url.split("://", 1)[0]:
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting -- for reviewing DDL before applying."""
    context.configure(
        url=_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _sync_url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # SQLite cannot ALTER most columns; batch mode rewrites the table
            # instead, so one migration script works on both backends.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

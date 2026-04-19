"""Alembic env synchrone (psycopg) — appelé via `alembic upgrade head`."""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ajoute v4/ au sys.path pour pouvoir importer backend.app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.db.base import Base  # noqa: E402
from backend.app.db import models  # noqa: E402,F401  (enregistre les modèles)
from backend.app.settings import get_settings  # noqa: E402


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _sync_db_url() -> str:
    """Convertit l'URL asyncpg en URL psycopg pour Alembic."""
    url = str(get_settings().database_url)
    return url.replace("+asyncpg", "+psycopg")


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=_sync_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section) or {}
    cfg["sqlalchemy.url"] = _sync_db_url()
    connectable = engine_from_config(
        cfg, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

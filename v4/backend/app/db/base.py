"""Base SQLAlchemy + factory de session async.

Postgres + pgvector + tsvector. Utilise `asyncpg` en runtime, `psycopg`
pour Alembic (synchrone).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from ..settings import get_settings


class Base(DeclarativeBase):
    """Base déclarative pour tous les modèles ORM."""

    metadata_naming_convention: dict[str, Any] = {}


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            str(settings.database_url),
            pool_size=10,
            max_overflow=10,
            pool_pre_ping=True,
            future=True,
        )
        _sessionmaker = async_sessionmaker(
            _engine, expire_on_commit=False, class_=AsyncSession
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dépendance FastAPI : ouvre une session par requête."""
    sm = get_sessionmaker()
    async with sm() as session:
        yield session

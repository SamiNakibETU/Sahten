"""Tâches arq exécutées par le worker.

Fonctions enregistrées :
  - ingest_one(article_external_id)         : ingestion + reindexation
  - ingest_backfill(category=None)          : backfill complet WhiteBeard

Lancer le worker :
    arq worker.tasks.WorkerSettings
"""

from __future__ import annotations

from arq.connections import RedisSettings

from app.db.base import get_sessionmaker
from app.ingestion.service import backfill_all, ingest_article_id
from app.ingestion.whitebeard_client import WhiteBeardClient
from app.rag.embeddings import OpenAIEmbeddings
from app.rag.indexer import reindex_article
from app.settings import get_settings


async def startup(ctx: dict) -> None:
    ctx["session_factory"] = get_sessionmaker()
    ctx["embedder"] = OpenAIEmbeddings()


async def shutdown(ctx: dict) -> None:
    pass


async def ingest_one(ctx: dict, article_external_id: int) -> dict:
    sm = ctx["session_factory"]
    embedder = ctx["embedder"]
    async with WhiteBeardClient() as cli:
        async with sm() as session:
            result = await ingest_article_id(
                session, article_external_id, client=cli
            )
            article = await session.get(
                __import__("app.db.models", fromlist=["Article"]).Article,
                result.article_id,
            )
            n_chunks = await reindex_article(session, article, embedder)
            await session.commit()
    return {"article_id": result.article_id, "status": result.status, "chunks": n_chunks}


async def ingest_backfill(
    ctx: dict, category: str | None = None, page_size: int = 50
) -> dict:
    sm = ctx["session_factory"]
    counts = await backfill_all(sm, category=category, page_size=page_size)
    return counts


class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(str(get_settings().redis_url))
    functions = [ingest_one, ingest_backfill]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = get_settings().arq_queue
    job_timeout = 600

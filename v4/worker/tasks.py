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
from app.ingestion.service import ingest_article_id
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
    from app.db import models

    sm = ctx["session_factory"]
    embedder = ctx["embedder"]
    counts = {"ok": 0, "partial": 0, "failed": 0, "needs_playwright": 0, "chunks": 0}
    async with WhiteBeardClient() as cli:
        async for item in cli.iter_all_articles(
            category=category, page_size=page_size
        ):
            external_id = int(item.get("id") or 0)
            if not external_id:
                continue
            try:
                async with sm() as session:
                    result = await ingest_article_id(
                        session, external_id, client=cli
                    )
                    article = await session.get(models.Article, result.article_id)
                    if article is not None:
                        n_chunks = await reindex_article(session, article, embedder)
                        counts["chunks"] = counts.get("chunks", 0) + n_chunks
                    await session.commit()
                counts[result.status] = counts.get(result.status, 0) + 1
            except Exception:  # noqa: BLE001
                counts["failed"] += 1
    return counts


class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(str(get_settings().redis_url))
    functions = [ingest_one, ingest_backfill]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = get_settings().arq_queue
    job_timeout = 600

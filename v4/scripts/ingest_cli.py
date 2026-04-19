"""CLI ingestion (utile en local + en CI Railway pour le backfill initial).

Usage :
    python -m scripts.ingest_cli one 1227694
    python -m scripts.ingest_cli backfill --category Cuisine --max-pages 10
    python -m scripts.ingest_cli reindex 1227694
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.db.base import get_sessionmaker
from app.db.models import Article
from app.ingestion.service import backfill_all, ingest_article_id
from app.ingestion.whitebeard_client import WhiteBeardClient
from app.rag.embeddings import OpenAIEmbeddings
from app.rag.indexer import reindex_article


async def cmd_one(article_id: int) -> None:
    sm = get_sessionmaker()
    embedder = OpenAIEmbeddings()
    async with WhiteBeardClient() as cli:
        async with sm() as session:
            res = await ingest_article_id(session, article_id, client=cli)
            article = await session.get(Article, res.article_id)
            n = await reindex_article(session, article, embedder)
            await session.commit()
    print(f"ok status={res.status} article_id={res.article_id} chunks={n}")


async def cmd_backfill(category: str | None, max_pages: int | None) -> None:
    sm = get_sessionmaker()
    counts = await backfill_all(sm, category=category, max_pages=max_pages)
    print(counts)


async def cmd_reindex(article_external_id: int) -> None:
    from sqlalchemy import select

    sm = get_sessionmaker()
    embedder = OpenAIEmbeddings()
    async with sm() as session:
        res = await session.execute(
            select(Article).where(Article.external_id == article_external_id)
        )
        article = res.scalar_one_or_none()
        if not article:
            print(f"article {article_external_id} introuvable", file=sys.stderr)
            sys.exit(1)
        n = await reindex_article(session, article, embedder)
        await session.commit()
    print(f"reindexed chunks={n}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p_one = sub.add_parser("one")
    p_one.add_argument("article_id", type=int)
    p_b = sub.add_parser("backfill")
    p_b.add_argument("--category")
    p_b.add_argument("--max-pages", type=int)
    p_r = sub.add_parser("reindex")
    p_r.add_argument("article_id", type=int)

    args = p.parse_args()
    if args.cmd == "one":
        asyncio.run(cmd_one(args.article_id))
    elif args.cmd == "backfill":
        asyncio.run(cmd_backfill(args.category, args.max_pages))
    elif args.cmd == "reindex":
        asyncio.run(cmd_reindex(args.article_id))


if __name__ == "__main__":
    main()

"""CLI ingestion (utile en local + en CI Railway pour le backfill initial).

Usage :
    python -m scripts.ingest_cli one 1227694
    python -m scripts.ingest_cli backfill --category Cuisine --max-pages 10
    python -m scripts.ingest_cli from-ids --file data/olj_seed_ids.json
    python -m scripts.ingest_cli reindex 1227694

Note : `backfill` appelle `LIST /content` qui exige une clé WhiteBeard admin.
Si tu as une clé "lecture par ID" uniquement, utilise `from-ids` à la place
(boucle sur `GET /content/{id}` autorisé).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

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


def _load_ids(path: Path) -> list[int]:
    """Charge une liste d'IDs depuis un .json (array ou {ids: [...]}) ou .txt (un ID/ligne)."""
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    # `utf-8-sig` accepte aussi bien avec ou sans BOM (PowerShell en ajoute un).
    raw = path.read_text(encoding="utf-8-sig").strip()
    ids: list[int] = []
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, dict) and "ids" in data:
            data = data["ids"]
        if not isinstance(data, list):
            raise ValueError(f"Le JSON doit être une liste d'IDs (ou {{ids: [...]}}). Reçu : {type(data)}")
        ids = [int(x) for x in data if x is not None]
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(int(line))
    seen: set[int] = set()
    out: list[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


async def cmd_from_ids(file_path: str, *, embed: bool, skip_existing: bool, limit: int | None) -> None:
    """Ingère une liste d'IDs (un par un, via GET /content/{id} → autorisé clé non-admin)."""
    from sqlalchemy import select

    path = Path(file_path)
    ids = _load_ids(path)
    if limit:
        ids = ids[:limit]
    print(f"[from-ids] {len(ids)} IDs à traiter depuis {path}")

    sm = get_sessionmaker()
    embedder = OpenAIEmbeddings() if embed else None

    counts = {"ok": 0, "partial": 0, "failed": 0, "skipped": 0, "chunks": 0}

    async with WhiteBeardClient() as cli:
        for idx, ext_id in enumerate(ids, 1):
            async with sm() as session:
                try:
                    if skip_existing:
                        existing = await session.execute(
                            select(Article.id).where(Article.external_id == ext_id)
                        )
                        if existing.scalar_one_or_none() is not None:
                            counts["skipped"] += 1
                            print(f"  [{idx}/{len(ids)}] {ext_id} — déjà présent, skip")
                            continue

                    res = await ingest_article_id(session, ext_id, client=cli)
                    n_chunks = 0
                    if embedder is not None and res.status in ("ok", "partial"):
                        article = await session.get(Article, res.article_id)
                        if article is not None:
                            n_chunks = await reindex_article(session, article, embedder)
                            counts["chunks"] += n_chunks
                    await session.commit()
                    counts[res.status] = counts.get(res.status, 0) + 1
                    print(f"  [{idx}/{len(ids)}] {ext_id} — status={res.status} chunks={n_chunks}")
                except Exception as exc:  # noqa: BLE001
                    await session.rollback()
                    counts["failed"] += 1
                    print(f"  [{idx}/{len(ids)}] {ext_id} — ERREUR: {exc}", file=sys.stderr)

    print(f"[from-ids] terminé : {counts}")


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
    p_b = sub.add_parser("backfill", help="LIST /content (clé admin requise)")
    p_b.add_argument("--category")
    p_b.add_argument("--max-pages", type=int)
    p_f = sub.add_parser(
        "from-ids",
        help="Boucle GET /content/{id} (compatible clé non-admin)",
    )
    p_f.add_argument("--file", required=True, help="Fichier .json (array d'IDs) ou .txt (un ID/ligne)")
    p_f.add_argument(
        "--no-embed",
        action="store_true",
        help="Ne pas générer les embeddings (utile pour ingestion rapide)",
    )
    p_f.add_argument(
        "--skip-existing",
        action="store_true",
        help="Sauter les articles déjà présents en base",
    )
    p_f.add_argument("--limit", type=int, help="N'ingérer que les N premiers IDs")
    p_r = sub.add_parser("reindex")
    p_r.add_argument("article_id", type=int)

    args = p.parse_args()
    if args.cmd == "one":
        asyncio.run(cmd_one(args.article_id))
    elif args.cmd == "backfill":
        asyncio.run(cmd_backfill(args.category, args.max_pages))
    elif args.cmd == "from-ids":
        asyncio.run(
            cmd_from_ids(
                args.file,
                embed=not args.no_embed,
                skip_existing=args.skip_existing,
                limit=args.limit,
            )
        )
    elif args.cmd == "reindex":
        asyncio.run(cmd_reindex(args.article_id))


if __name__ == "__main__":
    main()

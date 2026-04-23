"""CLI ingestion (utile en local + en CI Railway pour le backfill initial).

Usage :
    python -m scripts.ingest_cli one 1227694
    python -m scripts.ingest_cli backfill --category Cuisine --max-pages 10
    python -m scripts.ingest_cli from-ids --file data/olj_seed_ids.json
    python -m scripts.ingest_cli reindex 1227694
    python -m scripts.ingest_cli reindex-all --publication 17 --content-type 4

Note : `backfill` appelle `LIST /content` qui exige une clé WhiteBeard admin.
Si tu as une clé "lecture par ID" uniquement, utilise `from-ids` à la place
(boucle sur `GET /content/{id}` autorisé).

`reindex-all` est la commande à utiliser pour la mise à jour complète de la
rubrique recettes après un changement de mapper/chunker : elle liste tous
les articles via ``/publication/17/content?content_type=4`` puis ré-ingère
ET ré-indexe chacun (chunks + embeddings). Idempotente : l'upsert ne crée
pas de doublon.
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


async def cmd_reindex_all(
    *,
    publication_id: int,
    content_type: int | None,
    page_size: int,
    max_pages: int | None,
    limit: int | None,
    skip_ingest: bool,
    dry_run: bool,
    seed_file: str | None = None,
    seed_only: bool = False,
) -> None:
    """Re-ingère **et** ré-indexe tous les articles d'une publication.

    Workflow :
      1. ``GET /publication/{publication_id}/content?content_type=…`` paginé,
      2. Pour chaque ID : ``ingest_article_id`` (avec enrichissement chef bio),
      3. Puis ``reindex_article`` pour produire les chunks + embeddings.

    Idempotent : l'upsert se base sur ``Article.external_id`` ; les chunks
    existants sont remplacés par ``reindex_article``.
    """
    sm = get_sessionmaker()
    embedder = OpenAIEmbeddings()

    counts: dict[str, int] = {"ok": 0, "partial": 0, "failed": 0, "chunks": 0}
    ids_seen: list[int] = []
    seen_set: set[int] = set()

    def _add(ext: int) -> bool:
        if ext in seen_set:
            return False
        seen_set.add(ext)
        ids_seen.append(ext)
        return True

    async with WhiteBeardClient() as cli:
        if not seed_only:
            print(
                f"[reindex-all] listing publication={publication_id} "
                f"content_type={content_type} page_size={page_size}"
            )
            async for ext_id in cli.iter_publication_ids(
                publication_id=publication_id,
                content_type=content_type,
                page_size=page_size,
                max_pages=max_pages,
            ):
                _add(ext_id)
                if limit is not None and len(ids_seen) >= limit:
                    break
            print(f"[reindex-all] {len(ids_seen)} IDs depuis l'API")

        if seed_file:
            seed_ids = _load_ids(Path(seed_file))
            before = len(ids_seen)
            for sid in seed_ids:
                if limit is not None and len(ids_seen) >= limit:
                    break
                _add(sid)
            print(
                f"[reindex-all] {len(seed_ids)} IDs lus depuis {seed_file} "
                f"(+{len(ids_seen) - before} après dédoublonnage)"
            )
        print(f"[reindex-all] {len(ids_seen)} articles uniques à traiter")
        if dry_run:
            print(f"[reindex-all] DRY RUN — IDs : {ids_seen[:20]}{'…' if len(ids_seen) > 20 else ''}")
            return

        for idx, ext_id in enumerate(ids_seen, 1):
            async with sm() as session:
                try:
                    if not skip_ingest:
                        res = await ingest_article_id(session, ext_id, client=cli)
                        article_obj = await session.get(Article, res.article_id)
                        status = res.status
                    else:
                        from sqlalchemy import select

                        sel = await session.execute(
                            select(Article).where(Article.external_id == ext_id)
                        )
                        article_obj = sel.scalar_one_or_none()
                        if article_obj is None:
                            counts["failed"] += 1
                            print(f"  [{idx}/{len(ids_seen)}] {ext_id} — absent (skip-ingest)", file=sys.stderr)
                            continue
                        status = "ok"

                    n_chunks = 0
                    if article_obj is not None:
                        n_chunks = await reindex_article(session, article_obj, embedder)
                        counts["chunks"] += n_chunks
                    await session.commit()
                    counts[status] = counts.get(status, 0) + 1
                    print(
                        f"  [{idx}/{len(ids_seen)}] {ext_id} — status={status} chunks={n_chunks}"
                    )
                except Exception as exc:  # noqa: BLE001
                    await session.rollback()
                    counts["failed"] += 1
                    print(f"  [{idx}/{len(ids_seen)}] {ext_id} — ERREUR: {exc}", file=sys.stderr)

    print(f"[reindex-all] terminé : {counts}")


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

    p_ra = sub.add_parser(
        "reindex-all",
        help="Re-ingestion + ré-indexation complète d'une publication WhiteBeard",
    )
    p_ra.add_argument("--publication", type=int, default=17, help="ID publication (défaut 17 = À table)")
    p_ra.add_argument(
        "--content-type", type=int, default=4,
        help="content_type (défaut 4 = Recipes ; passer 0 pour tout type)",
    )
    p_ra.add_argument("--page-size", type=int, default=50)
    p_ra.add_argument("--max-pages", type=int, default=None)
    p_ra.add_argument("--limit", type=int, default=None, help="Borne dure sur le nombre d'articles à traiter")
    p_ra.add_argument(
        "--skip-ingest", action="store_true",
        help="Ne pas refetcher l'API : juste ré-indexer les chunks à partir de la DB",
    )
    p_ra.add_argument(
        "--seed-file", type=str, default=None,
        help="Fichier JSON/TXT d'IDs supplémentaires à fusionner (workaround pagination cassée)",
    )
    p_ra.add_argument(
        "--seed-only", action="store_true",
        help="Ignorer l'API publication et n'utiliser que --seed-file",
    )
    p_ra.add_argument("--dry-run", action="store_true")

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
    elif args.cmd == "reindex-all":
        ct = args.content_type if args.content_type and args.content_type > 0 else None
        asyncio.run(
            cmd_reindex_all(
                publication_id=args.publication,
                content_type=ct,
                page_size=args.page_size,
                max_pages=args.max_pages,
                limit=args.limit,
                skip_ingest=args.skip_ingest,
                dry_run=args.dry_run,
                seed_file=args.seed_file,
                seed_only=args.seed_only,
            )
        )


if __name__ == "__main__":
    main()

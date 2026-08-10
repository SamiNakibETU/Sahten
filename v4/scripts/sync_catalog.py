"""Synchronisation périodique catalogue OLJ -> base (remplace le webhook mort).

Historique : le CMS devait pousser chaque recette publiée via webhook
(``/api/webhook/recipe``), mais son automatisation pointe toujours sur
l'ancienne URL Railway — l'endpoint du serveur AWS n'a jamais reçu un appel
(constaté 2026-08-09 ; 13 recettes manquantes rattrapées à la main ce jour-là).
Ce script inverse le sens : il TIRE le catalogue et ingère ce qui manque.
Une nouvelle recette est par construction dans la première page de l'API
(tri anté-chronologique), donc un tirage périodique suffit ; le webhook
redevient un simple bonus de latence s'ils le repointent un jour.

Usage :
    python -m scripts.sync_catalog             # tirage + ingestion
    python -m scripts.sync_catalog --dry-run   # liste les manquants, n'ingère rien

Prévu pour tourner sous timer systemd (voir v4/infra/aws/sahten-sync.*).
Idempotent : zéro manquant -> zéro écriture, sortie en quelques secondes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_sessionmaker  # noqa: E402
from app.db.models import Article  # noqa: E402
from app.ingestion.service import ingest_article_id  # noqa: E402
from app.ingestion.whitebeard_client import WhiteBeardClient  # noqa: E402
from app.rag.embeddings import OpenAIEmbeddings  # noqa: E402
from app.rag.indexer import reindex_article  # noqa: E402

PUBLICATION_ID = 17  # "À table - OLJ"
CONTENT_TYPE = 4  # recettes


async def run(dry_run: bool) -> int:
    async with WhiteBeardClient() as cli:
        catalog_ids: list[int] = []
        async for ext_id in cli.iter_publication_ids(
            publication_id=PUBLICATION_ID, content_type=CONTENT_TYPE, page_size=100
        ):
            catalog_ids.append(int(ext_id))
        print(f"[sync] catalogue accessible : {len(catalog_ids)} recettes")

        sm = get_sessionmaker()
        async with sm() as session:
            rows = await session.execute(select(Article.external_id))
            in_db = {int(x) for x in rows.scalars().all()}
        missing = [i for i in catalog_ids if i not in in_db]
        print(f"[sync] en base : {len(in_db)} — manquants : {len(missing)} {missing}")

        if not missing:
            return 0
        if dry_run:
            print("[sync] dry-run : aucune ingestion")
            return 0

        embedder = OpenAIEmbeddings()
        ok = failed = 0
        for ext_id in missing:
            async with sm() as session:
                try:
                    res = await ingest_article_id(session, ext_id, client=cli)
                    n_chunks = 0
                    if res.status in ("ok", "partial"):
                        article = await session.get(Article, res.article_id)
                        if article is not None:
                            n_chunks = await reindex_article(session, article, embedder)
                    await session.commit()
                    ok += 1
                    print(f"[sync]   {ext_id} status={res.status} chunks={n_chunks}")
                except Exception as exc:  # noqa: BLE001
                    await session.rollback()
                    failed += 1
                    print(f"[sync]   {ext_id} ERREUR: {exc}", file=sys.stderr)
        print(f"[sync] terminé : {ok} ingérés, {failed} échecs")
        return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return asyncio.run(run(args.dry_run))


if __name__ == "__main__":
    sys.exit(main())

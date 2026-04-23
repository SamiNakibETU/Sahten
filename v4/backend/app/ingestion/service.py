"""Orchestration ingestion : fetch -> map -> upsert -> log -> chunk+embed."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models
from .mapper import map_article
from .repository import upsert_article
from .whitebeard_client import WhiteBeardClient

log = structlog.get_logger(__name__)


def _payload_item(payload: dict[str, Any]) -> dict[str, Any]:
    """Renvoie l'objet article quel que soit l'enrobage `{data: [...]}`."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list) and data:
        first = data[0]
        return first if isinstance(first, dict) else {}
    if isinstance(data, dict):
        return data
    return payload if isinstance(payload, dict) else {}


async def _enrich_chef_bio(
    payload: dict[str, Any], cli: WhiteBeardClient
) -> dict[str, Any]:
    """Si le payload référence un chef sans bio, fetcher sa fiche complète.

    Le payload ``/content/{recette_id}`` n'inclut PAS le champ ``contents``
    (bio HTML) du chef ; il faut un 2ᵉ ``GET /content/{chef_id}``. On
    patche directement l'objet ``chef`` du payload pour que le mapper
    récupère la bio comme s'il s'agissait d'un payload natif riche.
    """
    item = _payload_item(payload)
    chef = item.get("chef") if isinstance(item, dict) else None
    if not isinstance(chef, dict):
        return payload
    if isinstance(chef.get("contents"), str) and chef["contents"].strip():
        return payload
    chef_id = chef.get("id")
    try:
        chef_id_int = int(chef_id) if chef_id is not None else None
    except (TypeError, ValueError):
        chef_id_int = None
    if not chef_id_int:
        return payload
    try:
        chef_payload = await cli.fetch_content(chef_id_int)
    except Exception as exc:  # noqa: BLE001
        log.warning("ingest.chef_fetch_failed", chef_id=chef_id_int, error=str(exc))
        return payload
    chef_full = _payload_item(chef_payload)
    if isinstance(chef_full, dict) and chef_full:
        item["chef"] = chef_full
    return payload


@dataclass
class IngestionResult:
    article_id: int
    external_id: int
    status: str
    duration_ms: int
    notes: str | None


async def ingest_article_id(
    session: AsyncSession,
    external_id: int,
    *,
    client: WhiteBeardClient | None = None,
) -> IngestionResult:
    """Ingère un article par son external_id WhiteBeard."""
    own_client = client is None
    cli = client or WhiteBeardClient()
    started = time.perf_counter()
    error: str | None = None
    payload_size = 0
    article: models.Article | None = None
    status = "ok"

    try:
        payload = await cli.fetch_content(external_id)
        payload = await _enrich_chef_bio(payload, cli)
        payload_size = len(str(payload))
        mapped = map_article(payload)
        article = await upsert_article(session, mapped)
        status = mapped.ingestion_status
    except Exception as e:  # noqa: BLE001
        log.exception("ingest.failed", external_id=external_id)
        error = str(e)
        status = "failed"
        raise
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        session.add(
            models.IngestionLog(
                article_external_id=external_id,
                source="whitebeard",
                status=status,
                duration_ms=duration_ms,
                error_message=error,
                payload_size=payload_size,
            )
        )
        await session.commit()
        if own_client:
            await cli.close()

    assert article is not None
    return IngestionResult(
        article_id=article.id,
        external_id=external_id,
        status=status,
        duration_ms=duration_ms,
        notes=article.ingestion_notes,
    )


async def backfill_all(
    session_factory,
    *,
    page_size: int = 50,
    max_pages: int | None = None,
    category: str | None = None,
) -> dict[str, int]:
    """Backfill complet (toutes les pages /content)."""
    counts = {"ok": 0, "partial": 0, "failed": 0, "needs_playwright": 0}
    async with WhiteBeardClient() as cli:
        async for item in cli.iter_all_articles(
            category=category, page_size=page_size, max_pages=max_pages
        ):
            external_id = int(item.get("id") or 0)
            if not external_id:
                continue
            try:
                async with session_factory() as session:
                    res = await ingest_article_id(session, external_id, client=cli)
                counts[res.status] = counts.get(res.status, 0) + 1
                log.info(
                    "ingest.ok",
                    external_id=external_id,
                    status=res.status,
                    duration_ms=res.duration_ms,
                )
            except Exception:  # noqa: BLE001
                counts["failed"] += 1
    return counts

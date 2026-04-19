"""POST /api/webhook/recipe — webhook WhiteBeard.

Vérification HMAC SHA-256 + déclenchement async d'une ré-ingestion exhaustive
(via le service `ingest_article_id`). Idempotent.
"""

from __future__ import annotations

import hmac
import json
from hashlib import sha256

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.base import get_session
from ..ingestion.service import ingest_article_id
from ..settings import get_settings

router = APIRouter(prefix="/api/webhook", tags=["webhook"])


def _require_production_secrets() -> None:
    get_settings().require_production_secrets()


class WebhookPayload(BaseModel):
    event: str
    article_id: int


def _verify(body: bytes, signature: str | None) -> None:
    secret = get_settings().webhook_secret
    if not secret:
        raise HTTPException(500, "WEBHOOK_SECRET non configuré")
    if not signature:
        raise HTTPException(401, "Signature absente (X-Signature-256)")
    expected = "sha256=" + hmac.new(
        secret.encode(), body, sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(401, "Signature invalide")


@router.post("/recipe", dependencies=[Depends(_require_production_secrets)])
async def webhook_recipe(
    request: Request,
    x_signature_256: str | None = Header(default=None, alias="X-Signature-256"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    body = await request.body()
    _verify(body, x_signature_256)
    try:
        data = json.loads(body)
        payload = WebhookPayload.model_validate(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Payload invalide: {e}") from e

    if payload.event in ("article.published", "article.updated"):
        result = await ingest_article_id(session, payload.article_id)
        return {"status": "reindexed", "article": result.article_id}
    if payload.event == "article.deleted":
        # Suppression : on délègue à un endpoint dédié dans une itération suivante
        return {"status": "noop_deletion_pending"}
    return {"status": "ignored", "event": payload.event}

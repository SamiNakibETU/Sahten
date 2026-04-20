"""Endpoints utilitaires consommés par le widget v3 :
- GET  /api/models       — liste statique des modèles disponibles
- POST /api/events       — tracking impressions/clics (no-op log)
- POST /api/feedback     — feedback 👍/👎 (no-op log)
- GET  /api/health/deep  — santé approfondie (DB, Redis, secrets)
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .. import __version__, sessions as sessions_store
from ..db.base import get_session
from ..settings import get_settings

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api", tags=["misc"])


@router.get("/models")
def models() -> dict[str, Any]:
    s = get_settings()
    return {
        "default": s.llm_model,
        "models": [
            {"id": s.llm_model, "label": s.llm_model, "provider": "openai"},
        ],
    }


class EventPayload(BaseModel):
    session_id: str | None = None
    request_id: str | None = None
    event: str | None = None
    payload: dict[str, Any] | None = None


@router.post("/events")
async def events(p: EventPayload) -> dict[str, str]:
    log.info(
        "ui.event",
        session_id=p.session_id,
        request_id=p.request_id,
        event=p.event,
        payload=p.payload,
    )
    return {"status": "ok"}


class FeedbackPayload(BaseModel):
    session_id: str | None = None
    request_id: str | None = None
    rating: str | None = None
    comment: str | None = None


@router.post("/feedback")
async def feedback(p: FeedbackPayload) -> dict[str, str]:
    log.info(
        "ui.feedback",
        session_id=p.session_id,
        request_id=p.request_id,
        rating=p.rating,
        comment=(p.comment or "")[:280],
    )
    return {"status": "ok"}


@router.get("/health/deep")
async def health_deep(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    s = get_settings()
    out: dict[str, Any] = {"version": __version__, "env": s.app_env}

    try:
        await session.execute(text("SELECT 1"))
        out["db"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        out["db"] = {"ok": False, "error": str(exc)}

    out["sessions"] = await sessions_store.healthcheck()

    out["secrets"] = {
        "openai": bool(s.openai_api_key),
        "olj": bool(s.olj_api_key),
        "webhook": bool(s.webhook_secret),
        "cohere": bool(s.cohere_api_key),
    }

    out["status"] = "ok" if out["db"].get("ok") else "degraded"
    return out

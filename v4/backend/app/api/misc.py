"""Endpoints utilitaires consommés par le widget v3 :
- GET  /api/models       — liste statique des modèles disponibles
- POST /api/events       — tracking impressions/clics (no-op log)
- POST /api/feedback     — feedback 👍/👎 (no-op log)
- GET  /api/traces       — compat dashboard (stub v4, pas d’agrégation Redis type MVP)
- GET  /api/analytics    — idem
- GET  /api/feedback/stats — idem
- GET  /api/health/deep  — santé approfondie (DB, Redis, secrets)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query
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


@router.get("/traces")
async def traces(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    """Compat `/dashboard` : le MVP agrège les traces dans Redis (Upstash REST)."""
    return {"traces": [], "count": 0, "dashboard_mode": "stub"}


@router.get("/feedback/stats")
async def feedback_stats() -> dict[str, Any]:
    return {
        "positive": 0,
        "negative": 0,
        "positive_rate": 0,
        "recent_negative_reasons": [],
        "dashboard_mode": "stub",
    }


@router.get("/analytics")
async def analytics() -> dict[str, Any]:
    """Même forme que l’ancien `/api/analytics` pour le dashboard statique ; compteurs à 0."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "status": "ok",
        "dashboard_mode": "stub",
        "timestamp": now,
        "conversations": {"total": 0, "recent_traces": 0},
        "events": {"impressions": 0, "clicks": 0, "feedback": 0},
        "rates": {"ctr": 0, "satisfaction": 0, "fallback_rate": 0},
        "fallback": {"total_requests": 0, "base2_count": 0},
        "feedback": {"positive": 0, "negative": 0, "total": 0},
        "models": {},
        "intents": {},
        "top_recipes": [],
        "quality": {
            "exact_match_count": 0,
            "proven_alternative_count": 0,
            "recipe_not_found_count": 0,
            "safety_block_count": 0,
        },
        "response_types": {},
        "routing": {},
    }


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

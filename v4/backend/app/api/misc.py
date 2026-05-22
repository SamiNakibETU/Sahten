"""Endpoints utilitaires : modèles, events, feedback, analytics, santé."""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .. import __version__, analytics_store
from .. import sessions as sessions_store
from ..auth_deps import require_admin_token
from ..db.base import get_session
from ..db.models import Chunk
from ..limiter_support import limiter
from ..llm.models_config import list_llm_models
from ..settings import get_settings

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api", tags=["misc"])


@router.get("/models")
def models() -> dict[str, Any]:
    s = get_settings()
    return {
        "default": s.llm_model,
        "models": list_llm_models(),
    }


@router.post("/events")
@limiter.limit("120/minute")
async def events_post(request: Request) -> dict[str, str]:
    max_b = get_settings().events_max_body_bytes
    try:
        raw = await request.body()
        if len(raw) > max_b:
            raise HTTPException(status_code=413, detail="Corps de requête trop volumineux.")
        data = json.loads(raw.decode("utf-8")) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = {}

    et = (data.get("event_type") or data.get("event") or "").strip().lower()
    session_id = data.get("session_id")
    request_id = data.get("request_id")
    recipe_url = data.get("recipe_url")
    recipe_title = data.get("recipe_title")
    intent = data.get("intent")
    model_used = data.get("model_used")

    if et in ("impression", "click"):
        await analytics_store.record_widget_event(
            event_type=et,
            session_id=session_id,
            request_id=request_id,
            recipe_url=recipe_url,
            recipe_title=recipe_title,
            intent=intent,
            model_used=model_used,
        )
    elif et == "feedback":
        rating = (data.get("rating") or "").lower() or "negative"
        reason = data.get("reason") or data.get("comment")
        await analytics_store.record_feedback_rating(
            request_id=request_id,
            session_id=session_id,
            rating=rating,
            reason=str(reason)[:500] if reason else None,
        )
    else:
        log.info(
            "ui.event.ignored",
            event=et,
            session_id=session_id,
            request_id=request_id,
            extra_keys=list(data.keys()),
        )
        return {"status": "ignored"}

    log.info("ui.event", event_type=et, session_id=session_id, request_id=request_id)
    return {"status": "ok"}


class FeedbackPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str | None = None
    request_id: str | None = None
    rating: str | None = None
    reason: str | None = None
    comment: str | None = None


@router.post("/feedback")
@limiter.limit("60/minute")
async def feedback(request: Request, p: FeedbackPayload) -> dict[str, str]:
    note = (p.reason or p.comment or "").strip()
    await analytics_store.record_feedback_rating(
        request_id=p.request_id,
        session_id=p.session_id,
        rating=(p.rating or "").lower() or "negative",
        reason=note[:500] if note else None,
    )
    log.info(
        "ui.feedback",
        session_id=p.session_id,
        request_id=p.request_id,
        rating=p.rating,
    )
    return {"status": "ok"}


@router.get("/traces", dependencies=[Depends(require_admin_token)])
async def traces(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    return await analytics_store.get_traces(limit)


@router.get("/feedback/stats", dependencies=[Depends(require_admin_token)])
async def feedback_stats() -> dict[str, Any]:
    return await analytics_store.get_feedback_stats()


@router.get("/analytics", dependencies=[Depends(require_admin_token)])
async def analytics() -> dict[str, Any]:
    return await analytics_store.get_analytics()


@router.get("/health/deep", dependencies=[Depends(require_admin_token)])
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

    try:
        n_chunks = int(
            (await session.execute(select(func.count()).select_from(Chunk))).scalar_one()
            or 0
        )
        n_embedded = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(Chunk)
                    .where(Chunk.embedding.is_not(None))
                )
            ).scalar_one()
            or 0
        )
        out["corpus"] = {
            "chunks": n_chunks,
            "chunks_embedded": n_embedded,
            "searchable": n_embedded > 0,
        }
        if n_embedded == 0:
            out["status"] = "degraded"
    except Exception as exc:  # noqa: BLE001
        out["corpus"] = {"error": str(exc)}

    if out.get("status") != "degraded":
        out["status"] = "ok" if out["db"].get("ok") else "degraded"
    return out

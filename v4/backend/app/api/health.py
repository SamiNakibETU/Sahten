"""Healthchecks Railway / Docker."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .. import __version__
from ..db.base import get_session
from ..settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/readyz")
async def readyz(session: AsyncSession = Depends(get_session)) -> dict:
    try:
        await session.execute(text("SELECT 1"))
        db = "ok"
    except Exception as e:  # noqa: BLE001
        if get_settings().app_env == "production":
            db = "error"
        else:
            db = f"error: {e}"
    return {"status": "ok" if db == "ok" else "degraded", "db": db, "version": __version__}

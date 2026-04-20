"""GET /api/sessions, GET /api/sessions/{sid} — historique de chat."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .. import sessions as sessions_store

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def list_sessions(limit: int = Query(50, ge=1, le=500)) -> dict:
    items = await sessions_store.list_sessions(limit=limit)
    return {"items": items, "count": len(items)}


@router.get("/{session_id}")
async def get_session(session_id: str) -> dict:
    if not session_id or len(session_id) > 64:
        raise HTTPException(status_code=400, detail="session_id invalide")
    messages = await sessions_store.get_session_messages(session_id)
    return {
        "session_id": session_id,
        "messages": messages,
        "count": len(messages),
    }

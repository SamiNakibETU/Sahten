"""POST /api/chat — pipeline RAG SOTA (compat widget v3 + clients structurés v4).

Le frontend `frontend/js/sahten.js` attend un objet `{html, request_id,
session_id, model_used}`. Les intégrations OLJ / WhiteBeard côté serveur
préfèrent les champs structurés (`answer_sentences`, `recipe_card`,
`chef_card`, `sources`). On expose les deux dans la même réponse.
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .. import sessions
from ..db.base import get_session
from ..rag.html_renderer import render_answer_html
from ..rag.pipeline import RagPipeline
from ..settings import get_settings

router = APIRouter(prefix="/api", tags=["chat"])


def _require_production_secrets() -> None:
    """En prod uniquement : secrets obligatoires avant le pipeline (pas au boot /healthz)."""
    get_settings().require_production_secrets()


_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


class ChatRequest(BaseModel):
    """Accepte les deux contrats : v3 (`message`) et v4 (`query`)."""

    query: str | None = Field(default=None, max_length=2000)
    message: str | None = Field(default=None, max_length=2000)
    session_id: str | None = None
    debug: bool = False
    model: str | None = None

    def get_text(self) -> str:
        text = (self.query or self.message or "").strip()
        return text


class ChatHit(BaseModel):
    chunk_id: int
    article_external_id: int
    article_title: str
    article_url: str
    section_kind: str
    rerank_score: float


class ChatResponse(BaseModel):
    html: str
    request_id: str
    session_id: str
    model_used: str
    answer_sentences: list[dict]
    recipe_card: dict | None
    chef_card: dict | None
    follow_up: str
    confidence: float
    sources: list[ChatHit]
    timings_ms: dict[str, int]
    intent: str


def _new_session_id() -> str:
    return "ses_" + uuid.uuid4().hex[:12]


def _new_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:16]


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(_require_production_secrets)],
)
async def chat(
    payload: ChatRequest,
    session: AsyncSession = Depends(get_session),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> ChatResponse:
    text = payload.get_text()
    if not text:
        raise HTTPException(status_code=400, detail="Champ `query` ou `message` requis.")

    sid = payload.session_id or _new_session_id()
    request_id = _new_request_id()
    started = time.perf_counter()

    try:
        result = await pipeline.answer(session, text)
    except Exception as exc:  # noqa: BLE001
        log.exception("chat.rag_failed", query_preview=text[:200])
        # On persiste quand même l'échec dans l'historique pour debug UI.
        fallback_html = (
            '<div class="sahten-narrative"><p><em>Mille excuses, un petit incident '
            "en cuisine est survenu. Réessayez dans un instant ou reformulez.</em>"
            "</p></div>"
        )
        await sessions.record_turn(
            sid,
            user_message=text,
            assistant_html=fallback_html,
            request_id=request_id,
            intent="error",
            confidence=0.0,
            sources=[],
            timings_ms={"total_ms": int((time.perf_counter() - started) * 1000)},
            model_used=payload.model or get_settings().llm_model,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [
        ChatHit(
            chunk_id=r.hit.chunk_id,
            article_external_id=r.hit.article_external_id,
            article_title=r.hit.article_title,
            article_url=r.hit.article_url,
            section_kind=r.hit.section_kind,
            rerank_score=r.rerank_score,
        )
        for r in result.reranked
    ]
    html_str = render_answer_html(result.answer, result.reranked)
    model_used = payload.model or get_settings().llm_model

    try:
        await sessions.record_turn(
            sid,
            user_message=text,
            assistant_html=html_str,
            request_id=request_id,
            intent=result.plan.intent,
            confidence=result.answer.confidence,
            sources=[s.model_dump() for s in sources],
            timings_ms=result.timings_ms,
            model_used=model_used,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("chat.record_turn_failed", error=str(exc))

    return ChatResponse(
        html=html_str,
        request_id=request_id,
        session_id=sid,
        model_used=model_used,
        answer_sentences=[s.model_dump() for s in result.answer.answer_sentences],
        recipe_card=result.answer.recipe_card.model_dump()
        if result.answer.recipe_card else None,
        chef_card=result.answer.chef_card.model_dump()
        if result.answer.chef_card else None,
        follow_up=result.answer.follow_up,
        confidence=result.answer.confidence,
        sources=sources,
        timings_ms=result.timings_ms,
        intent=result.plan.intent,
    )

"""POST /api/chat — pipeline RAG SOTA (compat widget v3 + clients structurés v4).

Le frontend `frontend/js/sahten.js` attend un objet `{html, request_id,
session_id, model_used}`. Les intégrations OLJ / WhiteBeard côté serveur
préfèrent les champs structurés (`answer_sentences`, `recipe_card`,
`chef_card`, `sources`). On expose les deux dans la même réponse.
"""

from __future__ import annotations

import json
import time
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from .. import analytics_store, sessions
from ..db.base import get_session
from ..limiter_support import limiter
from ..rag.html_renderer import render_answer_html
from ..rag.pipeline import RagPipeline
from ..llm.models_config import resolve_llm_model
from ..settings import get_settings

# Aligné sur les usages réels (recettes collées, contexte long) — au-delà, 422 côté validation.
CHAT_INPUT_MAX_LEN = 12_000

router = APIRouter(prefix="/api", tags=["chat"])
log = structlog.get_logger(__name__)


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

    model_config = ConfigDict(extra="ignore")

    query: str | None = Field(default=None, max_length=CHAT_INPUT_MAX_LEN)
    message: str | None = Field(default=None, max_length=CHAT_INPUT_MAX_LEN)
    # Format limité : préfixe ses_ ou req_, suivi de caractères alphanum/tiret.
    # max_length=80 empêche les tentatives d'injection via session_id long.
    session_id: str | None = Field(
        default=None,
        max_length=80,
        pattern=r"^[a-zA-Z0-9_\-]{1,80}$",
    )
    debug: bool = False
    model: str | None = None

    @field_validator("query", "message", mode="before")
    @classmethod
    def _coerce_text_fields(cls, v: object) -> str | None:
        """Évite les 422 si un client envoie un nombre ou un bool à la place du texte."""
        if v is None:
            return None
        if isinstance(v, bool):
            return "oui" if v else "non"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            return v
        return str(v)

    @field_validator("session_id", "model", mode="before")
    @classmethod
    def _coerce_optional_str(cls, v: object) -> str | None:
        if v is None:
            return None
        return str(v).strip() or None

    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "oui", "on")
        return bool(v)

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
    recipe_card_secondary: dict | None = None
    chef_card: dict | None
    follow_up: str
    confidence: float
    sources: list[ChatHit]
    timings_ms: dict[str, int]
    intent: str
    cost_usd: float | None = None
    cost_breakdown: dict | None = None
    # Observabilité : stratégie de routage + plan (debuggable côté client/QA).
    answer_strategy: str | None = None
    debug_plan: dict | None = None


def _new_session_id() -> str:
    return "ses_" + uuid.uuid4().hex[:12]


def _new_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:16]


async def _payload_from_request(request: Request) -> ChatRequest:
    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="JSON invalide.") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="Corps JSON invalide.")
    try:
        return ChatRequest.model_validate(raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _fallback_chat_response(
    *,
    payload: ChatRequest,
    session_id: str,
    request_id: str,
    detail: str,
    total_ms: int = 0,
) -> ChatResponse:
    model_used = resolve_llm_model(payload.model)
    # Ne pas exposer le détail de l'erreur interne à l'utilisateur final.
    _ = detail  # conservé pour le log interne seulement
    html = (
        '<div class="sahten-narrative"><p><em>Mille excuses, un petit incident '
        "en cuisine est survenu.</em></p>"
        "<p>Le service est temporairement indisponible. Réessayez dans un instant.</p>"
        "</div>"
    )
    follow_up = "Souhaitez-vous réessayer dans quelques instants ?"
    return ChatResponse(
        html=html,
        request_id=request_id,
        session_id=session_id,
        model_used=model_used,
        answer_sentences=[],
        recipe_card=None,
        recipe_card_secondary=None,
        chef_card=None,
        follow_up=follow_up,
        confidence=0.0,
        sources=[],
        timings_ms={"total_ms": max(0, int(total_ms))},
        intent="error",
    )


async def _run_chat_pipeline(
    *,
    session: AsyncSession,
    pipeline: RagPipeline,
    payload: ChatRequest,
) -> ChatResponse:
    """Exécute le RAG + persistance ; utilisé par POST /chat et POST /chat/stream."""
    text = payload.get_text()
    if not text:
        raise HTTPException(status_code=400, detail="Champ `query` ou `message` requis.")

    sid = payload.session_id or _new_session_id()
    request_id = _new_request_id()
    model_used = resolve_llm_model(payload.model)
    started = time.perf_counter()

    history_block = ""
    try:
        history_block = await sessions.conversation_block_for_llm(sid)
    except Exception as exc:
        log.warning("chat.history_block_failed", error=str(exc))

    try:
        result = await pipeline.answer(
            session,
            text,
            session_id=sid,
            conversation_history=history_block or None,
            llm_model=model_used,
        )
    except Exception as exc:
        log.exception("chat.rag_failed", query_preview=text[:200])
        # On persiste quand même l'échec dans l'historique pour debug UI.
        fallback_html = (
            '<div class="sahten-narrative"><p><em>Mille excuses, un petit incident '
            "en cuisine est survenu. Réessayez dans un instant ou reformulez.</em>"
            "</p></div>"
        )
        total_ms = int((time.perf_counter() - started) * 1000)
        try:
            await sessions.record_turn(
                sid,
                user_message=text,
                assistant_html=fallback_html,
                request_id=request_id,
                intent="error",
                confidence=0.0,
                sources=[],
                timings_ms={"total_ms": total_ms},
                model_used=model_used,
            )
        except Exception as rec_exc:
            log.warning("chat.record_turn_failed_on_error", error=str(rec_exc))
        return _fallback_chat_response(
            payload=payload,
            session_id=sid,
            request_id=request_id,
            detail=str(exc),
            total_ms=total_ms,
        )

    sources = [
        ChatHit(
            chunk_id=r.hit.chunk_id,
            article_external_id=r.hit.article_external_id,
            article_title=r.hit.article_title or "",
            article_url=r.hit.article_url or "",
            section_kind=r.hit.section_kind or "",
            rerank_score=float(r.rerank_score),
        )
        for r in result.reranked
    ]
    html_str = render_answer_html(result.answer, result.reranked)

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
    except Exception as exc:
        log.warning("chat.record_turn_failed", error=str(exc))

    try:
        await analytics_store.record_chat_trace(
            request_id=request_id,
            session_id=sid,
            user_message=text,
            response_html=html_str,
            intent=result.plan.intent,
            confidence=result.answer.confidence,
            recipe_count=len(sources),
            model_used=model_used,
            timings_ms=result.timings_ms,
            is_base2_fallback=bool(result.is_base2_fallback),
            cost_breakdown=result.cost_breakdown,
            answer_strategy=result.answer_strategy,
        )
    except Exception as exc:
        log.warning("chat.analytics_trace_failed", error=str(exc))

    return ChatResponse(
        html=html_str,
        request_id=request_id,
        session_id=sid,
        model_used=model_used,
        answer_sentences=[s.model_dump() for s in result.answer.answer_sentences],
        recipe_card=result.answer.recipe_card.model_dump()
        if result.answer.recipe_card else None,
        recipe_card_secondary=result.answer.recipe_card_secondary.model_dump()
        if result.answer.recipe_card_secondary else None,
        chef_card=result.answer.chef_card.model_dump()
        if result.answer.chef_card else None,
        follow_up=result.answer.follow_up,
        confidence=result.answer.confidence,
        sources=sources,
        timings_ms=result.timings_ms,
        intent=result.plan.intent,
        cost_usd=(
            float(result.cost_breakdown.get("estimated_usd", 0.0))
            if result.cost_breakdown
            else None
        ),
        cost_breakdown=result.cost_breakdown,
        answer_strategy=result.answer_strategy,
        debug_plan={
            "rewritten_query": result.plan.rewritten_query,
            "intent": result.plan.intent,
            "ingredient_slugs": result.plan.ingredient_slugs,
            "chef_slugs": result.plan.chef_slugs,
            "n_reranked": len(result.reranked),
        },
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(_require_production_secrets)],
)
@limiter.limit("45/minute")
async def chat(
    request: Request,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> ChatResponse:
    payload = await _payload_from_request(request)
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        log.exception("chat.pipeline_init_failed", error=str(exc))
        sid = payload.session_id or _new_session_id()
        rid = _new_request_id()
        return _fallback_chat_response(
            payload=payload,
            session_id=sid,
            request_id=rid,
            detail=str(exc),
            total_ms=0,
        )
    return await _run_chat_pipeline(session=session, pipeline=pipeline, payload=payload)


@router.post(
    "/chat/stream",
    dependencies=[Depends(_require_production_secrets)],
)
@limiter.limit("45/minute")
async def chat_stream(
    request: Request,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> StreamingResponse:
    """SSE minimal : un unique événement `done` (même charge utile que POST /chat).

    Le widget v4 appelle d'abord cette route ; sans elle, il retombait sur POST /chat
    et pouvait cumuler erreurs réseau / validation.
    """

    async def events():
        payload = await _payload_from_request(request)
        try:
            pipeline = get_pipeline()
            out = await _run_chat_pipeline(session=session, pipeline=pipeline, payload=payload)
        except Exception as exc:
            log.exception("chat.stream_failed", error=str(exc))
            out = _fallback_chat_response(
                payload=payload,
                session_id=payload.session_id or _new_session_id(),
                request_id=_new_request_id(),
                detail=str(exc),
                total_ms=0,
            )
        body = out.model_dump(mode="json")
        body["type"] = "done"
        yield "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")

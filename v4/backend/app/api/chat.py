"""POST /api/chat — pipeline RAG SOTA."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.base import get_session
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
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class ChatHit(BaseModel):
    chunk_id: int
    article_external_id: int
    article_title: str
    article_url: str
    section_kind: str
    rerank_score: float


class ChatResponse(BaseModel):
    answer_sentences: list[dict]
    recipe_card: dict | None
    chef_card: dict | None
    follow_up: str
    confidence: float
    sources: list[ChatHit]
    timings_ms: dict[str, int]
    intent: str


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
    try:
        result = await pipeline.answer(session, payload.query)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    return ChatResponse(
        answer_sentences=[s.model_dump() for s in result.answer.answer_sentences],
        recipe_card=result.answer.recipe_card.model_dump()
            if result.answer.recipe_card else None,
        chef_card=result.answer.chef_card.model_dump()
            if result.answer.chef_card else None,
        follow_up=result.answer.follow_up,
        confidence=result.answer.confidence,
        sources=[
            ChatHit(
                chunk_id=r.hit.chunk_id,
                article_external_id=r.hit.article_external_id,
                article_title=r.hit.article_title,
                article_url=r.hit.article_url,
                section_kind=r.hit.section_kind,
                rerank_score=r.rerank_score,
            )
            for r in result.reranked
        ],
        timings_ms=result.timings_ms,
        intent=result.plan.intent,
    )

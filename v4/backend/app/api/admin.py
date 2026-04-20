"""GET /api/admin/* — navigation read-only de la base RAG.

But : permettre aux relecteurs OLJ et aux ops de voir d'un coup d'œil
- combien d'articles sont ingérés,
- quels chefs / catégories / mots-clés sont indexés,
- le détail d'un article (sections + chunks + couverture embedding).

Aucune mutation ici. Pas d'auth pour l'instant : à protéger via reverse-proxy
ou ajout d'un middleware basé sur `WEBHOOK_SECRET` côté Railway si on
expose publiquement.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.base import get_session
from ..db.models import (
    Article,
    Category,
    Chunk,
    Keyword,
    Person,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats")
async def stats(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    """Compteurs globaux pour la page admin."""
    counts: dict[str, int] = {}
    for label, table in [
        ("articles", Article),
        ("chunks", Chunk),
        ("persons", Person),
        ("categories", Category),
        ("keywords", Keyword),
    ]:
        result = await session.execute(select(func.count()).select_from(table))
        counts[label] = int(result.scalar_one() or 0)

    embedded = await session.execute(
        select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
    )
    counts["chunks_embedded"] = int(embedded.scalar_one() or 0)
    return {"counts": counts}


@router.get("/articles")
async def list_articles(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: str | None = Query(None, max_length=200),
) -> dict[str, Any]:
    stmt = select(
        Article.id,
        Article.external_id,
        Article.slug,
        Article.title,
        Article.url,
        Article.first_published_at,
        Article.ingestion_status,
    ).order_by(Article.first_published_at.desc().nullslast(), Article.id.desc())
    if search:
        like = f"%{search.lower()}%"
        stmt = stmt.where(func.lower(Article.title).like(like))
    stmt = stmt.limit(limit).offset(offset)
    rows = (await session.execute(stmt)).all()

    total_stmt = select(func.count()).select_from(Article)
    if search:
        total_stmt = total_stmt.where(func.lower(Article.title).like(f"%{search.lower()}%"))
    total = int((await session.execute(total_stmt)).scalar_one() or 0)

    return {
        "items": [
            {
                "id": r.id,
                "external_id": r.external_id,
                "slug": r.slug,
                "title": r.title,
                "url": r.url,
                "first_published_at": r.first_published_at.isoformat()
                if r.first_published_at else None,
                "ingestion_status": r.ingestion_status,
            }
            for r in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/articles/{article_id}")
async def article_detail(
    article_id: int, session: AsyncSession = Depends(get_session)
) -> dict[str, Any]:
    article = (
        await session.execute(select(Article).where(Article.id == article_id))
    ).scalar_one_or_none()
    if article is None:
        raise HTTPException(status_code=404, detail="article introuvable")

    chunks = (
        await session.execute(
            select(
                Chunk.id, Chunk.position, Chunk.kind, Chunk.text,
                Chunk.token_count, Chunk.embedding_model,
            )
            .where(Chunk.article_id == article_id)
            .order_by(Chunk.position)
        )
    ).all()

    return {
        "id": article.id,
        "external_id": article.external_id,
        "title": article.title,
        "subtitle": article.subtitle,
        "url": article.url,
        "summary": article.summary,
        "is_premium": article.is_premium,
        "first_published_at": article.first_published_at.isoformat()
        if article.first_published_at else None,
        "ingestion_status": article.ingestion_status,
        "ingestion_notes": article.ingestion_notes,
        "n_chunks": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "position": c.position,
                "kind": c.kind,
                "token_count": c.token_count,
                "embedding_model": c.embedding_model,
                "text_preview": (c.text or "")[:300],
            }
            for c in chunks
        ],
    }


@router.get("/persons")
async def list_persons(
    session: AsyncSession = Depends(get_session),
    role: str | None = Query(None, max_length=64),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    stmt = select(
        Person.id, Person.external_id, Person.name, Person.slug,
        Person.role, Person.department,
    ).order_by(Person.name)
    if role:
        stmt = stmt.where(Person.role == role)
    stmt = stmt.limit(limit)
    rows = (await session.execute(stmt)).all()
    return {
        "items": [
            {
                "id": r.id,
                "external_id": r.external_id,
                "name": r.name,
                "slug": r.slug,
                "role": r.role,
                "department": r.department,
            }
            for r in rows
        ],
        "count": len(rows),
    }

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
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.base import get_session
from ..llm.query_understanding import QueryAnalyzer, QueryPlan
from ..rag.embeddings import OpenAIEmbeddings
from ..rag.pipeline import _retrieval_fallback_queries
from ..rag.retriever import HybridRetriever
from ..settings import get_settings
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


@router.get("/diagnose-retrieval")
async def diagnose_retrieval(
    q: str = Query(
        ...,
        min_length=1,
        max_length=4000,
        description="Requête à tester (ex. recette avec du concombre)",
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Diagnostic RAG : compteurs DB, plan LLM, nombre de hits par étape (comme le pipeline).

    Appeler en prod : ``GET /api/admin/diagnose-retrieval?q=recette%20avec%20du%20concombre``
    """
    s = get_settings()
    rerank_top_n = s.rag_rerank_top_k
    final_limit = max(30, rerank_top_n * 4)

    counts: dict[str, int] = {}
    for label, table in [
        ("articles", Article),
        ("chunks", Chunk),
    ]:
        r = await session.execute(select(func.count()).select_from(table))
        counts[label] = int(r.scalar_one() or 0)
    embedded = await session.execute(
        select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
    )
    counts["chunks_embedded"] = int(embedded.scalar_one() or 0)

    hints: list[str] = []
    if counts["chunks"] == 0:
        hints.append("chunks=0 : corpus vide — lancer l’ingestion des articles.")
    elif counts["chunks_embedded"] == 0:
        hints.append("chunks_embedded=0 : aucun vecteur — lancer le job d’embeddings / indexer.")
    elif counts["chunks_embedded"] < counts["chunks"]:
        hints.append(
            f'{counts["chunks"] - counts["chunks_embedded"]} chunks sans embedding : '
            "retrieval vectoriel partiellement inopérant."
        )

    # Sanity SQL : lexique seul sur la requête (sans filtre article)
    chunk_sanity: dict[str, Any] = {}
    try:
        lex_row = (
            await session.execute(
                text(
                    """
                    SELECT
                        COUNT(*)::int AS n_chunks,
                        COUNT(*) FILTER (
                            WHERE c.embedding IS NOT NULL
                        )::int AS n_with_embedding,
                        COUNT(*) FILTER (
                            WHERE c.search_tsv @@ websearch_to_tsquery('french', :q)
                        )::int AS n_lex_match
                    FROM chunks c
                    """
                ),
                {"q": q.strip()},
            )
        ).mappings().first()
        chunk_sanity = dict(lex_row) if lex_row else {}
    except Exception as exc:  # noqa: BLE001
        chunk_sanity = {"error": str(exc)}

    plan_dump: dict[str, Any] | None = None
    plan_error: str | None = None
    try:
        plan_obj = await QueryAnalyzer().analyze(q)
        plan_dump = plan_obj.model_dump()
    except Exception as exc:  # noqa: BLE001
        plan_error = str(exc)
        plan_obj = QueryPlan(
            rewritten_query=q.strip(),
            intent="mixed",
            chef_slugs=[],
            ingredient_slugs=[],
            category_slugs=[],
            keyword_slugs=[],
            focus_section_kinds=[],
            needs_context_after=False,
        )
        plan_dump = plan_obj.model_dump()

    q_main = (plan_obj.rewritten_query or q).strip()
    retriever = HybridRetriever(OpenAIEmbeddings())

    searches: list[dict[str, Any]] = []

    async def _add(step: str, query_str: str, **kwargs: Any) -> None:
        try:
            hits = await retriever.search(
                session, query_str, final_limit=final_limit, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            searches.append(
                {
                    "step": step,
                    "query": query_str,
                    "error": str(exc),
                    "n_hits": 0,
                }
            )
            return
        titles = list({h.article_title for h in hits[:8]})
        searches.append(
            {
                "step": step,
                "query": query_str,
                "n_hits": len(hits),
                "sample_titles": titles,
            }
        )

    await _add(
        "1_with_struct_filters",
        q_main,
        chef_slugs=plan_obj.chef_slugs,
        ingredient_slugs=plan_obj.ingredient_slugs,
        category_slugs=plan_obj.category_slugs,
        keyword_slugs=plan_obj.keyword_slugs,
    )
    await _add("2_broad_no_filters", q_main)

    seen_q = {q_main.strip()}
    for alt in _retrieval_fallback_queries(q, plan_obj):
        if alt.strip() in seen_q:
            continue
        seen_q.add(alt.strip())
        await _add(f"3_fallback:{alt[:40]}", alt)

    if chunk_sanity.get("n_lex_match") == 0 and chunk_sanity.get("n_chunks", 0) > 0:
        hints.append(
            "n_lex_match=0 pour cette requête : aucun chunk ne matche le plein texte "
            "(websearch_to_tsquery). Vérifier search_tsv / langue du contenu."
        )

    return {
        "query": q,
        "rewritten_or_main": q_main,
        "embedding_model": s.embedding_model,
        "embedding_dim": s.embedding_dim,
        "counts": counts,
        "chunk_sanity": chunk_sanity,
        "query_plan": plan_dump,
        "query_plan_error": plan_error,
        "retrieval_steps": searches,
        "hints": hints,
    }


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

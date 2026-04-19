"""Indexation : article -> chunks -> embeddings -> upsert dans `chunks`."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models
from ..ingestion.html_sectionizer import Section
from .chunker import chunk_article
from .embeddings import EmbeddingProvider


async def reindex_article(
    session: AsyncSession,
    article: models.Article,
    embedder: EmbeddingProvider,
) -> int:
    """Recalcule tous les chunks + embeddings d'un article. Retourne le nombre."""
    # 1. Récupérer les sections fraîches depuis la DB
    res = await session.execute(
        select(models.ArticleSection)
        .where(models.ArticleSection.article_id == article.id)
        .order_by(models.ArticleSection.position)
    )
    db_sections = res.scalars().all()
    sections = [
        Section(
            position=s.position,
            kind=s.kind,
            heading=s.heading,
            html=s.html,
            text=s.text,
            metadata=s.metadata_json or {},
        )
        for s in db_sections
    ]

    chunks = chunk_article(
        article_external_id=article.external_id,
        article_title=article.title,
        sections=sections,
    )
    if not chunks:
        return 0

    embeddings = await embedder.embed([c.text for c in chunks])
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding mismatch: {len(embeddings)} vs {len(chunks)}"
        )

    # Wipe & insert (idempotent + pas besoin de diff)
    await session.execute(
        models.Chunk.__table__.delete().where(
            models.Chunk.article_id == article.id
        )
    )

    rows = []
    section_pos_to_id = {s.position: s.id for s in db_sections}
    for ch, emb in zip(chunks, embeddings, strict=True):
        rows.append({
            "article_id": article.id,
            "section_id": section_pos_to_id.get(ch.section_position or -1),
            "position": ch.position,
            "kind": ch.kind,
            "text": ch.text,
            "token_count": ch.token_count,
            "metadata_json": ch.metadata,
            "embedding": emb,
            "embedding_model": embedder.model,
        })
    await session.execute(models.Chunk.__table__.insert(), rows)
    return len(rows)

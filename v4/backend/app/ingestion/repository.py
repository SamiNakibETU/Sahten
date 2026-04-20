"""Upsert idempotent d'un `MappedArticle` vers la base.

Idempotence basée sur `external_id`. Effets :
  - upsert article + raw_payload + ingestion_status
  - upsert authors et liens article_authors
  - upsert keywords / categories et liens
  - remplacement complet des sections (delete-then-insert : c'est plus
    simple et plus prévisible que diff sémantique pour des sections HTML)
  - pas de chunking ici (responsabilité du module RAG)
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models
from .mapper import (
    MappedArticle,
    MappedAuthor,
    MappedCategory,
    MappedKeyword,
)


async def upsert_article(session: AsyncSession, mapped: MappedArticle) -> models.Article:
    article = await _upsert_article_row(session, mapped)
    await _replace_sections(session, article.id, mapped)
    await _link_authors(session, article.id, mapped.authors)
    await _link_keywords(session, article.id, mapped.keywords)
    await _link_categories(session, article.id, mapped.categories)
    return article


async def _upsert_article_row(
    session: AsyncSession, m: MappedArticle
) -> models.Article:
    stmt = pg_insert(models.Article).values(
        external_id=m.external_id,
        url=m.url,
        slug=m.slug,
        title=m.title,
        subtitle=m.subtitle,
        summary=m.summary,
        introduction=m.introduction,
        signature=m.signature,
        body_html=m.body_html,
        body_text=m.body_text,
        content_length=m.content_length,
        time_to_read=m.time_to_read,
        is_premium=m.is_premium,
        cover_image_url=m.cover_image_url,
        cover_image_caption=m.cover_image_caption,
        first_published_at=m.first_published_at,
        last_updated_at=m.last_updated_at,
        seo=m.seo,
        raw_payload=m.raw_payload,
        ingestion_source="whitebeard",
        ingestion_status=m.ingestion_status,
        ingestion_notes=m.ingestion_notes,
    )
    # `search_tsv` est une colonne GENERATED (Computed côté ORM) : Postgres
    # refuse `SET search_tsv = excluded.search_tsv` sur ON CONFLICT.
    # Elle se recalcule automatiquement quand title/summary/body_text changent.
    _skip_on_conflict_update = frozenset(
        {"id", "external_id", "created_at", "search_tsv"}
    )
    update_cols = {
        c.name: c
        for c in stmt.excluded
        if c.name not in _skip_on_conflict_update
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=[models.Article.external_id],
        set_=update_cols,
    ).returning(models.Article)
    result = await session.execute(stmt)
    return result.scalar_one()


async def _replace_sections(
    session: AsyncSession, article_id: int, m: MappedArticle
) -> None:
    await session.execute(
        models.ArticleSection.__table__.delete().where(
            models.ArticleSection.article_id == article_id
        )
    )
    if not m.sections:
        return
    rows = [
        {
            "article_id": article_id,
            "position": s.position,
            "kind": s.kind,
            "heading": s.heading,
            "html": s.html,
            "text": s.text,
            "metadata_json": s.metadata or None,
        }
        for s in m.sections
    ]
    await session.execute(models.ArticleSection.__table__.insert(), rows)


async def _link_authors(
    session: AsyncSession, article_id: int, authors: list[MappedAuthor]
) -> None:
    await session.execute(
        models.ArticleAuthor.__table__.delete().where(
            models.ArticleAuthor.article_id == article_id
        )
    )
    for pos, a in enumerate(authors):
        person = await _upsert_person(session, a)
        await session.execute(
            pg_insert(models.ArticleAuthor)
            .values(
                article_id=article_id,
                person_id=person.id,
                role=a.role,
                position=pos,
            )
            .on_conflict_do_update(
                index_elements=["article_id", "person_id"],
                set_={"role": a.role, "position": pos},
            )
        )


async def _upsert_person(
    session: AsyncSession, a: MappedAuthor
) -> models.Person:
    if a.external_id is not None:
        existing = await session.execute(
            select(models.Person).where(models.Person.external_id == a.external_id)
        )
        person = existing.scalar_one_or_none()
        if person:
            person.name = a.name
            person.slug = a.slug or person.slug
            person.role = a.role
            person.department = a.department
            person.biography_html = a.biography_html
            person.biography_text = a.biography_text
            person.description = a.description
            person.image_url = a.image_url
            person.raw_payload = a.raw
            await session.flush()
            return person
    # Insert nouveau
    person = models.Person(
        external_id=a.external_id,
        name=a.name,
        slug=a.slug,
        role=a.role,
        department=a.department,
        biography_html=a.biography_html,
        biography_text=a.biography_text,
        description=a.description,
        image_url=a.image_url,
        raw_payload=a.raw,
    )
    session.add(person)
    await session.flush()
    return person


async def _link_keywords(
    session: AsyncSession, article_id: int, keywords: list[MappedKeyword]
) -> None:
    await session.execute(
        models.ArticleKeyword.__table__.delete().where(
            models.ArticleKeyword.article_id == article_id
        )
    )
    for k in keywords:
        kw = await _upsert_keyword(session, k)
        await session.execute(
            pg_insert(models.ArticleKeyword)
            .values(article_id=article_id, keyword_id=kw.id)
            .on_conflict_do_nothing()
        )


async def _upsert_keyword(
    session: AsyncSession, k: MappedKeyword
) -> models.Keyword:
    res = await session.execute(
        select(models.Keyword).where(models.Keyword.slug == k.slug)
    )
    existing = res.scalar_one_or_none()
    if existing:
        if k.description and not existing.description:
            existing.description = k.description
        if k.external_id and not existing.external_id:
            existing.external_id = k.external_id
        await session.flush()
        return existing
    obj = models.Keyword(
        external_id=k.external_id,
        name=k.name,
        slug=k.slug,
        description=k.description,
    )
    session.add(obj)
    await session.flush()
    return obj


async def _link_categories(
    session: AsyncSession, article_id: int, cats: list[MappedCategory]
) -> None:
    await session.execute(
        models.ArticleCategory.__table__.delete().where(
            models.ArticleCategory.article_id == article_id
        )
    )
    for c in cats:
        obj = await _upsert_category(session, c)
        await session.execute(
            pg_insert(models.ArticleCategory)
            .values(article_id=article_id, category_id=obj.id)
            .on_conflict_do_nothing()
        )


async def _upsert_category(
    session: AsyncSession, c: MappedCategory
) -> models.Category:
    res = await session.execute(
        select(models.Category).where(models.Category.slug == c.slug)
    )
    existing = res.scalar_one_or_none()
    if existing:
        if c.description and not existing.description:
            existing.description = c.description
        if c.external_id and not existing.external_id:
            existing.external_id = c.external_id
        await session.flush()
        return existing
    obj = models.Category(
        external_id=c.external_id,
        name=c.name,
        slug=c.slug,
        description=c.description,
    )
    session.add(obj)
    await session.flush()
    return obj

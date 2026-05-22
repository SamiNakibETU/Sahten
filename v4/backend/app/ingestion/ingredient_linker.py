"""Lie les ingrédients connus aux articles (table ``article_ingredients``)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models
from ..rag.ingredient_match import (
    INGREDIENT_SLUG_ALIASES,
    extract_ingredient_slugs_from_text,
    ingredient_display_name,
    scan_known_ingredient_slugs,
)


def _ingredient_name(slug: str) -> str:
    label = ingredient_display_name(slug)
    return label[:1].upper() + label[1:] if label else slug.replace("-", " ")


async def _upsert_ingredient(session: AsyncSession, slug: str) -> models.Ingredient:
    aliases = list(INGREDIENT_SLUG_ALIASES.get(slug, ()))
    name = _ingredient_name(slug)
    res = await session.execute(
        select(models.Ingredient).where(models.Ingredient.slug == slug)
    )
    existing = res.scalar_one_or_none()
    if existing is not None:
        if aliases and not existing.aliases:
            existing.aliases = aliases
        return existing
    obj = models.Ingredient(name=name, slug=slug, aliases=aliases or None)
    session.add(obj)
    await session.flush()
    return obj


async def link_article_ingredients(
    session: AsyncSession,
    article_id: int,
    *,
    section_kinds: tuple[str, ...] = (
        "ingredients_list",
        "recipe_summary",
        "recipe_steps",
    ),
) -> int:
    """Extrait les ingrédients connus des sections et met à jour ``article_ingredients``."""
    res = await session.execute(
        select(models.ArticleSection)
        .where(models.ArticleSection.article_id == article_id)
        .where(models.ArticleSection.kind.in_(section_kinds))
        .order_by(models.ArticleSection.position)
    )
    sections = res.scalars().all()
    blob = "\n".join((s.text or "") for s in sections)
    slugs = extract_ingredient_slugs_from_text(blob)
    slugs.extend(scan_known_ingredient_slugs(blob))
    seen: set[str] = set()
    ordered: list[str] = []
    for slug in slugs:
        if slug not in seen:
            seen.add(slug)
            ordered.append(slug)

    await session.execute(
        models.ArticleIngredient.__table__.delete().where(
            models.ArticleIngredient.article_id == article_id
        )
    )
    for slug in ordered:
        ing = await _upsert_ingredient(session, slug)
        await session.execute(
            pg_insert(models.ArticleIngredient)
            .values(
                article_id=article_id,
                ingredient_id=ing.id,
                raw_text=_ingredient_name(slug),
                is_main=(slug == ordered[0]),
            )
            .on_conflict_do_nothing()
        )
    return len(ordered)


async def link_all_article_ingredients(session: AsyncSession) -> dict[str, int]:
    """Backfill ``article_ingredients`` pour tous les articles."""
    res = await session.execute(select(models.Article.id))
    article_ids = [int(row[0]) for row in res.all()]
    total_links = 0
    for aid in article_ids:
        total_links += await link_article_ingredients(session, aid)
    return {"articles": len(article_ids), "links": total_links}

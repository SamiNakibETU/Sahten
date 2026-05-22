"""Lie les ingrédients connus aux articles (table ``article_ingredients``)."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models
from ..rag.ingredient_match import (
    INGREDIENT_SLUG_ALIASES,
    canonical_ingredient_slug,
    extract_ingredient_slugs_from_text,
    ingredient_display_name,
    scan_known_ingredient_slugs,
)


def _ingredient_name(slug: str) -> str:
    label = ingredient_display_name(slug)
    return label[:1].upper() + label[1:] if label else slug.replace("-", " ")


def _aliases_for_slug(slug: str) -> list[str]:
    canonical = canonical_ingredient_slug(slug)
    terms: set[str] = set()
    for key, aliases in INGREDIENT_SLUG_ALIASES.items():
        if canonical_ingredient_slug(key) == canonical:
            terms.update(aliases)
            terms.add(key.replace("-", " "))
    return sorted(terms)


async def _find_ingredient_by_name(
    session: AsyncSession, name: str
) -> models.Ingredient | None:
    res = await session.execute(
        select(models.Ingredient).where(
            func.lower(models.Ingredient.name) == name.strip().lower()
        )
    )
    return res.scalar_one_or_none()


async def _upsert_ingredient(session: AsyncSession, slug: str) -> models.Ingredient:
    slug = canonical_ingredient_slug(slug)
    aliases = _aliases_for_slug(slug)
    name = _ingredient_name(slug)

    res = await session.execute(
        select(models.Ingredient).where(models.Ingredient.slug == slug)
    )
    existing = res.scalar_one_or_none()
    if existing is not None:
        if aliases and not existing.aliases:
            existing.aliases = aliases
        return existing

    by_name = await _find_ingredient_by_name(session, name)
    if by_name is not None:
        if by_name.slug != slug and not by_name.slug:
            by_name.slug = slug
        if aliases:
            merged = sorted(set((by_name.aliases or []) + aliases))
            by_name.aliases = merged
        await session.flush()
        return by_name

    stmt = (
        pg_insert(models.Ingredient)
        .values(name=name, slug=slug, aliases=aliases or None)
        .on_conflict_do_update(
            index_elements=[models.Ingredient.slug],
            set_={"aliases": aliases or None},
        )
        .returning(models.Ingredient)
    )
    row = (await session.execute(stmt)).scalar_one()
    return row


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
        canonical = canonical_ingredient_slug(slug)
        if canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)

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

"""Audit offline du referentiel ingredients Sahtein v4.

Le script lit la base Postgres et produit un rapport Markdown ou JSON sur:
  - les ingredients canoniques existants;
  - leurs aliases stockes en base;
  - leurs frequences article et mentions dans les chunks ingredients_list;
  - les ecarts entre aliases DB et aliases encore codes dans ingredient_match.py.

Usage depuis ``sahten_github/v4``:

    python scripts/audit_ingredients.py --out docs/ingredient-audit.md
    python scripts/audit_ingredients.py --format json --out docs/ingredient-audit.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.base import get_sessionmaker
from backend.app.rag.ingredient_match import INGREDIENT_SLUG_ALIASES


@dataclass(frozen=True)
class IngredientAuditRow:
    slug: str
    name: str
    aliases: list[str]
    category: str
    article_count: int
    ingredients_list_mentions: int
    example_articles: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AliasGapReport:
    ingredients_without_db_aliases: list[str]
    code_aliases_missing_in_db: dict[str, list[str]]
    db_aliases_missing_in_code: dict[str, list[str]]
    code_aliases_without_db_ingredient: list[str]


def _normalize_term(value: str) -> str:
    text_value = unicodedata.normalize("NFKD", value or "")
    text_value = "".join(ch for ch in text_value if not unicodedata.combining(ch))
    text_value = text_value.lower().replace("\u2019", "'")
    text_value = re.sub(r"[^a-z0-9'\s-]+", " ", text_value)
    return " ".join(text_value.replace("-", " ").split())


def _alias_set(name: str, aliases: list[str]) -> set[str]:
    terms = {name, *aliases}
    return {_normalize_term(term) for term in terms if _normalize_term(term)}


def build_alias_gap_report(
    rows: list[IngredientAuditRow],
    *,
    code_aliases: dict[str, tuple[str, ...]] | None = None,
) -> AliasGapReport:
    known_code_aliases = code_aliases or INGREDIENT_SLUG_ALIASES
    rows_by_slug = {row.slug: row for row in rows}

    without_db_aliases = sorted(row.slug for row in rows if not row.aliases)
    code_missing: dict[str, list[str]] = {}
    db_missing: dict[str, list[str]] = {}

    for slug, row in rows_by_slug.items():
        db_terms = _alias_set(row.name, row.aliases)
        code_terms_raw = list(known_code_aliases.get(slug, ()))
        code_terms = {_normalize_term(term) for term in code_terms_raw if _normalize_term(term)}

        missing_code_terms = [
            term for term in code_terms_raw if _normalize_term(term) not in db_terms
        ]
        if missing_code_terms:
            code_missing[slug] = sorted(missing_code_terms, key=_normalize_term)

        if code_terms:
            missing_db_terms = [
                alias for alias in row.aliases if _normalize_term(alias) not in code_terms
            ]
            if missing_db_terms:
                db_missing[slug] = sorted(missing_db_terms, key=_normalize_term)

    code_without_db = sorted(
        slug for slug in known_code_aliases if slug not in rows_by_slug
    )
    return AliasGapReport(
        ingredients_without_db_aliases=without_db_aliases,
        code_aliases_missing_in_db=dict(sorted(code_missing.items())),
        db_aliases_missing_in_code=dict(sorted(db_missing.items())),
        code_aliases_without_db_ingredient=code_without_db,
    )


def render_markdown_report(rows: list[IngredientAuditRow], gaps: AliasGapReport) -> str:
    sorted_rows = sorted(
        rows,
        key=lambda row: (-row.article_count, -row.ingredients_list_mentions, row.slug),
    )
    lines = [
        "# Audit ingredients Sahtein v4",
        "",
        "## Resume",
        "",
        f"- Ingredients canoniques: {len(rows)}",
        f"- Ingredients sans aliases DB: {len(gaps.ingredients_without_db_aliases)}",
        f"- Slugs avec aliases code manquants en DB: {len(gaps.code_aliases_missing_in_db)}",
        f"- Slugs declares dans le code mais absents de la DB: {len(gaps.code_aliases_without_db_ingredient)}",
        "",
        "## Ingredients",
        "",
        "| slug | nom | categorie | articles | mentions ingredients_list | aliases | exemples |",
        "|---|---|---|---:|---:|---|---|",
    ]

    for row in sorted_rows:
        aliases = ", ".join(row.aliases) if row.aliases else "-"
        examples = "<br>".join(row.example_articles) if row.example_articles else "-"
        lines.append(
            f"| {row.slug} | {row.name} | {row.category or '-'} | "
            f"{row.article_count} | {row.ingredients_list_mentions} | {aliases} | {examples} |"
        )

    lines.extend(["", "## Trous d'aliases", ""])
    if gaps.ingredients_without_db_aliases:
        lines.append("### Ingredients sans aliases stockes en DB")
        lines.append("")
        lines.append(", ".join(f"`{slug}`" for slug in gaps.ingredients_without_db_aliases))
        lines.append("")

    if gaps.code_aliases_missing_in_db:
        lines.append("### Aliases presents dans le code mais manquants en DB")
        lines.append("")
        for slug, aliases in gaps.code_aliases_missing_in_db.items():
            lines.append(f"- `{slug}`: {', '.join(aliases)}")
        lines.append("")

    if gaps.db_aliases_missing_in_code:
        lines.append("### Aliases DB absents du code")
        lines.append("")
        for slug, aliases in gaps.db_aliases_missing_in_code.items():
            lines.append(f"- `{slug}`: {', '.join(aliases)}")
        lines.append("")

    if gaps.code_aliases_without_db_ingredient:
        lines.append("### Slugs d'aliases code absents du referentiel DB")
        lines.append("")
        lines.append(", ".join(f"`{slug}`" for slug in gaps.code_aliases_without_db_ingredient))
        lines.append("")

    lines.extend([
        "## Lecture",
        "",
        "- Les aliases manquants en DB sont de bons candidats a migrer vers `Ingredient.aliases`.",
        "- Les ingredients sans aliases ne sont pas forcement faux, mais ils sont moins robustes aux variantes utilisateur.",
        "- Les mentions `ingredients_list` aident a reperer les ingredients presents dans le texte mais peu relies structurellement.",
        "",
    ])
    return "\n".join(lines)


def render_json_report(rows: list[IngredientAuditRow], gaps: AliasGapReport) -> str:
    payload = {
        "summary": {
            "ingredients": len(rows),
            "ingredients_without_db_aliases": len(gaps.ingredients_without_db_aliases),
            "code_aliases_missing_in_db": len(gaps.code_aliases_missing_in_db),
            "code_aliases_without_db_ingredient": len(gaps.code_aliases_without_db_ingredient),
        },
        "ingredients": [asdict(row) for row in rows],
        "gaps": asdict(gaps),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _coerce_aliases(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    return []


def _coerce_examples(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


async def fetch_ingredient_audit_rows(session: AsyncSession) -> list[IngredientAuditRow]:
    result = await session.execute(
        text(
            """
WITH ingredient_article_counts AS (
    SELECT
        i.id AS ingredient_id,
        COUNT(DISTINCT ai.article_id) AS article_count
    FROM ingredients i
    LEFT JOIN article_ingredients ai ON ai.ingredient_id = i.id
    GROUP BY i.id
),
ingredient_mentions AS (
    SELECT
        i.id AS ingredient_id,
        COUNT(DISTINCT c.id) AS ingredients_list_mentions
    FROM ingredients i
    LEFT JOIN chunks c
      ON c.kind = 'ingredients_list'
     AND (
        c.text ILIKE ('%%' || i.name || '%%')
        OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(COALESCE(i.aliases, '[]'::jsonb)) AS alias(value)
            WHERE c.text ILIKE ('%%' || alias.value || '%%')
        )
     )
    GROUP BY i.id
),
examples AS (
    SELECT
        ingredient_id,
        array_agg(example ORDER BY article_count DESC, external_id ASC)[:3] AS examples
    FROM (
        SELECT
            ai.ingredient_id,
            a.external_id,
            COUNT(*) AS article_count,
            (a.external_id::text || ': ' || a.title) AS example
        FROM article_ingredients ai
        JOIN articles a ON a.id = ai.article_id
        GROUP BY ai.ingredient_id, a.external_id, a.title
    ) ranked
    GROUP BY ingredient_id
)
SELECT
    i.slug,
    i.name,
    COALESCE(i.aliases, '[]'::jsonb) AS aliases,
    COALESCE(i.category, '') AS category,
    COALESCE(iac.article_count, 0) AS article_count,
    COALESCE(im.ingredients_list_mentions, 0) AS ingredients_list_mentions,
    COALESCE(e.examples, ARRAY[]::text[]) AS example_articles
FROM ingredients i
LEFT JOIN ingredient_article_counts iac ON iac.ingredient_id = i.id
LEFT JOIN ingredient_mentions im ON im.ingredient_id = i.id
LEFT JOIN examples e ON e.ingredient_id = i.id
ORDER BY iac.article_count DESC NULLS LAST, i.slug ASC
"""
        )
    )
    rows = result.mappings().all()
    return [
        IngredientAuditRow(
            slug=str(row["slug"] or ""),
            name=str(row["name"] or ""),
            aliases=_coerce_aliases(row["aliases"]),
            category=str(row["category"] or ""),
            article_count=int(row["article_count"] or 0),
            ingredients_list_mentions=int(row["ingredients_list_mentions"] or 0),
            example_articles=_coerce_examples(row["example_articles"]),
        )
        for row in rows
    ]


async def build_report(output_format: Literal["markdown", "json"]) -> str:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        rows = await fetch_ingredient_audit_rows(session)
    gaps = build_alias_gap_report(rows)
    if output_format == "json":
        return render_json_report(rows, gaps)
    return render_markdown_report(rows, gaps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit du referentiel ingredients Sahtein v4.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Format de sortie.",
    )
    parser.add_argument(
        "--out",
        default="docs/ingredient-audit.md",
        help="Chemin de sortie. Utilise stdout si vide.",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = _parse_args()
    content = await build_report(args.format)
    if not args.out:
        print(content)
        return
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"rapport ecrit: {output_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

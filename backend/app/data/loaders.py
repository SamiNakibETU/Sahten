"""
Data loaders for Sahtein 3.0
Loads and processes JSON datasets from the repository
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any

from app.models.config import settings
from app.models.schemas import (
    RecipeArticle,
    StructuredRecipe,
    GoldenExample,
    Ingredient,
    ArticleEnrichment,
)
from app.data.normalizers import (
    normalize_text,
    normalize_recipe_name,
    extract_slug_from_url,
)
from app.data.ingredient_normalizer import ingredient_normalizer

logger = logging.getLogger(__name__)


def _safe_str(value: Any) -> str:
    """Return a sanitized string without None surprises."""
    if isinstance(value, str):
        return value.strip()
    return ""


def _safe_list(value: Any) -> list:
    """Ensure we always manipulate a list."""
    if isinstance(value, list):
        return value
    return []


def _extract_text_from_html(html_text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    if not html_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = unescape(text)
    return " ".join(text.split()).strip()


def _normalize_datetime(value: datetime | None) -> datetime | None:
    """Ensure all datetimes are stored as naive UTC for consistent comparisons."""
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _derive_title(article_data: dict, recipe_name: str, url: str) -> str:
    """Best-effort strategy to determine article title."""
    title = _safe_str(article_data.get("title"))
    if title:
        return title

    if recipe_name:
        return recipe_name

    html_content = _safe_str(article_data.get("content_html"))
    if html_content:
        heading_match = re.search(r"<h[12][^>]*>(.*?)</h[12]>", html_content, re.IGNORECASE | re.DOTALL)
        if heading_match:
            heading = _extract_text_from_html(heading_match.group(1))
            if heading:
                return heading

    slug = extract_slug_from_url(url)
    if slug:
        return slug.replace("-", " ").strip()

    return ""


COURSE_KEYWORDS = {
    "mezze": ["mezze", "meze", "mezze froid", "mezze chaud", "entree", "entrée"],
    "dessert": ["dessert", "sucre", "gateau", "gâteau", "douceur"],
    "plat": ["plat", "plat principal", "plat familial", "cocotte", "riz", "ragout"],
    "soupe": ["soupe", "potage", "bouillon"],
    "boisson": ["boisson", "drink", "cocktail", "jus"],
    "pain": ["pain", "boulangerie", "manakish"],
}

FISH_KEYWORDS = {"poisson", "thon", "crevette", "saumon", "moule", "calamar", "gambas"}
MEAT_KEYWORDS = {"poulet", "veau", "boeuf", "agneau", "viande", "oeuf", "foie", "dinde", "saucisse"}

CUISINE_KEYWORDS = {
    "arménienne": ["armenien", "armenienne"],
    "syrienne": ["syrien", "syrienne"],
    "palestinienne": ["palestinien", "palestinienne"],
    "levantine": ["levantin", "levantine"],
    "libanaise": ["liban", "libanais", "libanaise"],
}


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        value = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        logger.debug("Unable to parse datetime: %s", value)
        return None


def _infer_course(tags: list[str], category: str | None) -> str | None:
    candidates = [normalize_text(tag) for tag in tags if tag]
    if category:
        candidates.append(normalize_text(category))

    for course, keywords in COURSE_KEYWORDS.items():
        for keyword in keywords:
            normalized_kw = normalize_text(keyword)
            if any(normalized_kw in candidate for candidate in candidates):
                return course

    return "plat" if candidates else None


def _extract_main_ingredients(raw_ingredients: list[str]) -> list[str]:
    cleaned: list[str] = []
    for entry in raw_ingredients or []:
        text = _safe_str(entry)
        if not text or text.endswith(":"):
            continue
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"\d+[\/\.,]?\d*\s*[a-zàâçéèêëîïôûùüÿñæœ\.]*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(c\.|cuill|sachet|tasse|verre|pincee)\b", " ", text, flags=re.IGNORECASE)
        normalized = normalize_text(text)
        if normalized:
            cleaned.append(normalized)

    counter = Counter()
    for phrase in cleaned:
        tokens = phrase.split()
        if not tokens:
            continue
        # take the last token to capture ingredient (skip leading verbs)
        counter[" ".join(tokens[-2:])] += 1

    results: list[str] = []
    seen: set[str] = set()
    for ingredient, _ in counter.most_common(12):
        equivalents = ingredient_normalizer.get_equivalents(ingredient)
        canonical = next(iter(sorted(equivalents))) if equivalents else ingredient
        if canonical and canonical not in seen:
            seen.add(canonical)
            results.append(canonical)
        if len(results) >= 5:
            break

    return results


def _infer_diet(main_ingredients: list[str]) -> str:
    normalized = {normalize_text(ing) for ing in main_ingredients}

    if any(word in normalized for word in FISH_KEYWORDS):
        return "poisson"
    if any(word in normalized for word in MEAT_KEYWORDS):
        return "viande"
    return "vege"


def _infer_cuisine(tags: list[str]) -> str:
    normalized_tags = [normalize_text(tag) for tag in tags if tag]
    for cuisine, keywords in CUISINE_KEYWORDS.items():
        for keyword in keywords:
            norm = normalize_text(keyword)
            if any(norm in tag for tag in normalized_tags):
                return cuisine
    return "libanaise"


def _calculate_editorial_score(has_chef: bool, tags: list[str], difficulty: str | None) -> float:
    score = 0.5
    if has_chef:
        score += 0.2
    if difficulty:
        score += 0.1
    if any("chef" in normalize_text(tag) for tag in tags):
        score += 0.1
    return min(score, 1.0)


def _calculate_recency_score(published: datetime | None, updated: datetime | None) -> float:
    reference = updated or published
    if not reference:
        return 0.4
    delta = datetime.now(timezone.utc) - reference
    days = max(delta.days, 0)
    window = 5 * 365  # 5 years
    score = 1.0 - min(days / window, 1.0)
    return max(0.1, score)


# ============================================================================
# Base 1 - OLJ Recipe Articles
# ============================================================================

def load_olj_articles() -> list[RecipeArticle]:
    """Load OLJ recipe articles from the refreshed dataset."""
    
    # Priority: Enriched > Configured Path > Standard > Fallback
    
    # 1. Enriched Data (if enabled)
    if settings.use_enriched_data:
        enriched_path = settings.data_dir / "data_base_OLJ_enriched.json"
        if enriched_path.exists():
            logger.info("Loading ENRICHED OLJ articles from %s", enriched_path)
            olj_path = enriched_path
        else:
            logger.warning("Enriched data not found at %s, falling back to standard.", enriched_path)
            olj_path = settings.olj_recipes_path
    else:
        olj_path = settings.olj_recipes_path

    # 2. Configured/Standard Path check
    if not olj_path.exists():
        fallback_paths = [
            settings.data_dir / "data_base_OLJ_final.json",
            settings.data_dir / "olj_recette_liban_a_table.json",
        ]
        for candidate in fallback_paths:
            if candidate.exists():
                olj_path = candidate
                break

    logger.info("Final OLJ source: %s", olj_path)

    with open(olj_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        raw_articles = raw_data.get("articles") or raw_data.get("data") or []
    else:
        raw_articles = raw_data

    logger.info("Found %d raw OLJ records", len(raw_articles))

    articles: list[RecipeArticle] = []
    skipped = 0
    enriched_count = 0

    for idx, article_data in enumerate(raw_articles):
        try:
            url = _safe_str(article_data.get("url"))
            if not url:
                skipped += 1
                continue

            slug = article_data.get("slug") or extract_slug_from_url(url) or f"article-{idx}"
            article_id = (
                _safe_str(article_data.get("id"))
                or _safe_str(article_data.get("article_id"))
                or slug
            )

            primary_title = (
                _safe_str(article_data.get("title"))
                or _safe_str(article_data.get("recipe_section_title"))
                or slug.replace("-", " ").title()
            )
            dish_name = _safe_str(article_data.get("recipe_section_title")) or primary_title
            normalized_title = normalize_recipe_name(primary_title)

            ingredients_raw = _safe_list(article_data.get("ingredients"))
            instructions_raw = _safe_list(article_data.get("instructions"))

            tags_raw = article_data.get("tags") or []
            if isinstance(tags_raw, list):
                tags = [tag for tag in tags_raw if isinstance(tag, str)]
            elif isinstance(tags_raw, str):
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            else:
                tags = []

            main_ingredients = _extract_main_ingredients(ingredients_raw)
            diet = _infer_diet(main_ingredients)
            course = _infer_course(tags, article_data.get("category"))
            cuisine = _infer_cuisine(tags)

            published = _parse_datetime(
                article_data.get("publication_date") or article_data.get("published_at")
            )
            updated = _parse_datetime(article_data.get("updated_at") or article_data.get("modified_at"))

            chef_name = _safe_str(article_data.get("chef_name"))
            difficulty = _safe_str(article_data.get("difficulty"))
            editorial_score = _calculate_editorial_score(bool(chef_name), tags, difficulty)
            recency_score = _calculate_recency_score(published, updated)

            description = _safe_str(article_data.get("recipe_description"))
            anecdote = _safe_str(article_data.get("chef_bio")) or description
            doc_text = _extract_text_from_html(
                " ".join(
                    [
                        description,
                        " ".join(instructions_raw),
                        " ".join(ingredients_raw),
                        chef_name,
                    ]
                )
            )

            # Mark recipe vs cultural article
            has_error = "error" in article_data
            has_ingredients = bool(ingredients_raw)
            
            # --- Enrichment Handling ---
            enrichment = None
            if "enrichment" in article_data and article_data["enrichment"]:
                try:
                    enrichment = ArticleEnrichment(**article_data["enrichment"])
                    enriched_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse enrichment for {article_id}: {e}")

            article = RecipeArticle(
                article_id=article_id,
                title=primary_title,
                normalized_title=normalized_title,
                slug=slug,
                url=url,
                chef=chef_name or None,
                author=_safe_str(article_data.get("author")) or None,
                section=_safe_str(article_data.get("category")) or "Liban à table",
                tags=tags,
                publish_date=published,
                modified_date=updated,
                popularity_score=editorial_score,
                short_summary=description[:200],
                description=description,
                anecdote=anecdote,
                tips=[],
                is_editor_pick=bool(article_data.get("is_editor_pick", False)),
                ingredients=[_safe_str(i) for i in ingredients_raw if _safe_str(i)],
                instructions=[_safe_str(step) for step in instructions_raw if _safe_str(step)],
                dish_name=dish_name,
                course=course,
                diet=diet,
                main_ingredients=main_ingredients,
                cuisine=cuisine,
                editorial_score=editorial_score,
                recency_score=recency_score,
                image_url=_safe_str(article_data.get("recipe_image_url")) or None,
                doc_text=doc_text,
                is_recipe=not has_error and has_ingredients,
                enrichment=enrichment
            )

            articles.append(article)

        except Exception as exc:
            skipped += 1
            logger.warning("Failed to parse OLJ article #%s: %s", idx, exc)

    logger.info("Loaded %d OLJ articles (skipped %d, enriched %d)", len(articles), skipped, enriched_count)
    return articles


# ============================================================================
# Base 2 - Structured Recipes
# ============================================================================

def load_structured_recipes() -> list[StructuredRecipe]:
    """Load structured recipes from Base 2"""
    logger.info(f"Loading structured recipes from {settings.base2_recipes_path}")

    with open(settings.base2_recipes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    recipes: list[StructuredRecipe] = []
    recipe_id_counter = 1

    # Base 2 is organized by category
    categories = [
        "mezzes_froids", "mezzes_chauds", "plats_principaux",
        "soupes_potages", "salades", "desserts", "boissons"
    ]

    for category in categories:
        if category not in data:
            continue

        for recipe_data in data[category]:
            try:
                # Parse ingredients
                ingredients = []
                for ing_data in recipe_data.get("ingredients", []):
                    if isinstance(ing_data, dict):
                        ingredients.append(Ingredient(**ing_data))
                    elif isinstance(ing_data, str):
                        # Simple string ingredient
                        ingredients.append(Ingredient(nom=ing_data))

                recipe = StructuredRecipe(
                    recipe_id=f"base2_{recipe_id_counter}",
                    name=recipe_data.get("nom", ""),
                    normalized_name=normalize_recipe_name(recipe_data.get("nom", "")),
                    category=category.replace("_", " ").title(),
                    ingredients=ingredients,
                    etapes=recipe_data.get("etapes", []),
                    nombre_de_personnes=recipe_data.get("nombre_de_personnes"),
                    temps_preparation=recipe_data.get("temps_preparation"),
                    temps_cuisson=recipe_data.get("temps_cuisson"),
                    difficulte=recipe_data.get("difficulte"),
                    tags=[category],
                )

                recipes.append(recipe)
                recipe_id_counter += 1

            except Exception as e:
                logger.warning(f"Failed to parse recipe {recipe_data.get('nom')}: {e}")
                continue

    logger.info(f"Loaded {len(recipes)} structured recipes")
    return recipes


# ============================================================================
# Golden Examples
# ============================================================================

def load_golden_examples() -> list[GoldenExample]:
    """Load golden examples from test dataset"""
    logger.info(f"Loading golden examples from {settings.golden_examples_path}")

    with open(settings.golden_examples_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: list[GoldenExample] = []

    for example_data in data.get("examples", []):
        try:
            metadata = example_data.get("metadata", {})

            example = GoldenExample(
                id=example_data.get("id", ""),
                scenario=example_data.get("scenario", ""),
                title=example_data.get("title", ""),
                user_query=example_data.get("user_query", ""),
                response=example_data.get("response", ""),
                expected_intent=metadata.get("intent"),
                expected_url=metadata.get("url"),
                metadata=metadata,
            )

            examples.append(example)

        except Exception as e:
            logger.warning(f"Failed to parse golden example {example_data.get('id')}: {e}")
            continue

    logger.info(f"Loaded {len(examples)} golden examples")
    return examples


# ============================================================================
# Data Cache (singleton pattern for efficiency)
# ============================================================================

class DataCache:
    """Singleton cache for loaded data"""

    _instance = None
    _olj_articles: list[RecipeArticle] | None = None
    _structured_recipes: list[StructuredRecipe] | None = None
    _golden_examples: list[GoldenExample] | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_olj_articles(self, reload: bool = False) -> list[RecipeArticle]:
        """Get OLJ articles (cached)"""
        if self._olj_articles is None or reload:
            self._olj_articles = load_olj_articles()
        return self._olj_articles

    def get_structured_recipes(self, reload: bool = False) -> list[StructuredRecipe]:
        """Get structured recipes (cached)"""
        if self._structured_recipes is None or reload:
            self._structured_recipes = load_structured_recipes()
        return self._structured_recipes

    def get_golden_examples(self, reload: bool = False) -> list[GoldenExample]:
        """Get golden examples (cached)"""
        if self._golden_examples is None or reload:
            self._golden_examples = load_golden_examples()
        return self._golden_examples


# Global cache instance
data_cache = DataCache()

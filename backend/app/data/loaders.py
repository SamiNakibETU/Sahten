"""
Data loaders for Sahtein 3.0
Loads and processes JSON datasets from the repository
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.config import settings
from app.models.schemas import (
    RecipeArticle,
    StructuredRecipe,
    GoldenExample,
    Ingredient,
)
from app.data.normalizers import (
    normalize_text,
    normalize_recipe_name,
    extract_slug_from_url,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Base 1 - OLJ Recipe Articles
# ============================================================================

def load_olj_articles() -> list[RecipeArticle]:
    """Load OLJ recipe articles from Base 1"""
    logger.info(f"Loading OLJ articles from {settings.olj_recipes_path}")

    with open(settings.olj_recipes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles: list[RecipeArticle] = []

    for article_data in data.get("articles", []):
        try:
            # Parse dates
            published = None
            if article_data.get("published"):
                try:
                    published = datetime.fromisoformat(article_data["published"])
                except Exception:
                    pass

            modified = None
            if article_data.get("modified"):
                try:
                    modified = datetime.fromisoformat(article_data["modified"])
                except Exception:
                    pass

            # Extract recipe name
            recipe_name = ""
            if article_data.get("recipe"):
                recipe_name = article_data["recipe"].get("name", "")

            # Build article
            url = article_data.get("url", "")
            title = article_data.get("title", recipe_name)

            article = RecipeArticle(
                article_id=article_data.get("id", url),
                title=title,
                normalized_title=normalize_recipe_name(title),
                slug=extract_slug_from_url(url),
                url=url,
                chef=extract_chef_from_article(article_data),
                author=article_data.get("author"),
                section="Liban à table",
                tags=parse_tags(article_data.get("tags")),
                publish_date=published,
                modified_date=modified,
                popularity_score=calculate_popularity(published, modified),
                short_summary=article_data.get("description", "")[:200],
                description=article_data.get("description", ""),
                anecdote=extract_anecdote(article_data),
                tips=extract_tips(article_data),
                is_editor_pick=is_editor_pick(article_data),
            )

            articles.append(article)

        except Exception as e:
            logger.warning(f"Failed to parse article {article_data.get('url')}: {e}")
            continue

    logger.info(f"Loaded {len(articles)} OLJ articles")
    return articles


def extract_chef_from_article(article_data: dict) -> str | None:
    """Extract chef name from article"""
    # Try to find chef in title or description
    title = article_data.get("title", "").lower()
    desc = article_data.get("description", "").lower()

    # Common patterns: "de [Chef Name]", "par [Chef Name]"
    import re

    patterns = [
        r"de ([A-Z][a-z]+ [A-Z][a-z]+(?:-[A-Z][a-z]+)?)",
        r"par ([A-Z][a-z]+ [A-Z][a-z]+(?:-[A-Z][a-z]+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, article_data.get("title", ""))
        if match:
            return match.group(1)

    return None


def parse_tags(tags: Any) -> list[str]:
    """Parse tags from various formats"""
    if not tags:
        return []
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if t]
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def calculate_popularity(published: datetime | None, modified: datetime | None) -> float:
    """Calculate popularity score based on recency"""
    if not published:
        return 0.5

    # More recent articles get higher scores
    days_old = (datetime.now(published.tzinfo) - published).days
    recency_score = max(0, 1.0 - (days_old / 365.0))  # Decay over 1 year

    # Bonus for recently modified
    if modified and modified > published:
        days_since_modified = (datetime.now(modified.tzinfo) - modified).days
        if days_since_modified < 30:
            recency_score += 0.2

    return min(1.0, recency_score)


def is_editor_pick(article_data: dict) -> bool:
    """Determine if article is an editor's pick"""
    # For now, consider recent articles with chef names as editor picks
    # This can be enhanced with actual metadata if available
    return False  # TODO: Add logic based on actual editorial signals


def extract_anecdote(article_data: dict) -> str:
    """Extract anecdote or story from article"""
    # Try to find storytelling elements in description
    desc = article_data.get("description", "")
    # For now, return first part of description as anecdote
    if len(desc) > 100:
        return desc[:150] + "..."
    return desc


def extract_tips(article_data: dict) -> list[str]:
    """Extract cooking tips from article"""
    # Tips might be in instructions
    tips = []
    if article_data.get("recipe", {}).get("instructions"):
        instructions = article_data["recipe"]["instructions"]
        for instruction in instructions:
            if isinstance(instruction, str) and ("astuce" in instruction.lower() or "secret" in instruction.lower()):
                tips.append(instruction)
    return tips


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

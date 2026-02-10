"""
Recipe Enricher
===============

Enriches raw recipe data with structured metadata via LLM:
- prep_time_minutes: Cooking time extracted from text
- main_ingredients: Top 5 key ingredients
- occasion: When to serve (quotidien, fete, ramadan, etc.)
- mood: Emotional qualities (frais, reconfortant, leger, etc.)
- dietary: Dietary restrictions (vegetarien, vegan, sans-gluten)
- keywords: Search keywords including Arabic transliterations

This enables:
- Better retrieval without embeddings
- Mood-based queries ("something fresh")
- Occasion filtering ("ramadan recipe")
- Dietary filtering ("vegetarian")
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ..core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EnrichedRecipe:
    """Enriched recipe with all metadata."""
    id: str
    title: str
    url: str
    author: Optional[str]
    published_date: Optional[str]
    content: str
    image_url: Optional[str]
    
    # Enriched fields
    categories: List[str]
    difficulty: str
    is_lebanese: bool
    keywords: List[str]
    prep_time_minutes: Optional[int]
    main_ingredients: List[str]
    occasion: List[str]
    mood: List[str]
    dietary: List[str]
    
    # For retrieval
    search_text: str


ENRICHMENT_PROMPT = """Tu es un expert en cuisine libanaise et méditerranéenne.
Analyse cette recette et extrais les métadonnées structurées.

Recette:
Titre: {title}
Contenu: {content}

Réponds en JSON strict avec ces champs:

{{
  "categories": ["liste de: mezze_froid, mezze_chaud, entree, plat_principal, dessert, salade, soupe, boisson, sauce"],
  "difficulty": "facile | moyen | difficile",
  "is_lebanese": true ou false,
  "keywords": ["5-10 mots-clés incluant translitérations arabes si applicable"],
  "prep_time_minutes": nombre ou null si pas mentionné,
  "main_ingredients": ["les 5 ingrédients principaux"],
  "occasion": ["liste de: quotidien, fete, ramadan, paques, noel, ete, hiver, pique-nique, buffet"],
  "mood": ["liste de: frais, reconfortant, leger, copieux, festif, rapide, traditionnel, moderne"],
  "dietary": ["liste parmi: vegetarien, vegan, sans-gluten, sans-lactose, halal, casher ou liste vide"]
}}

Retourne UNIQUEMENT le JSON, sans texte avant ou après."""


class RecipeEnricher:
    """
    Enriches recipes via OpenAI API.
    
    Usage:
        enricher = RecipeEnricher()
        enriched = await enricher.enrich(raw_recipe)
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model
    
    async def enrich(self, recipe: Dict[str, Any]) -> EnrichedRecipe:
        """
        Enrich a single recipe.
        
        Args:
            recipe: Raw recipe dict with at least: id, title, content, url
        
        Returns:
            EnrichedRecipe with all metadata filled
        """
        title = recipe.get("title", "")
        content = recipe.get("content", "")
        
        # Call LLM for enrichment
        prompt = ENRICHMENT_PROMPT.format(title=title, content=content[:3000])
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            
            raw_json = response.choices[0].message.content.strip()
            
            # Clean up potential markdown code blocks
            if raw_json.startswith("```"):
                raw_json = re.sub(r"```json?\s*", "", raw_json)
                raw_json = raw_json.replace("```", "")
            
            enrichment = json.loads(raw_json)
            
        except Exception as e:
            logger.error("Enrichment failed for %s: %s", title, e)
            # Return default enrichment on failure
            enrichment = {
                "categories": ["autre"],
                "difficulty": "moyen",
                "is_lebanese": True,
                "keywords": [title.lower()],
                "prep_time_minutes": None,
                "main_ingredients": [],
                "occasion": ["quotidien"],
                "mood": [],
                "dietary": [],
            }
        
        # Build search text
        search_text = self._build_search_text(
            title=title,
            content=content,
            author=recipe.get("author"),
            keywords=enrichment.get("keywords", []),
            ingredients=enrichment.get("main_ingredients", []),
            mood=enrichment.get("mood", []),
            occasion=enrichment.get("occasion", []),
        )
        
        return EnrichedRecipe(
            id=recipe.get("id", ""),
            title=title,
            url=recipe.get("url", ""),
            author=recipe.get("author"),
            published_date=recipe.get("published_date"),
            content=content,
            image_url=recipe.get("image_url"),
            categories=enrichment.get("categories", ["autre"]),
            difficulty=enrichment.get("difficulty", "moyen"),
            is_lebanese=enrichment.get("is_lebanese", True),
            keywords=enrichment.get("keywords", []),
            prep_time_minutes=enrichment.get("prep_time_minutes"),
            main_ingredients=enrichment.get("main_ingredients", []),
            occasion=enrichment.get("occasion", []),
            mood=enrichment.get("mood", []),
            dietary=enrichment.get("dietary", []),
            search_text=search_text,
        )
    
    def _build_search_text(
        self,
        title: str,
        content: str,
        author: Optional[str],
        keywords: List[str],
        ingredients: List[str],
        mood: List[str],
        occasion: List[str],
    ) -> str:
        """Build comprehensive search text for TF-IDF."""
        parts = [
            title,
            author or "",
            content[:1000],  # First 1000 chars
            " ".join(keywords),
            " ".join(ingredients),
            " ".join(mood),
            " ".join(occasion),
        ]
        return " ".join(p for p in parts if p).strip()
    
    async def enrich_batch(
        self,
        recipes: List[Dict[str, Any]],
        *,
        on_progress: Optional[callable] = None,
    ) -> List[EnrichedRecipe]:
        """
        Enrich multiple recipes.
        
        Args:
            recipes: List of raw recipe dicts
            on_progress: Optional callback(current, total)
        
        Returns:
            List of EnrichedRecipe
        """
        results = []
        total = len(recipes)
        
        for i, recipe in enumerate(recipes):
            try:
                enriched = await self.enrich(recipe)
                results.append(enriched)
            except Exception as e:
                logger.error("Failed to enrich recipe %s: %s", recipe.get("id"), e)
                continue
            
            if on_progress:
                on_progress(i + 1, total)
        
        return results


async def enrich_recipe(recipe: Dict[str, Any]) -> EnrichedRecipe:
    """Convenience function to enrich a single recipe."""
    enricher = RecipeEnricher()
    return await enricher.enrich(recipe)


def enriched_to_canonical(enriched: EnrichedRecipe) -> Dict[str, Any]:
    """Convert EnrichedRecipe to canonical JSON format for storage."""
    # Map categories to canonical format
    cat_map = {
        "mezze_froid": "mezze_froid",
        "mezze_chaud": "mezze_chaud",
        "mezze": "mezze_froid",
        "entree": "entree",
        "entrée": "entree",
        "plat_principal": "plat_principal",
        "plat principal": "plat_principal",
        "dessert": "dessert",
        "salade": "salade",
        "soupe": "soupe",
        "boisson": "boisson",
        "sauce": "sauces",
        "sauces": "sauces",
    }
    
    category_canonical = "autre"
    for cat in enriched.categories:
        if cat.lower() in cat_map:
            category_canonical = cat_map[cat.lower()]
            break
    
    diff_map = {"facile": "facile", "moyen": "moyenne", "difficile": "difficile"}
    difficulty_canonical = diff_map.get(enriched.difficulty.lower(), "non_specifie")
    
    return {
        "url": enriched.url,
        "title": enriched.title,
        "chef_name": enriched.author,
        "cuisine_type": "libanaise" if enriched.is_lebanese else "internationale",
        "is_lebanese": enriched.is_lebanese,
        "is_recipe": True,
        "category_canonical": category_canonical,
        "difficulty_canonical": difficulty_canonical,
        "tags": enriched.keywords,
        "main_ingredients": enriched.main_ingredients,
        "aliases": [],
        "search_text": enriched.search_text,
        "raw_category": enriched.categories[0] if enriched.categories else None,
        "raw_difficulty": enriched.difficulty,
        "raw_enrichment_present": True,
        # MVP enhanced fields
        "prep_time_minutes": enriched.prep_time_minutes,
        "occasion": enriched.occasion,
        "mood": enriched.mood,
        "dietary": enriched.dietary,
    }




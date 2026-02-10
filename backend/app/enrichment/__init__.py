"""
Enrichment Module
=================

Automatic recipe enrichment via LLM for:
- New recipes from CMS webhook
- Batch re-enrichment of existing recipes
"""

from .enricher import RecipeEnricher, enrich_recipe

__all__ = ["RecipeEnricher", "enrich_recipe"]




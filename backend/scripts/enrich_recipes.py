#!/usr/bin/env python3
"""
Recipe Enrichment Script
========================

Re-enriches existing recipes with enhanced metadata:
- prep_time_minutes
- main_ingredients (top 5)
- occasion (quotidien, fete, ramadan, etc.)
- mood (frais, reconfortant, leger, etc.)
- dietary (vegetarien, vegan, sans-gluten)

Usage:
    cd Sahten_MVP/backend
    python scripts/enrich_recipes.py

    # With options:
    python scripts/enrich_recipes.py --dry-run        # Preview without saving
    python scripts/enrich_recipes.py --limit 10       # Only enrich 10 recipes
    python scripts/enrich_recipes.py --force          # Re-enrich all (even already enriched)
    python scripts/enrich_recipes.py --output new.json # Save to different file

Environment:
    OPENAI_API_KEY: Required for LLM enrichment
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.enrichment.enricher import RecipeEnricher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_canonical(path: Path) -> list:
    """Load existing canonical recipes."""
    if not path.exists():
        logger.error(f"File not found: {path}")
        return []
    
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return []


def save_canonical(recipes: list, path: Path) -> None:
    """Save enriched recipes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(recipes)} recipes to {path}")


def needs_enrichment(recipe: dict) -> bool:
    """Check if recipe needs MVP enhanced enrichment."""
    # Check for new MVP fields
    has_mood = bool(recipe.get("mood"))
    has_occasion = bool(recipe.get("occasion"))
    has_dietary = bool(recipe.get("dietary"))
    has_prep_time = recipe.get("prep_time_minutes") is not None
    
    # If any MVP field is missing, needs enrichment
    return not (has_mood and has_occasion and has_dietary)


async def enrich_recipe_mvp(enricher: RecipeEnricher, recipe: dict) -> dict:
    """Enrich a single recipe with MVP fields."""
    # Build content from existing fields
    content_parts = [
        recipe.get("title", ""),
        recipe.get("search_text", ""),
    ]
    content = " ".join(p for p in content_parts if p)
    
    # Create payload for enricher
    payload = {
        "id": recipe.get("url", "unknown"),
        "title": recipe.get("title", ""),
        "url": recipe.get("url", ""),
        "author": recipe.get("chef_name"),
        "content": content[:3000],  # Limit for API
    }
    
    try:
        enriched = await enricher.enrich(payload)
        
        # Merge new fields into existing recipe
        recipe_copy = recipe.copy()
        recipe_copy["prep_time_minutes"] = enriched.prep_time_minutes
        recipe_copy["occasion"] = enriched.occasion
        recipe_copy["mood"] = enriched.mood
        recipe_copy["dietary"] = enriched.dietary
        
        # Update main_ingredients if empty
        if not recipe_copy.get("main_ingredients"):
            recipe_copy["main_ingredients"] = enriched.main_ingredients
        
        # Update keywords/tags
        existing_tags = set(recipe_copy.get("tags", []))
        new_keywords = set(enriched.keywords)
        recipe_copy["tags"] = list(existing_tags | new_keywords)
        
        return recipe_copy
        
    except Exception as e:
        logger.error(f"Failed to enrich {recipe.get('title')}: {e}")
        return recipe  # Return unchanged


async def main():
    parser = argparse.ArgumentParser(description="Enrich recipes with MVP metadata")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--limit", type=int, help="Limit number of recipes to enrich")
    parser.add_argument("--force", action="store_true", help="Re-enrich all recipes")
    parser.add_argument("--output", help="Output file path (default: overwrite input)")
    parser.add_argument("--input", help="Input file path", default="../data/olj_canonical.json")
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    
    output_path = Path(args.output) if args.output else input_path
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    
    # Check API key
    settings = get_settings()
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY not set. Please set it in .env or environment.")
        sys.exit(1)
    
    # Load recipes
    logger.info(f"Loading recipes from {input_path}")
    recipes = load_canonical(input_path)
    
    if not recipes:
        logger.error("No recipes found")
        sys.exit(1)
    
    logger.info(f"Loaded {len(recipes)} recipes")
    
    # Filter recipes needing enrichment
    if args.force:
        to_enrich = recipes
    else:
        to_enrich = [r for r in recipes if needs_enrichment(r)]
    
    if args.limit:
        to_enrich = to_enrich[:args.limit]
    
    logger.info(f"Recipes to enrich: {len(to_enrich)}")
    
    if not to_enrich:
        logger.info("All recipes already enriched. Use --force to re-enrich.")
        return
    
    if args.dry_run:
        logger.info("DRY RUN - Would enrich these recipes:")
        for r in to_enrich[:10]:
            logger.info(f"  - {r.get('title')}")
        if len(to_enrich) > 10:
            logger.info(f"  ... and {len(to_enrich) - 10} more")
        return
    
    # Enrich
    enricher = RecipeEnricher()
    enriched_count = 0
    failed_count = 0
    
    # Build URL -> index map for updating
    url_to_idx = {r.get("url"): i for i, r in enumerate(recipes)}
    
    for i, recipe in enumerate(to_enrich):
        try:
            logger.info(f"[{i+1}/{len(to_enrich)}] Enriching: {recipe.get('title')[:50]}")
            enriched = await enrich_recipe_mvp(enricher, recipe)
            
            # Update in main list
            url = recipe.get("url")
            if url in url_to_idx:
                recipes[url_to_idx[url]] = enriched
                enriched_count += 1
            
            # Progress
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(to_enrich)} ({enriched_count} enriched)")
                
        except Exception as e:
            logger.error(f"Error enriching {recipe.get('title')}: {e}")
            failed_count += 1
            continue
    
    # Summary
    logger.info(f"\nEnrichment complete:")
    logger.info(f"  Enriched: {enriched_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Skipped: {len(recipes) - len(to_enrich)}")
    
    # Save
    if not args.dry_run:
        # Backup original
        backup_path = output_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        if input_path.exists() and output_path == input_path:
            import shutil
            shutil.copy(input_path, backup_path)
            logger.info(f"Backup saved to {backup_path}")
        
        save_canonical(recipes, output_path)


if __name__ == "__main__":
    asyncio.run(main())




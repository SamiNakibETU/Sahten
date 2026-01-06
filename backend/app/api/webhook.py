"""
CMS Webhook API
===============

Endpoint for receiving new recipes from the journal's CMS.

Integration Flow:
  1. CMS publishes a new recipe
  2. CMS calls POST /api/webhook/recipe with recipe data
  3. Sahten enriches the recipe via LLM
  4. Sahten adds it to olj_canonical.json
  5. Sahten reloads the retriever index
  6. Recipe is immediately searchable

Authentication:
  - X-Webhook-Secret header must match WEBHOOK_SECRET env var
  
Payload Format:
  {
    "id": "article_12345",
    "title": "Fattouch aux herbes fraiches",
    "url": "https://lorientlejour.com/article/12345",
    "author": "Maya Sfeir",
    "published_date": "2025-01-05",
    "content": "La fattouch est une salade...",
    "image_url": "https://..."
  }
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, HttpUrl

from ..bot import get_bot
from ..core.config import get_settings
from ..enrichment.enricher import RecipeEnricher, enriched_to_canonical

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


# ============================================================================
# MODELS
# ============================================================================

class RecipePayload(BaseModel):
    """Recipe data from CMS."""
    id: str
    title: str
    url: str
    author: Optional[str] = None
    published_date: Optional[str] = None
    content: str
    image_url: Optional[str] = None


class WebhookResponse(BaseModel):
    """Webhook response."""
    status: str
    id: str
    message: str
    enriched: bool


class WebhookError(BaseModel):
    """Webhook error response."""
    error: str
    detail: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_canonical_path() -> Path:
    """Get path to olj_canonical.json."""
    settings = get_settings()
    base_dir = Path(__file__).parent.parent.parent.parent
    return base_dir / settings.olj_canonical_path


def load_canonical() -> list:
    """Load current canonical recipes."""
    path = get_canonical_path()
    if not path.exists():
        return []
    
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to load canonical: %s", e)
        return []


def save_canonical(recipes: list) -> None:
    """Save canonical recipes."""
    path = get_canonical_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)


def log_webhook_event(recipe_id: str, event: str, details: Optional[dict] = None) -> None:
    """Log webhook event to stdout and optionally Redis."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "recipe_id": recipe_id,
        "details": details or {},
    }
    print(f"[WEBHOOK] {json.dumps(log_entry, ensure_ascii=False)}")
    
    # TODO: Log to Upstash Redis if available
    # redis = get_redis()
    # if redis:
    #     redis.lpush("sahten:webhook:events", json.dumps(log_entry))


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/recipe", response_model=WebhookResponse)
async def receive_recipe(
    payload: RecipePayload,
    x_webhook_secret: str = Header(..., description="Webhook authentication secret"),
):
    """
    Receive a new recipe from CMS.
    
    1. Validates webhook secret
    2. Enriches recipe via LLM (categories, mood, ingredients, etc.)
    3. Adds to canonical dataset
    4. Reloads retriever index
    5. Recipe is immediately searchable
    
    Returns:
        WebhookResponse with status and enrichment details
    
    Raises:
        401: Invalid webhook secret
        500: Enrichment or storage failed
    """
    settings = get_settings()
    
    # 1. Validate webhook secret
    expected_secret = settings.webhook_secret or os.getenv("WEBHOOK_SECRET", "")
    
    if not expected_secret:
        logger.warning("WEBHOOK_SECRET not configured - accepting any request (DEV MODE)")
    elif x_webhook_secret != expected_secret:
        log_webhook_event(payload.id, "auth_failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook secret",
        )
    
    log_webhook_event(payload.id, "received", {"title": payload.title})
    
    # 2. Check for duplicates
    canonical = load_canonical()
    existing_urls = {r.get("url") for r in canonical}
    
    if payload.url in existing_urls:
        log_webhook_event(payload.id, "duplicate", {"url": payload.url})
        return WebhookResponse(
            status="skipped",
            id=payload.id,
            message="Recipe already exists",
            enriched=False,
        )
    
    # 3. Enrich via LLM (if enabled)
    enriched_data = None
    if settings.auto_enrich_on_webhook:
        try:
            enricher = RecipeEnricher()
            enriched = await enricher.enrich(payload.model_dump())
            enriched_data = enriched_to_canonical(enriched)
            log_webhook_event(payload.id, "enriched", {
                "categories": enriched_data.get("category_canonical"),
                "mood": enriched_data.get("mood"),
            })
        except Exception as e:
            logger.error("Enrichment failed for %s: %s", payload.id, e)
            # Fall back to minimal enrichment
            enriched_data = {
                "url": payload.url,
                "title": payload.title,
                "chef_name": payload.author,
                "is_lebanese": True,
                "is_recipe": True,
                "category_canonical": "autre",
                "difficulty_canonical": "non_specifie",
                "tags": [],
                "main_ingredients": [],
                "aliases": [],
                "search_text": f"{payload.title} {payload.content[:500]}",
                "raw_enrichment_present": False,
            }
    else:
        # No enrichment - store minimal data
        enriched_data = {
            "url": payload.url,
            "title": payload.title,
            "chef_name": payload.author,
            "is_lebanese": True,
            "is_recipe": True,
            "category_canonical": "autre",
            "difficulty_canonical": "non_specifie",
            "tags": [],
            "main_ingredients": [],
            "aliases": [],
            "search_text": f"{payload.title} {payload.content[:500]}",
            "raw_enrichment_present": False,
        }
    
    # 4. Add to canonical
    canonical.append(enriched_data)
    
    try:
        save_canonical(canonical)
        log_webhook_event(payload.id, "saved")
    except Exception as e:
        logger.error("Failed to save canonical: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save recipe: {str(e)}",
        )
    
    # 5. Reload retriever
    try:
        bot = get_bot()
        bot.retriever.reload()
        log_webhook_event(payload.id, "indexed")
    except Exception as e:
        logger.error("Failed to reload retriever: %s", e)
        # Recipe is saved but not indexed - will be indexed on next restart
    
    return WebhookResponse(
        status="indexed",
        id=payload.id,
        message=f"Recipe '{payload.title}' added and indexed",
        enriched=settings.auto_enrich_on_webhook,
    )


@router.get("/health")
async def webhook_health():
    """Webhook health check."""
    settings = get_settings()
    return {
        "status": "ready",
        "auto_enrich": settings.auto_enrich_on_webhook,
        "webhook_configured": bool(settings.webhook_secret),
    }




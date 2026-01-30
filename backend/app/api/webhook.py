"""
CMS Webhook API
===============

Endpoint for receiving new recipes from the journal's CMS.

Integration Flow (Two-Step):
  1. CMS publishes a new recipe
  2. CMS calls POST /api/webhook/recipe with { article_id, action }
  3. Sahten fetches full recipe data via OLJ API
  4. Sahten enriches the recipe via LLM
  5. Sahten adds it to olj_canonical.json
  6. Sahten reloads the retriever index
  7. Recipe is immediately searchable

Authentication:
  - X-Webhook-Signature header contains HMAC SHA256 signature
  - Format: sha256=<hex_digest>
  - Signature is computed over the raw JSON payload using WEBHOOK_SECRET

Payload Format (from CMS):
  {
    "article_id": 1227694,
    "action": "publish"  // or "update", "delete"
  }
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import re
import httpx

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from ..bot import get_bot
from ..core.config import get_settings
from ..enrichment.enricher import RecipeEnricher, enriched_to_canonical
from ..schemas.canonical import CanonicalRecipeDoc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


# ============================================================================
# MODELS
# ============================================================================


class WebhookPayload(BaseModel):
    """Webhook payload from CMS (minimal - just article_id)."""

    article_id: Optional[int] = None
    action: Optional[str] = "publish"  # publish, update, delete


class WhiteBeardPayload(BaseModel):
    """
    WhiteBeard automation webhook payload.
    
    WhiteBeard sends automation data, we need to extract article_id from it.
    The exact field names depend on how OLJ configured their automation.
    
    Possible fields where article_id might be:
    - automationId (but this is the automation ID, not article)
    - subject (might contain article ID or title)
    - Custom fields configured in WhiteBeard
    """
    
    automationId: Optional[str] = None
    subject: Optional[str] = None
    # Add other fields as we discover them
    # These are optional to capture any structure
    
    class Config:
        extra = "allow"  # Accept any additional fields


class RecipeData(BaseModel):
    """Recipe data fetched from OLJ API."""

    id: str
    title: str
    url: str
    author: Optional[str] = None
    published_date: Optional[str] = None
    content: str
    ingredients: str = ""
    image_url: Optional[str] = None
    category: Optional[str] = None
    keywords: Optional[str] = None


class WebhookResponse(BaseModel):
    """Webhook response."""

    status: str
    article_id: int
    message: str
    enriched: bool


class WebhookError(BaseModel):
    """Webhook error response."""

    error: str
    detail: str


# ============================================================================
# SIGNATURE VALIDATION (WhiteBeard HMAC SHA256)
# ============================================================================


def verify_webhook_signature(
    payload: bytes, signature_header: str, secret: str
) -> bool:
    """
    Validate WhiteBeard webhook signature (HMAC SHA256).

    WhiteBeard CMS signs webhooks using HMAC SHA256:
    - Header: X-Webhook-Signature
    - Format: sha256=<hex_digest>

    Args:
        payload: Raw request body (bytes)
        signature_header: Value of X-Webhook-Signature header
        secret: Configured WEBHOOK_SECRET

    Returns:
        True if signature is valid, False otherwise

    Reference: https://docs.whitebeard.net/marketingguides/setting_up_automations/#webhook-secret
    """
    if not signature_header or not secret:
        return False

    # Calculate expected signature
    expected_sig = (
        "sha256="
        + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    )

    # Use timing-safe comparison to prevent timing attacks
    return hmac.compare_digest(expected_sig, signature_header)


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

    def json_serializer(obj):
        """Handle non-serializable types like Pydantic HttpUrl."""
        if hasattr(obj, '__str__'):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2, default=json_serializer)


def log_webhook_event(
    article_id: int, event: str, details: Optional[dict] = None
) -> None:
    """Log webhook event to stdout and optionally Redis."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "article_id": article_id,
        "details": details or {},
    }
    print(f"[WEBHOOK] {json.dumps(log_entry, ensure_ascii=False)}")


def strip_html_tags(html: str) -> str:
    """Remove HTML tags from text."""
    if not html:
        return ""
    text = re.sub(r"\[\[image[^\]]*\]\]", "", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================================
# OLJ API INTEGRATION
# ============================================================================


async def fetch_recipe_from_api(article_id: int) -> Optional[RecipeData]:
    """
    Fetch recipe data from OLJ CMS API.

    API Endpoint: GET https://api.lorientlejour.com/cms/content/{article_id}
    Headers: API-Key: {olj_api_key}

    Returns:
        RecipeData object or None if not found
    """
    settings = get_settings()

    # OLJ API configuration check (no sensitive data logged)
    if settings.olj_api_key:
        logger.debug("OLJ API key configured (length: %d)", len(settings.olj_api_key))
    logger.debug("OLJ API base: %s", settings.olj_api_base)

    if not settings.olj_api_key:
        logger.error("OLJ_API_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OLJ API key not configured",
        )

    url = f"{settings.olj_api_base}/content/{article_id}"

    # Headers for OLJ CMS API (API-Key format per WhiteBeard docs)
    headers = {
        "API-Key": settings.olj_api_key,
        "Accept": "application/json",
    }
    logger.info("Fetching article %d from OLJ API", article_id)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            logger.info("OLJ API response: %d", response.status_code)

            if response.status_code == 404:
                logger.warning("Recipe not found in API: %s", article_id)
                return None

            response.raise_for_status()
            response_data = response.json()
            
            # WhiteBeard API returns { "data": [...] } structure
            # See: https://docs.whitebeard.net/api/operations/d68912ffb167a021bbfac4780cf30512/
            
            # Extract the content item from data array
            data_array = response_data.get("data", [])
            if not data_array:
                logger.warning("Empty data array in API response for article %s", article_id)
                return None
            
            # Get first item from data array
            item = data_array[0] if isinstance(data_array, list) else data_array
            
            # Extract title (required field in WhiteBeard API)
            title = item.get("title") or f"Recipe {article_id}"
            logger.info("Fetched recipe: %s", title.strip()[:50])
            
            # Extract author from authors array
            authors = item.get("authors", [])
            author_name = None
            if authors and len(authors) > 0:
                author_name = authors[0].get("name")
            
            # Extract content - can be string or object with html field
            contents = item.get("contents", "")
            if isinstance(contents, dict):
                content_text = contents.get("html", "")
            else:
                content_text = str(contents) if contents else ""
            
            # Extract image from attachments array
            attachments = item.get("attachments", [])
            image_url = None
            if attachments and len(attachments) > 0:
                image_url = attachments[0].get("url")
            
            # Extract category from categories array
            categories = item.get("categories", [])
            category_name = None
            if categories and len(categories) > 0:
                category_name = categories[0].get("name")
            
            # Extract keywords from keywords array
            keywords_list = item.get("keywords", [])
            keywords_str = None
            if keywords_list:
                keywords_str = ", ".join([kw.get("name", "") for kw in keywords_list if kw.get("name")])
            
            return RecipeData(
                id=str(article_id),
                title=title,
                url=item.get("url") or f"https://www.lorientlejour.com/article/{article_id}",
                author=author_name,
                published_date=item.get("firstPublished"),
                content=content_text,
                ingredients=item.get("summary", ""),
                image_url=image_url,
                category=category_name,
                keywords=keywords_str,
            )

    except httpx.HTTPStatusError as e:
        logger.error("API request failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch recipe from OLJ API: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error fetching recipe: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching recipe: {e}",
        )


def recipe_data_to_canonical(recipe: RecipeData) -> dict:
    """Convert RecipeData to canonical format."""
    # Clean HTML from content
    instructions_text = strip_html_tags(recipe.content)
    ingredients_text = strip_html_tags(recipe.ingredients)

    # Parse keywords/tags
    tags = []
    if recipe.keywords:
        tags = [k.strip() for k in recipe.keywords.split(",") if k.strip()]

    # Detect if Lebanese
    is_lebanese = (
        any(
            kw in (recipe.keywords or "").lower()
            for kw in ["libanaise", "libanais", "liban", "lebanese"]
        )
        or "liban" in recipe.title.lower()
    )

    # Build search_text
    search_parts = [
        recipe.title,
        recipe.author or "",
        instructions_text,
        ingredients_text,
        " ".join(tags),
    ]
    search_text = " ".join(filter(None, search_parts))

    # Normalize category
    category = "plat_principal"
    if recipe.category:
        cat_lower = recipe.category.lower()
        if "dessert" in cat_lower:
            category = "dessert"
        elif "entrée" in cat_lower or "entree" in cat_lower:
            category = "entree"
        elif "mezzé" in cat_lower or "mezze" in cat_lower:
            category = "mezze_froid"
        elif "salade" in cat_lower:
            category = "salade"
        elif "soupe" in cat_lower:
            category = "soupe"

    return {
        "url": recipe.url,
        "title": recipe.title,
        "chef_name": recipe.author,
        "cuisine_type": "libanaise" if is_lebanese else "méditerranéenne",
        "is_lebanese": is_lebanese,
        "is_recipe": True,
        "category_canonical": category,
        "difficulty_canonical": "non_specifie",
        "tags": tags,
        "main_ingredients": [],  # Could extract from ingredients_text
        "aliases": [],
        "search_text": search_text,
        "source": "olj",
        "raw_category": recipe.category,
        "raw_difficulty": None,
        "raw_enrichment_present": False,
        "_image_url": recipe.image_url,
        "_publish_date": recipe.published_date,
        "_content_id": recipe.id,
    }


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("/recipe", response_model=WebhookResponse)
async def receive_recipe(request: Request):
    """
    Receive notification of new/updated recipe from CMS.

    Flow:
    1. Validates HMAC signature (WhiteBeard format: sha256=<hash>)
    2. Fetches full recipe data from OLJ API
    3. Enriches recipe via LLM (optional)
    4. Adds to canonical dataset
    5. Reloads retriever index
    6. Recipe is immediately searchable

    Returns:
        WebhookResponse with status and enrichment details

    Raises:
        401: Invalid webhook signature
        404: Recipe not found in OLJ API
        500: Enrichment or storage failed
    """
    settings = get_settings()

    # 1. Get raw body for signature verification
    body = await request.body()
    
    # DEBUG: Log raw body to understand WhiteBeard payload structure
    print(f"[WEBHOOK DEBUG] Raw body received: {body.decode('utf-8', 'ignore')[:2000]}")

    # 2. Extract signature from headers (case-insensitive lookup)
    # WhiteBeard sends: x-webhook-signature (lowercase)
    headers_lower = {k.lower(): v for k, v in request.headers.items()}
    x_webhook_signature = headers_lower.get("x-webhook-signature")

    # Log received headers for debugging
    logger.info("Webhook received - Headers: %s", dict(request.headers))
    logger.info("Webhook signature value: %s", x_webhook_signature)

    # 3. Validate HMAC signature (WhiteBeard format)
    expected_secret = settings.webhook_secret or os.getenv("WEBHOOK_SECRET", "")

    if not expected_secret:
        logger.warning(
            "WEBHOOK_SECRET not configured - accepting any request (DEV MODE)"
        )
    elif not verify_webhook_signature(body, x_webhook_signature or "", expected_secret):
        # Try to parse article_id for logging (best effort)
        try:
            payload_data = json.loads(body)
            article_id = payload_data.get("article_id", 0)
        except Exception:
            article_id = 0
        log_webhook_event(article_id, "auth_failed", {"reason": "invalid_signature"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )

    # 4. Parse payload after signature validation
    try:
        payload_data = json.loads(body)
    except Exception as e:
        logger.error("Failed to parse JSON payload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON: {e}",
        )

    # Log the complete payload for debugging (helps understand WhiteBeard format)
    logger.info("Webhook payload received: %s", json.dumps(payload_data, ensure_ascii=False))
    print(f"[WEBHOOK DEBUG] Full payload: {json.dumps(payload_data, ensure_ascii=False)}")
    
    # Log parameters specifically (this is where article_id should be)
    parameters_obj = payload_data.get("parameters", {})
    logger.info("Webhook parameters object: %s (type: %s)", parameters_obj, type(parameters_obj).__name__)
    print(f"[WEBHOOK DEBUG] Parameters: {parameters_obj}")

    # 5. Extract article_id from payload (handle both formats)
    # 
    # WhiteBeard webhook format (actual from OLJ logs):
    # {
    #   "automationId": "2",
    #   "queueId": "...",
    #   "timestamp": "2026-01-23 12:23:37",
    #   "subject": "",
    #   "user_id": "56",
    #   "article_id": "1488083",      <-- AT ROOT LEVEL (as string!)
    #   "parent_automation_id": "2",
    #   "trigger_id": "content_publish"
    # }
    # 
    # Note: article_id is a STRING, not int!
    # Reference: https://docs.whitebeard.net/marketingguides/setting_up_automations/#webhooks
    
    article_id = None
    action = "publish"

    # Helper to safely convert to int (handles strings like "1488083")
    def safe_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    # WhiteBeard confirmed format (from OLJ team):
    # {
    #   "automationId": 10,
    #   "queueId": 123,
    #   "condition": "condition_success",
    #   "parameters": {"user_id": "123", "article_id": "123"},  <-- HERE!
    #   "timestamp": "2026-01-01 00:00:00",
    #   "subject": ""
    # }
    
    def as_dict(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    # Collect possible containers where WhiteBeard may place fields
    containers = [payload_data]
    for key in ["parameters", "data", "payload", "record"]:
        candidate = as_dict(payload_data.get(key))
        if candidate:
            containers.append(candidate)

    # DEBUG: Log containers for debugging
    print(f"[WEBHOOK DEBUG] Containers built: {len(containers)} - Keys: {[list(c.keys()) if isinstance(c,dict) else str(type(c)) for c in containers]}")
    
    # Extract article_id from any container
    for container in containers:
        for field in ["article_id", "articleId", "content_id", "contentId", "id"]:
            if field in container:
                article_id = safe_int(container[field])
                print(f"[WEBHOOK DEBUG] Found {field}={container[field]} -> parsed as {article_id}")
                if article_id:
                    logger.info("Found article_id in %s: %s -> %s", field, container[field], article_id)
                    break
        if article_id:
            break

    # Fallback: regex scan raw payload (covers edge cases with stringified blobs)
    if not article_id:
        try:
            body_text = body.decode("utf-8", errors="ignore")
            match = re.search(r'\"article_id\"\\s*:\\s*\"?(\\d+)\"?', body_text)
            if match:
                article_id = safe_int(match.group(1))
                if article_id:
                    logger.info("Found article_id via regex scan: %s", article_id)
        except Exception:
            pass

    # Determine action from trigger_id (check containers first)
    trigger_id = ""
    for container in containers:
        trigger_id = container.get("trigger_id", "") or container.get("triggerId", "")
        if trigger_id:
            break
    if trigger_id:
        if "delete" in trigger_id or "unpublish" in trigger_id:
            action = "delete"
        elif "update" in trigger_id:
            action = "update"
        else:
            action = "publish"

    # Also check explicit "action" field (highest priority)
    for container in containers:
        if "action" in container:
            action = container["action"]
            break
    
    logger.info("Extracted: article_id=%s, action=%s, trigger_id=%s", article_id, action, trigger_id)

    if not article_id:
        # Log the payload so the team can identify the correct field
        logger.error(
            "Could not extract article_id from payload. "
            "Please check the payload structure and update the code. "
            "Received: %s", 
            json.dumps(payload_data, ensure_ascii=False)
        )
        print(f"[WEBHOOK ERROR] Could not extract article_id. Full payload: {json.dumps(payload_data, ensure_ascii=False)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Could not extract article_id from payload. "
                f"Received fields: {list(payload_data.keys())}. "
                f"Please configure the automation to include 'article_id' or 'contentId'."
            ),
        )

    # Create our internal payload object
    payload = WebhookPayload(article_id=article_id, action=action)

    log_webhook_event(payload.article_id, "received", {"action": payload.action})

    # Handle delete action
    if payload.action == "delete":
        canonical = load_canonical()
        url_to_delete = f"https://www.lorientlejour.com/article/{payload.article_id}"
        canonical = [r for r in canonical if r.get("url") != url_to_delete]
        save_canonical(canonical)

        try:
            bot = get_bot()
            bot.retriever.reload()
        except Exception as e:
            logger.error("Failed to reload retriever after delete: %s", e)

        log_webhook_event(payload.article_id, "deleted")
        return WebhookResponse(
            status="deleted",
            article_id=payload.article_id,
            message=f"Recipe {payload.article_id} deleted",
            enriched=False,
        )

    # 2. Check for duplicates
    canonical = load_canonical()
    url = f"https://www.lorientlejour.com/article/{payload.article_id}"
    existing_urls = {r.get("url") for r in canonical}

    if url in existing_urls and payload.action != "update":
        log_webhook_event(payload.article_id, "duplicate", {"url": url})
        return WebhookResponse(
            status="skipped",
            article_id=payload.article_id,
            message="Recipe already exists",
            enriched=False,
        )

    # 3. Fetch recipe from OLJ API
    recipe_data = await fetch_recipe_from_api(payload.article_id)

    if not recipe_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recipe {payload.article_id} not found in OLJ API",
        )

    log_webhook_event(payload.article_id, "fetched", {"title": recipe_data.title})

    # 4. Convert to canonical format
    canonical_data = recipe_data_to_canonical(recipe_data)

    # 5. Enrich via LLM (if enabled)
    if settings.auto_enrich_on_webhook:
        try:
            enricher = RecipeEnricher()
            enriched = await enricher.enrich(
                {
                    "title": recipe_data.title,
                    "content": recipe_data.content,
                    "author": recipe_data.author,
                    "url": recipe_data.url,
                }
            )
            enriched_canonical = enriched_to_canonical(enriched)
            # Merge enriched data
            canonical_data.update(enriched_canonical)
            canonical_data["raw_enrichment_present"] = True
            log_webhook_event(
                payload.article_id,
                "enriched",
                {
                    "category": canonical_data.get("category_canonical"),
                },
            )
        except Exception as e:
            logger.error("Enrichment failed for %s: %s", payload.article_id, e)
            # Continue with non-enriched data

    # 6. Ensure search_text is never empty (fallback to title)
    if not canonical_data.get("search_text") or len(canonical_data.get("search_text", "").strip()) == 0:
        canonical_data["search_text"] = canonical_data.get("title", "Recipe")
        logger.warning("Empty search_text for article %s, using title as fallback", payload.article_id)

    # 7. Validate with CanonicalRecipeDoc schema before saving
    try:
        validated_doc = CanonicalRecipeDoc(**canonical_data)
        # Convert back to dict for JSON serialization, using aliases
        canonical_data = validated_doc.model_dump(by_alias=True, exclude_none=False)
        log_webhook_event(payload.article_id, "validated", {"schema": "CanonicalRecipeDoc"})
    except Exception as e:
        logger.error("Schema validation failed for article %s: %s", payload.article_id, e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Recipe data validation failed: {str(e)}",
        )

    # 8. Add/update in canonical
    if payload.action == "update":
        # Remove old version first
        canonical = [r for r in canonical if r.get("url") != url]

    canonical.append(canonical_data)

    try:
        save_canonical(canonical)
        log_webhook_event(payload.article_id, "saved")
    except Exception as e:
        logger.error("Failed to save canonical: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save recipe: {str(e)}",
        )

    # 9. Reload retriever
    try:
        bot = get_bot()
        bot.retriever.reload()
        log_webhook_event(payload.article_id, "indexed")
    except Exception as e:
        logger.error("Failed to reload retriever: %s", e)
        # Recipe is saved but not indexed - will be indexed on next restart

    return WebhookResponse(
        status="indexed",
        article_id=payload.article_id,
        message=f"Recipe '{recipe_data.title}' added and indexed",
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
        "olj_api_configured": bool(settings.olj_api_key),
    }

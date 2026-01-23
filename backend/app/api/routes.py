"""
Sahten API Routes (MVP)
=======================

Dedicated routes for the durable RAG pipeline with flexible model selection.

Endpoints:
  - POST /chat         - Main chat endpoint (supports model override)
  - GET  /health       - Health check
  - GET  /status       - Detailed status
  - GET  /traces       - Team review of conversations
  - GET  /models       - Available models list
  - POST /events       - Track user events (impression, click, feedback)
  - GET  /analytics    - Aggregated metrics
  - GET  /evaluate     - Run evaluation suite (golden dataset)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel

from ..bot import get_bot
from ..core.config import get_settings, get_available_models
from ..schemas.responses import SahtenResponse
from .response_composer import compose_html_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ============================================================================
# UPSTASH REDIS CONFIGURATION
# ============================================================================

_redis_client = None


def get_redis():
    """Lazy initialization of Upstash Redis client."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    
    if url and token:
        try:
            from upstash_redis import Redis
            _redis_client = Redis(url=url, token=token)
            logger.info("Upstash Redis connected successfully")
        except ImportError:
            logger.warning("upstash-redis not installed, traces will be stdout only")
            _redis_client = False
        except Exception as e:
            logger.warning("Failed to connect to Upstash Redis: %s", e)
            _redis_client = False
    else:
        logger.info("Upstash not configured, using stdout logging")
        _redis_client = False
    
    return _redis_client if _redis_client else None


def log_chat_trace(
    *,
    request_id: str,
    user_message: str,
    response_html: str,
    response: SahtenResponse,
    debug: bool,
    debug_info: Optional[dict],
    model_used: Optional[str] = None,
    is_base2_fallback: bool = False,
) -> None:
    """Log conversation trace to stdout and optionally Upstash Redis."""
    try:
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "user_message": user_message,
            "response_html": response_html,
            "response_type": response.response_type,
            "intent": response.intent_detected,
            "confidence": response.confidence,
            "recipe_count": response.recipe_count,
            "model_used": model_used or response.model_used,
            "is_base2_fallback": is_base2_fallback,
            "metadata": {
                "source": "api",
                "debug": debug,
            },
        }
        
        # Always log to stdout
        compact_trace = {
            "ts": trace["timestamp"],
            "id": request_id,
            "q": user_message[:100],
            "intent": response.intent_detected,
            "recipes": response.recipe_count,
            "model": model_used or response.model_used,
            "base2": is_base2_fallback,
        }
        print(f"[TRACE] {json.dumps(compact_trace, ensure_ascii=False)}")
        
        # Persist to Upstash if available
        redis = get_redis()
        if redis:
            try:
                key = f"trace:{request_id}"
                redis.set(key, json.dumps(trace, ensure_ascii=False), ex=60*60*24*30)
                redis.lpush("sahten:traces:recent", json.dumps(trace, ensure_ascii=False))
                redis.ltrim("sahten:traces:recent", 0, 499)
                
                # Track metrics
                redis.incr("sahten:metrics:total_requests")
                
                # Track fallback rate
                if is_base2_fallback:
                    redis.incr("sahten:metrics:base2_fallback_count")
                
                # Track by model
                if model_used or response.model_used:
                    model_key = (model_used or response.model_used).replace(".", "_")
                    redis.incr(f"sahten:metrics:model:{model_key}:count")
                
                # Track by intent
                if response.intent_detected:
                    redis.incr(f"sahten:metrics:intent:{response.intent_detected}:count")
                
                logger.debug("Trace %s saved to Redis", request_id)
            except Exception as e:
                logger.warning("Redis write failed: %s", e)
                
    except Exception as e:
        logger.warning("Failed to log trace: %s", e)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request with optional model override and session tracking."""
    message: str
    debug: bool = False
    model: Optional[str] = None  # "auto", "gpt-4.1-nano", "gpt-4o-mini"
    session_id: Optional[str] = None  # For conversation memory


class ChatResponseAPI(BaseModel):
    """Chat response with model info."""
    html: str
    response_type: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    recipe_count: int = 0
    model_used: Optional[str] = None
    request_id: Optional[str] = None  # For feedback reference
    debug_info: Optional[dict] = None


class ModelsResponse(BaseModel):
    """Available models response."""
    models: List[str]
    default: str
    ab_testing_enabled: bool


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/chat", response_model=ChatResponseAPI)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Supports model override:
    - model="auto" or None: Use default or A/B testing
    - model="gpt-4.1-nano": Force nano model (economique)
    - model="gpt-4o-mini": Force mini model (qualite)
    
    Supports session memory:
    - session_id: Optional client-generated ID for conversation continuity
    - Avoids re-proposing same recipes within a session
    """
    request_id = str(uuid.uuid4())[:8]
    
    # Use provided session_id or generate one
    session_id = request.session_id or request_id
    
    try:
        bot = get_bot()
        response, debug_info = await bot.chat(
            request.message,
            debug=request.debug,
            model=request.model,
            request_id=request_id,
            session_id=session_id,
        )
        html = compose_html_response(response)

        log_chat_trace(
            request_id=request_id,
            user_message=request.message,
            response_html=html,
            response=response,
            debug=request.debug,
            debug_info=debug_info,
            model_used=response.model_used,
            is_base2_fallback=(response.response_type == "recipe_base2"),
        )

        return ChatResponseAPI(
            html=html,
            response_type=response.response_type,
            intent=response.intent_detected,
            confidence=response.confidence,
            recipe_count=response.recipe_count,
            model_used=response.model_used,
            request_id=request_id,  # For feedback reference
            debug_info=debug_info if request.debug else None,
        )
    except Exception as e:
        logger.error("[%s] chat error: %s", request_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur est survenue. Réessaie !",
        )


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get available models for UI dropdown."""
    settings = get_settings()
    return ModelsResponse(
        models=get_available_models(),
        default=settings.openai_model,
        ab_testing_enabled=settings.enable_ab_testing,
    )


@router.get("/health")
async def health():
    """Health check endpoint."""
    redis = get_redis()
    settings = get_settings()
    return {
        "status": "healthy",
        "version": settings.app_version,
        "pipeline": "durable-rag-mvp",
        "logging": "upstash" if redis else "stdout",
    }


@router.get("/status")
async def get_status():
    """Detailed status with component info."""
    try:
        bot = get_bot()
        redis = get_redis()
        settings = get_settings()
        
        return {
            "status": "operational",
            "version": settings.app_version,
            "model": {
                "default": settings.openai_model,
                "ab_testing": settings.enable_ab_testing,
                "available": get_available_models(),
            },
            "embeddings": {
                "enabled": settings.enable_embeddings,
                "provider": settings.embedding_provider if settings.enable_embeddings else None,
            },
            "logging": {
                "backend": "upstash" if redis else "stdout",
                "traces_endpoint": "/api/traces" if redis else None,
            },
            "components": {
                "query_analyzer": "ready",
                "retriever": "ready",
                "response_generator": "ready",
            },
            "stats": {
                "olj_docs": len(bot.retriever.olj_docs),
                "base2_categories": len(bot.retriever.base2_recipes),
                "base2_total": sum(len(v) for v in bot.retriever.base2_recipes.values()),
            },
        }
    except Exception as e:
        return {"status": "initializing", "error": str(e)}


# ============================================================================
# TRACES ENDPOINT
# ============================================================================

class TraceItem(BaseModel):
    timestamp: str
    request_id: str
    user_message: str
    response_type: str
    intent: Optional[str]
    confidence: Optional[float]
    recipe_count: int
    model_used: Optional[str]


class TracesResponse(BaseModel):
    count: int
    logging_backend: str
    traces: List[dict]


@router.get("/traces", response_model=TracesResponse)
async def get_traces(
    limit: int = Query(default=50, ge=1, le=200, description="Number of traces to retrieve")
):
    """Retrieve recent conversation traces for team review."""
    redis = get_redis()
    
    if not redis:
        return TracesResponse(
            count=0,
            logging_backend="stdout",
            traces=[],
        )
    
    try:
        raw_traces = redis.lrange("sahten:traces:recent", 0, limit - 1)
        
        traces = []
        for raw in raw_traces:
            try:
                trace = json.loads(raw) if isinstance(raw, str) else raw
                traces.append({
                    "timestamp": trace.get("timestamp"),
                    "request_id": trace.get("request_id"),
                    "user_message": trace.get("user_message"),
                    "response_type": trace.get("response_type"),
                    "intent": trace.get("intent"),
                    "confidence": trace.get("confidence"),
                    "recipe_count": trace.get("recipe_count"),
                    "model_used": trace.get("model_used"),
                })
            except (json.JSONDecodeError, TypeError):
                continue
        
        return TracesResponse(
            count=len(traces),
            logging_backend="upstash",
            traces=traces,
        )
        
    except Exception as e:
        logger.error("Failed to retrieve traces: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve traces: {str(e)}",
        )


# ============================================================================
# EVENTS ENDPOINT - Unified event tracking (impression, click, feedback)
# ============================================================================

class EventRequest(BaseModel):
    """
    Unified event schema for analytics.
    
    Event types:
    - impression: Recipe card was shown to user
    - click: User clicked on a recipe link
    - feedback: User gave thumbs up/down
    """
    event_type: str  # "impression", "click", "feedback"
    request_id: str
    session_id: Optional[str] = None
    
    # For click events
    recipe_url: Optional[str] = None
    recipe_title: Optional[str] = None
    
    # For feedback events
    rating: Optional[str] = None  # "positive" or "negative"
    reason: Optional[str] = None
    
    # Optional context
    intent: Optional[str] = None
    model_used: Optional[str] = None


class EventResponse(BaseModel):
    """Event confirmation."""
    status: str
    event_type: str
    request_id: str


@router.post("/events", response_model=EventResponse)
async def track_event(request: EventRequest):
    """
    Track user events for analytics.
    
    Supports:
    - impression: When recipe cards are displayed
    - click: When user clicks a recipe link
    - feedback: When user gives thumbs up/down
    """
    try:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": request.event_type,
            "request_id": request.request_id,
            "session_id": request.session_id,
            "recipe_url": request.recipe_url,
            "recipe_title": request.recipe_title,
            "rating": request.rating,
            "reason": request.reason,
            "intent": request.intent,
            "model_used": request.model_used,
        }
        
        # Log to stdout
        print(f"[EVENT:{request.event_type.upper()}] {json.dumps(event, ensure_ascii=False)}")
        
        # Persist to Redis if available
        redis = get_redis()
        if redis:
            try:
                # Store in event stream
                redis.lpush(f"sahten:events:{request.event_type}", json.dumps(event, ensure_ascii=False))
                redis.ltrim(f"sahten:events:{request.event_type}", 0, 4999)  # Keep last 5000 per type
                
                # Increment counters
                redis.incr(f"sahten:events:{request.event_type}:count")
                
                # For impressions, track per-recipe counts
                if request.event_type == "impression" and request.recipe_url:
                    redis.hincrby("sahten:recipe:impressions", request.recipe_url, 1)
                
                # For clicks, track CTR data
                if request.event_type == "click" and request.recipe_url:
                    redis.hincrby("sahten:recipe:clicks", request.recipe_url, 1)
                
                # For feedback, maintain counters
                if request.event_type == "feedback" and request.rating:
                    if request.rating == "positive":
                        redis.incr("sahten:feedback:positive_count")
                    else:
                        redis.incr("sahten:feedback:negative_count")
                        
                        # Store negative reasons for review
                        if request.reason:
                            redis.lpush("sahten:feedback:negative_reasons", json.dumps({
                                "reason": request.reason,
                                "request_id": request.request_id,
                                "timestamp": event["timestamp"],
                            }, ensure_ascii=False))
                            redis.ltrim("sahten:feedback:negative_reasons", 0, 199)
                
            except Exception as e:
                logger.warning("Failed to persist event to Redis: %s", e)
        
        return EventResponse(
            status="received",
            event_type=request.event_type,
            request_id=request.request_id,
        )
        
    except Exception as e:
        logger.error("Failed to track event: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track event",
        )


# Legacy feedback endpoint (redirects to events)
class FeedbackRequest(BaseModel):
    """User feedback on a response."""
    request_id: str
    rating: str  # "positive" or "negative"
    reason: Optional[str] = None  # Optional text reason
    session_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Feedback confirmation."""
    status: str
    request_id: str


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on a response (legacy endpoint).
    Internally uses the unified events system.
    """
    # Convert to event format
    event_req = EventRequest(
        event_type="feedback",
        request_id=request.request_id,
        session_id=request.session_id,
        rating=request.rating,
        reason=request.reason,
    )
    
    await track_event(event_req)
    return FeedbackResponse(status="received", request_id=request.request_id)


@router.get("/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics for analytics."""
    redis = get_redis()
    
    if not redis:
        return {
            "status": "no_redis",
            "message": "Feedback stats require Redis connection",
        }
    
    try:
        positive = int(redis.get("sahten:feedback:positive_count") or 0)
        negative = int(redis.get("sahten:feedback:negative_count") or 0)
        total = positive + negative
        
        # Get recent feedback reasons (for negative)
        recent_raw = redis.lrange("sahten:feedback:negative_reasons", 0, 9)
        recent_negative_reasons = []
        
        for raw in recent_raw:
            try:
                fb = json.loads(raw) if isinstance(raw, str) else raw
                recent_negative_reasons.append(fb)
            except:
                continue
        
        return {
            "status": "ok",
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": round(positive / total * 100, 1) if total > 0 else 0,
            "recent_negative_reasons": recent_negative_reasons,
        }
        
    except Exception as e:
        logger.error("Failed to get feedback stats: %s", e)
        return {"status": "error", "message": str(e)}


# ============================================================================
# ANALYTICS ENDPOINT - Aggregated metrics
# ============================================================================

@router.get("/analytics")
async def get_analytics():
    """
    Get aggregated analytics metrics.
    
    Includes:
    - Total conversations
    - Event counts (impressions, clicks, feedback)
    - CTR (Click-Through Rate)
    - Satisfaction rate
    - Fallback rate (Base2 vs OLJ)
    - Model distribution
    - Intent distribution
    - Top recipes by clicks
    """
    redis = get_redis()
    
    if not redis:
        return {
            "status": "no_redis",
            "message": "Analytics require Redis connection",
        }
    
    try:
        # Get event counts
        impressions = int(redis.get("sahten:events:impression:count") or 0)
        clicks = int(redis.get("sahten:events:click:count") or 0)
        feedback_count = int(redis.get("sahten:events:feedback:count") or 0)
        
        # Get feedback breakdown
        positive = int(redis.get("sahten:feedback:positive_count") or 0)
        negative = int(redis.get("sahten:feedback:negative_count") or 0)
        
        # Calculate CTR
        ctr = round(clicks / impressions * 100, 2) if impressions > 0 else 0
        
        # Calculate satisfaction rate
        total_feedback = positive + negative
        satisfaction_rate = round(positive / total_feedback * 100, 1) if total_feedback > 0 else 0
        
        # Get conversation count and fallback rate
        total_requests = int(redis.get("sahten:metrics:total_requests") or 0)
        base2_fallbacks = int(redis.get("sahten:metrics:base2_fallback_count") or 0)
        fallback_rate = round(base2_fallbacks / total_requests * 100, 1) if total_requests > 0 else 0
        
        trace_count = redis.llen("sahten:traces:recent")
        
        # Get model distribution
        model_stats = {}
        for model_key in ["gpt-4_1-nano", "gpt-4o-mini"]:
            count = int(redis.get(f"sahten:metrics:model:{model_key}:count") or 0)
            if count > 0:
                model_stats[model_key.replace("_", ".")] = count
        
        # Get intent distribution (top 5)
        intent_keys = [
            "recipe_specific", "recipe_by_ingredient", "recipe_by_mood",
            "menu_composition", "multi_recipe", "greeting", "clarification",
            "off_topic", "redirect"
        ]
        intent_stats = {}
        for intent in intent_keys:
            count = int(redis.get(f"sahten:metrics:intent:{intent}:count") or 0)
            if count > 0:
                intent_stats[intent] = count
        
        # Get top recipes by clicks (top 10)
        top_clicks_raw = redis.hgetall("sahten:recipe:clicks") or {}
        top_recipes = sorted(
            [{"url": k, "clicks": int(v)} for k, v in top_clicks_raw.items()],
            key=lambda x: -x["clicks"]
        )[:10]
        
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conversations": {
                "total": total_requests or trace_count,
                "recent_traces": trace_count,
            },
            "events": {
                "impressions": impressions,
                "clicks": clicks,
                "feedback": feedback_count,
            },
            "rates": {
                "ctr": ctr,
                "satisfaction": satisfaction_rate,
                "fallback_rate": fallback_rate,
            },
            "fallback": {
                "total_requests": total_requests,
                "base2_count": base2_fallbacks,
            },
            "feedback": {
                "positive": positive,
                "negative": negative,
                "total": total_feedback,
            },
            "models": model_stats,
            "intents": intent_stats,
            "top_recipes": top_recipes,
        }
        
    except Exception as e:
        logger.error("Failed to get analytics: %s", e)
        return {"status": "error", "message": str(e)}


# ============================================================================
# EVALUATION ENDPOINT - Run golden dataset tests
# ============================================================================

@router.get("/evaluate")
async def run_evaluation(limit: int = Query(default=5, ge=1, le=20)):
    """
    Run evaluation on a subset of the golden dataset.
    
    This is a lightweight evaluation that runs a few test cases
    to validate the pipeline is working correctly.
    
    For full evaluation, use: python scripts/evaluate.py
    """
    import json as json_mod
    from pathlib import Path
    
    try:
        bot = get_bot()
        
        # Load golden dataset
        tests_dir = Path(__file__).parent.parent.parent / "tests"
        golden_path = tests_dir / "golden_dataset.json"
        
        if not golden_path.exists():
            # Fallback to test_matrix
            golden_path = tests_dir / "test_matrix.json"
        
        if not golden_path.exists():
            return {"status": "error", "message": "No test dataset found"}
        
        dataset = json_mod.loads(golden_path.read_text(encoding="utf-8"))
        cases = dataset.get("cases", [])[:limit]
        
        results = []
        passed = 0
        
        for case in cases:
            try:
                response, _ = await bot.chat(case["query"], debug=False)
                
                # Check basic constraints
                constraints = case.get("constraints", {})
                case_passed = True
                
                # Response type check
                if "response_type" in constraints:
                    if response.response_type != constraints["response_type"]:
                        case_passed = False
                
                # Min recipes check
                if "min_recipes" in constraints:
                    if response.recipe_count < int(constraints["min_recipes"]):
                        case_passed = False
                
                if case_passed:
                    passed += 1
                
                results.append({
                    "case_id": case["id"],
                    "query": case["query"][:50],
                    "passed": case_passed,
                    "response_type": response.response_type,
                    "recipe_count": response.recipe_count,
                })
                
            except Exception as e:
                results.append({
                    "case_id": case["id"],
                    "query": case["query"][:50],
                    "passed": False,
                    "error": str(e)[:100],
                })
        
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results) * 100, 1) if results else 0,
            "results": results,
        }
        
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        return {"status": "error", "message": str(e)}

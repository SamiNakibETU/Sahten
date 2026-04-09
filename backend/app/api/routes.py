"""
Sahten API Routes (MVP)
=======================

Dedicated routes for the durable RAG pipeline with flexible model selection.

Endpoints:
  - POST /chat         - Main chat endpoint (supports model override)
  - POST /chat/stream  - SSE stream (blocks + final HTML payload)
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
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request, status, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..bot import get_bot
from ..core.config import get_settings, get_available_models
from ..core.metrics import record_metrics
from ..core.redis_client import get_redis, invalidate_redis_client, redis_connection_error
from ..schemas.responses import SahtenResponse
from .response_composer import compose_html_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# In-memory rate limiter (sliding window per IP)
# ---------------------------------------------------------------------------
_rate_windows: dict[str, deque] = defaultdict(deque)

def _check_rate_limit(client_ip: str, max_requests: int = 30, window_secs: int = 60) -> None:
    """Raise HTTP 429 if client_ip exceeds max_requests in window_secs."""
    now = time.monotonic()
    window = _rate_windows[client_ip]
    # Evict old timestamps
    while window and window[0] < now - window_secs:
        window.popleft()
    if len(window) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Trop de requêtes. Veuillez réessayer dans une minute.",
            headers={"Retry-After": str(window_secs)},
        )
    window.append(now)


def _get_client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Forwarded-For from Railway proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


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
    trace_meta: Optional[dict] = None,
    latency_ms: Optional[int] = None,
    user_agent: Optional[str] = None,
    session_turn_count: Optional[int] = None,
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
            "latency_ms": latency_ms,
            "session_turn_count": session_turn_count,
            "metadata": {
                "source": "api",
                "debug": debug,
                "user_agent_short": (user_agent or "")[:80] if user_agent else None,
            },
        }
        if trace_meta:
            trace["trace_meta"] = trace_meta
        
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
                if redis_connection_error(e):
                    invalidate_redis_client()

    except Exception as e:
        logger.warning("Failed to log trace: %s", e)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request with optional model override and session tracking."""
    message: str
    debug: bool = False
    model: Optional[str] = None  # "auto", OpenAI (gpt-4.1-*), Groq (llama-3.*, openai/gpt-oss-*)
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
    scenario_id: Optional[int] = None
    scenario_name: Optional[str] = None
    primary_url: Optional[str] = None
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
async def chat(request: ChatRequest, http_request: Request):
    """
    Main chat endpoint.
    
    Supports model override:
    - model="auto" or None: default / A/B testing
    - OpenAI: gpt-4.1-nano, gpt-4.1-mini
    - Groq (GROQ_API_KEY): llama-3.1-8b-instant, llama-3.3-70b-versatile, openai/gpt-oss-20b
    
    Avec debug=true, debug_info inclut timings_ms (analysis, retrieval, generation, total).
    
    Supports session memory:
    - session_id: Optional client-generated ID for conversation continuity
    - Avoids re-proposing same recipes within a session
    """
    # Rate limit: 30 req/min per IP
    _check_rate_limit(_get_client_ip(http_request), max_requests=30, window_secs=60)

    request_id = str(uuid.uuid4())[:8]
    _t_start = time.monotonic()
    
    # Use provided session_id or generate one
    session_id = request.session_id or request_id
    user_agent = http_request.headers.get("user-agent")
    
    try:
        bot = get_bot()
        response, debug_info, trace_meta = await bot.chat(
            request.message,
            debug=request.debug,
            model=request.model,
            request_id=request_id,
            session_id=session_id,
        )
        html = compose_html_response(response)
        latency_ms = int((time.monotonic() - _t_start) * 1000)

        if request.debug and debug_info is not None:
            tm = trace_meta.get("timings_ms")
            if tm:
                debug_info = {**debug_info, "timings_ms": tm}

        # Extract session turn count from trace_meta if available
        _turn_count = (trace_meta or {}).get("session_turn_count")

        log_chat_trace(
            request_id=request_id,
            user_message=request.message,
            response_html=html,
            response=response,
            debug=request.debug,
            debug_info=debug_info,
            model_used=response.model_used,
            is_base2_fallback=(response.response_type == "recipe_base2"),
            trace_meta=trace_meta,
            latency_ms=latency_ms,
            user_agent=user_agent,
            session_turn_count=_turn_count,
        )
        record_metrics(response_type=response.response_type, trace_meta=trace_meta)

        api_scenario = trace_meta.get("api_scenario") or {}
        primary_url = api_scenario.get("primary_url")
        if not primary_url and response.recipes:
            primary_url = str(response.recipes[0].url)

        return ChatResponseAPI(
            html=html,
            response_type=response.response_type,
            intent=response.intent_detected,
            confidence=response.confidence,
            recipe_count=response.recipe_count,
            model_used=response.model_used,
            request_id=request_id,  # For feedback reference
            scenario_id=api_scenario.get("scenario_id"),
            scenario_name=api_scenario.get("scenario_name"),
            primary_url=primary_url,
            debug_info=debug_info if request.debug else None,
        )
    except Exception as e:
        logger.error("[%s] chat error: %s", request_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur est survenue. Réessaie !",
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE stream: status, optional block events, then final payload with full HTML (same as /chat).
    """
    request_id = str(uuid.uuid4())[:8]
    session_id = request.session_id or request_id

    async def event_generator():
        yield f"data: {json.dumps({'type': 'status', 'message': 'typing'})}\n\n"
        try:
            bot = get_bot()
            response, debug_info, trace_meta = await bot.chat(
                request.message,
                debug=request.debug,
                model=request.model,
                request_id=request_id,
                session_id=session_id,
            )
            html = compose_html_response(response)
            for block in response.conversation_blocks or []:
                yield f"data: {json.dumps({'type': 'block', 'block': block.model_dump()}, ensure_ascii=False)}\n\n"
            done_payload = {
                "type": "done",
                "html": html,
                "response_type": response.response_type,
                "intent": response.intent_detected,
                "confidence": response.confidence,
                "recipe_count": response.recipe_count,
                "model_used": response.model_used,
                "request_id": request_id,
                "trace_meta": trace_meta,
            }
            if request.debug and debug_info is not None:
                di = dict(debug_info)
                tm = trace_meta.get("timings_ms")
                if tm:
                    di["timings_ms"] = tm
                done_payload["debug_info"] = di
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

            log_chat_trace(
                request_id=request_id,
                user_message=request.message,
                response_html=html,
                response=response,
                debug=request.debug,
                debug_info=debug_info,
                model_used=response.model_used,
                is_base2_fallback=(response.response_type == "recipe_base2"),
                trace_meta=trace_meta,
            )
            record_metrics(response_type=response.response_type, trace_meta=trace_meta)
        except Exception as e:
            logger.error("[%s] chat_stream error: %s", request_id, e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Une erreur est survenue.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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


@router.get("/health/deep")
async def health_deep():
    """Deep health check: verifies all critical subsystems."""
    import time as _time
    import httpx
    checks: dict = {}
    overall_ok = True

    # 1. Bot / retriever
    try:
        bot = get_bot()
        doc_count = len(getattr(bot.retriever, '_docs', None) or [])
        checks["retriever"] = {"status": "ok", "doc_count": doc_count}
    except Exception as e:
        checks["retriever"] = {"status": "error", "detail": str(e)[:120]}
        overall_ok = False

    # 2. Redis connectivity (synchronous Upstash client — no await)
    redis = get_redis()
    if redis:
        try:
            t0 = _time.monotonic()
            result = redis.ping()  # synchronous call, returns "PONG" or True
            ok = result in (True, "PONG", b"PONG")
            latency = int((_time.monotonic() - t0) * 1000)
            checks["redis"] = {"status": "ok" if ok else "error", "latency_ms": latency}
        except Exception as e:
            checks["redis"] = {"status": "error", "detail": str(e)[:80]}
    else:
        checks["redis"] = {"status": "disabled"}

    # 3. OpenAI connectivity (lightweight models list call)
    settings = get_settings()
    openai_key = getattr(settings, "openai_api_key", "")
    if openai_key:
        try:
            t0 = _time.monotonic()
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {openai_key}"},
                )
            latency = int((_time.monotonic() - t0) * 1000)
            checks["openai"] = {"status": "ok" if r.status_code == 200 else "error", "latency_ms": latency, "http": r.status_code}
        except Exception as e:
            checks["openai"] = {"status": "error", "detail": str(e)[:80]}
    else:
        checks["openai"] = {"status": "no_key"}

    return {
        "status": "healthy" if overall_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/recipes")
async def get_recipes():
    """
    Get all canonical recipes for admin panel.
    
    Returns the full list of recipes from olj_canonical.json.
    """
    try:
        bot = get_bot()
        recipes = []
        
        for doc in bot.retriever.olj_docs:
            recipes.append({
                "url": str(doc.url),
                "title": doc.title,
                "chef_name": doc.chef_name,
                "cuisine_type": doc.cuisine_type,
                "is_lebanese": doc.is_lebanese,
                "category_canonical": doc.category_canonical,
                "difficulty_canonical": doc.difficulty_canonical,
                "tags": doc.tags,
                "main_ingredients": doc.main_ingredients,
                "_image_url": doc.image_url,
                "search_text": doc.search_text[:200] if doc.search_text else "",
            })
        
        return {
            "status": "ok",
            "count": len(recipes),
            "recipes": recipes,
        }
        
    except Exception as e:
        logger.error("Failed to get recipes: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load recipes: {str(e)}",
        )


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
        if redis_connection_error(e):
            invalidate_redis_client()
        return TracesResponse(
            count=0,
            logging_backend="unavailable",
            traces=[],
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
                if redis_connection_error(e):
                    invalidate_redis_client()

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
        if redis_connection_error(e):
            invalidate_redis_client()
        return {
            "status": "redis_unreachable",
            "message": str(e),
            "positive": 0,
            "negative": 0,
            "positive_rate": 0,
            "recent_negative_reasons": [],
        }


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
        for model_key in ["gpt-4_1-nano", "gpt-4_1-mini"]:
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

        # Quality metrics (Sahten zéro gap)
        exact_match_count = int(redis.get("sahten:metrics:exact_match_count") or 0)
        proven_alternative_count = int(redis.get("sahten:metrics:proven_alternative_count") or 0)
        recipe_not_found_count = int(redis.get("sahten:metrics:recipe_not_found_count") or 0)
        safety_block_count = int(redis.get("sahten:metrics:safety_block_count") or 0)
        response_type_keys = ["recipe_olj", "recipe_base2", "menu", "greeting", "redirect", "recipe_not_found", "clarification", "not_found_with_alternative"]
        response_type_stats = {}
        for rt in response_type_keys:
            count = int(redis.get(f"sahten:metrics:response_type:{rt}:count") or 0)
            if count > 0:
                response_type_stats[rt] = count
        routing_stats = {}
        for src in ["deterministic", "llm"]:
            count = int(redis.get(f"sahten:metrics:routing:{src}:count") or 0)
            if count > 0:
                routing_stats[src] = count
        
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
            "quality": {
                "exact_match_count": exact_match_count,
                "proven_alternative_count": proven_alternative_count,
                "recipe_not_found_count": recipe_not_found_count,
                "safety_block_count": safety_block_count,
            },
            "response_types": response_type_stats,
            "routing": routing_stats,
        }
        
    except Exception as e:
        logger.error("Failed to get analytics: %s", e)
        if redis_connection_error(e):
            invalidate_redis_client()
        return {
            "status": "redis_unreachable",
            "message": str(e),
        }


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
                response, _, _ = await bot.chat(case["query"], debug=False)
                
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

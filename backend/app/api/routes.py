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
                logger.debug("Trace %s saved to Redis", request_id)
            except Exception as e:
                logger.warning("Redis write failed: %s", e)
                
    except Exception as e:
        logger.warning("Failed to log trace: %s", e)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request with optional model override."""
    message: str
    debug: bool = False
    model: Optional[str] = None  # "auto", "gpt-4.1-nano", "gpt-4o-mini"


class ChatResponseAPI(BaseModel):
    """Chat response with model info."""
    html: str
    response_type: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    recipe_count: int = 0
    model_used: Optional[str] = None
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
    """
    request_id = str(uuid.uuid4())[:8]
    
    try:
        bot = get_bot()
        response, debug_info = await bot.chat(
            request.message,
            debug=request.debug,
            model=request.model,
            request_id=request_id,
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
        )

        return ChatResponseAPI(
            html=html,
            response_type=response.response_type,
            intent=response.intent_detected,
            confidence=response.confidence,
            recipe_count=response.recipe_count,
            model_used=response.model_used,
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

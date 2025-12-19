"""
Sahten API Routes
=================

Dedicated routes for the durable RAG pipeline.
Production version with Upstash Redis logging for persistent traces.

Endpoints:
  - POST /chat
  - GET  /health
  - GET  /status
  - GET  /traces  (team review - see recent conversations)
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
from ..schemas.responses import SahtenResponse
from .response_composer import compose_html_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ============================================================================
# UPSTASH REDIS CONFIGURATION
# ============================================================================
# Set these environment variables in Vercel:
#   - UPSTASH_REDIS_REST_URL
#   - UPSTASH_REDIS_REST_TOKEN
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
            _redis_client = False  # Mark as unavailable
        except Exception as e:
            logger.warning("Failed to connect to Upstash Redis: %s", e)
            _redis_client = False
    else:
        logger.info("Upstash not configured (no UPSTASH_REDIS_REST_URL), using stdout logging")
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
) -> None:
    """
    Log conversation trace.
    
    - Always logs to stdout (captured by Vercel Logs)
    - If Upstash is configured, also persists to Redis for team review
    """
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
            "has_olj_reco": response.olj_recommendation is not None,
            "metadata": {
                "source": "api",
                "debug": debug,
            },
        }
        
        # Always log to stdout (Vercel captures this)
        compact_trace = {
            "ts": trace["timestamp"],
            "id": request_id,
            "q": user_message[:100],  # Truncate for readability
            "intent": response.intent_detected,
            "recipes": response.recipe_count,
        }
        print(f"[TRACE] {json.dumps(compact_trace, ensure_ascii=False)}")
        
        # Persist to Upstash if available
        redis = get_redis()
        if redis:
            try:
                # Store full trace with 30-day expiry
                key = f"trace:{request_id}"
                redis.set(key, json.dumps(trace, ensure_ascii=False), ex=60*60*24*30)
                
                # Also push to a list for easy retrieval (keep last 500)
                redis.lpush("sahten:traces:recent", json.dumps(trace, ensure_ascii=False))
                redis.ltrim("sahten:traces:recent", 0, 499)
                
                logger.debug("Trace %s saved to Redis", request_id)
            except Exception as e:
                logger.warning("Redis write failed (trace still in stdout): %s", e)
                
    except Exception as e:
        logger.warning("Failed to log trace: %s", e)


class ChatRequest(BaseModel):
    message: str
    debug: bool = False


class ChatResponseAPI(BaseModel):
    html: str
    response_type: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    recipe_count: int = 0
    debug_info: Optional[dict] = None


@router.post("/chat", response_model=ChatResponseAPI)
async def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())[:8]
    try:
        bot = get_bot()
        response, debug_info = await bot.chat(request.message, debug=request.debug)
        html = compose_html_response(response)

        log_chat_trace(
            request_id=request_id,
            user_message=request.message,
            response_html=html,
            response=response,
            debug=request.debug,
            debug_info=debug_info,
        )

        return ChatResponseAPI(
            html=html,
            response_type=response.response_type,
            intent=response.intent_detected,
            confidence=response.confidence,
            recipe_count=response.recipe_count,
            debug_info=debug_info if request.debug else None,
        )
    except Exception as e:
        logger.error("[%s] chat error: %s", request_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur est survenue. Réessaie !",
        )


@router.get("/health")
async def health():
    """Health check endpoint."""
    redis = get_redis()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "pipeline": "durable-rag",
        "logging": "upstash" if redis else "stdout",
    }


@router.get("/status")
async def get_status():
    """Detailed status with component info."""
    try:
        bot = get_bot()
        redis = get_redis()
        return {
            "status": "operational",
            "version": "1.0.0",
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
# TRACES ENDPOINT - For team review
# ============================================================================

class TraceItem(BaseModel):
    timestamp: str
    request_id: str
    user_message: str
    response_type: str
    intent: Optional[str]
    confidence: Optional[float]
    recipe_count: int


class TracesResponse(BaseModel):
    count: int
    logging_backend: str
    traces: List[dict]


@router.get("/traces", response_model=TracesResponse)
async def get_traces(
    limit: int = Query(default=50, ge=1, le=200, description="Number of traces to retrieve")
):
    """
    Retrieve recent conversation traces for team review.
    
    Requires Upstash Redis to be configured.
    Returns the most recent conversations with user questions and bot responses.
    """
    redis = get_redis()
    
    if not redis:
        return TracesResponse(
            count=0,
            logging_backend="stdout",
            traces=[],
        )
    
    try:
        # Get recent traces from Redis list
        raw_traces = redis.lrange("sahten:traces:recent", 0, limit - 1)
        
        traces = []
        for raw in raw_traces:
            try:
                trace = json.loads(raw) if isinstance(raw, str) else raw
                # Return a clean subset for review
                traces.append({
                    "timestamp": trace.get("timestamp"),
                    "request_id": trace.get("request_id"),
                    "user_message": trace.get("user_message"),
                    "response_type": trace.get("response_type"),
                    "intent": trace.get("intent"),
                    "confidence": trace.get("confidence"),
                    "recipe_count": trace.get("recipe_count"),
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

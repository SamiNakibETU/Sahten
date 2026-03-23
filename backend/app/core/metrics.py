"""
Sahten Business Metrics
======================

Tracks key product and quality metrics for observability:
- exact_match_rate: Recipe found in OLJ canonical index
- proven_alternative_rate: Alternative with SharedIngredientProof
- recipe_not_found_rate: No recipe, no proof
- safety_block_count: Blocked by SafetyGate
- response_type counters for dashboards
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_redis():
    """Lazy Redis client for metrics."""
    import os
    try:
        url = os.getenv("UPSTASH_REDIS_REST_URL")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        if url and token:
            from upstash_redis import Redis
            return Redis(url=url, token=token)
    except Exception:
        pass
    return None


def record_metrics(
    *,
    response_type: str,
    redis_client: Optional[Any] = None,
    trace_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record business metrics for a chat response.
    Called from API route after each chat.
    """
    redis = redis_client or get_redis()
    if not redis:
        return
    trace_meta = trace_meta or {}
    try:
        redis.incr(f"sahten:metrics:response_type:{response_type}:count")
        if trace_meta.get("safety_blocked"):
            redis.incr("sahten:metrics:safety_block_count")
        if trace_meta.get("exact_match"):
            redis.incr("sahten:metrics:exact_match_count")
        if trace_meta.get("shared_ingredient_proof"):
            redis.incr("sahten:metrics:proven_alternative_count")
        if response_type == "recipe_not_found":
            redis.incr("sahten:metrics:recipe_not_found_count")
        if trace_meta.get("routing_source"):
            redis.incr(f"sahten:metrics:routing:{trace_meta['routing_source']}:count")
    except Exception as e:
        logger.warning("Metrics record failed: %s", e)

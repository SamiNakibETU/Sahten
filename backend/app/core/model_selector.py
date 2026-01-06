"""
Model Selector
==============

Flexible model selection with support for:
- Environment variable default
- API request override
- A/B testing (hash-based split)

Usage:
    model = get_model_for_request(request_id, requested_model)
"""

import hashlib
import logging
from typing import Optional

from .config import get_settings

logger = logging.getLogger(__name__)


def get_model_for_request(
    request_id: str,
    requested_model: Optional[str] = None
) -> str:
    """
    Determine which model to use for a request.
    
    Priority order:
    1. Explicit request model (from API call)
    2. A/B test assignment (if enabled)
    3. Default from environment
    
    Args:
        request_id: Unique request identifier (for A/B split)
        requested_model: Model explicitly requested in API call
    
    Returns:
        Model name to use (e.g., "gpt-4.1-nano" or "gpt-4o-mini")
    """
    settings = get_settings()
    
    # 1. Explicit request takes priority
    if requested_model and requested_model != "auto":
        logger.debug("Using requested model: %s", requested_model)
        return requested_model
    
    # 2. A/B testing
    if settings.enable_ab_testing:
        model = _ab_test_assignment(request_id, settings)
        logger.debug("A/B test assigned model: %s", model)
        return model
    
    # 3. Default from config
    logger.debug("Using default model: %s", settings.openai_model)
    return settings.openai_model


def _ab_test_assignment(request_id: str, settings) -> str:
    """
    Assign a model based on hash of request_id.
    
    Uses SHA256 hash for deterministic but uniform distribution.
    Same request_id always gets same model (consistent experience).
    """
    # Create hash and convert to float [0, 1)
    hash_bytes = hashlib.sha256(request_id.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    hash_ratio = hash_int / (2 ** 64)
    
    # Split based on ratio
    if hash_ratio < settings.ab_test_ratio:
        return settings.ab_test_model_a
    else:
        return settings.ab_test_model_b


def get_ab_test_stats(request_ids: list[str]) -> dict:
    """
    Calculate A/B test distribution for a list of request IDs.
    Useful for debugging and monitoring.
    """
    settings = get_settings()
    
    if not settings.enable_ab_testing:
        return {"enabled": False}
    
    model_a_count = 0
    model_b_count = 0
    
    for rid in request_ids:
        model = _ab_test_assignment(rid, settings)
        if model == settings.ab_test_model_a:
            model_a_count += 1
        else:
            model_b_count += 1
    
    total = len(request_ids)
    return {
        "enabled": True,
        "model_a": settings.ab_test_model_a,
        "model_b": settings.ab_test_model_b,
        "target_ratio": settings.ab_test_ratio,
        "actual_ratio_a": model_a_count / total if total > 0 else 0,
        "count_a": model_a_count,
        "count_b": model_b_count,
        "total": total,
    }




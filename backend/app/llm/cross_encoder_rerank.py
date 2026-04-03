"""
Rerank optionnel par cross-encoder (sentence-transformers).
Désactivé par défaut — activer via ENABLE_CROSS_ENCODER_RERANK=true et pip install sentence-transformers torch.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_model = None


def _get_model(model_name: str):
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        logger.warning("Cross-encoder unavailable (install sentence-transformers): %s", e)
        return None
    _model = CrossEncoder(model_name)
    return _model


def score_pairs(
    query: str,
    texts: List[str],
    *,
    model_name: str,
) -> Optional[List[float]]:
    """
    Retourne un score par texte (plus haut = plus pertinent), ou None si indisponible.
    """
    if not texts:
        return []
    model = _get_model(model_name)
    if model is None:
        return None
    pairs: List[Tuple[str, str]] = [(query, t[:2000]) for t in texts]
    try:
        raw = model.predict(pairs)
        return [float(x) for x in raw]
    except Exception as e:
        logger.warning("Cross-encoder predict failed: %s", e)
        return None

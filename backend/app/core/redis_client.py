"""
Client Upstash Redis (REST) partagé : une seule instance, ping réel au premier usage.

Railway fournit souvent REDIS_URL en redis:// — incompatible avec le client REST Upstash.
Utiliser l’URL HTTPS + token REST depuis la console Upstash.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# None = pas encore initialisé ou invalidé ; False = init échouée (config ou réseau) ;
# sinon instance Redis.
_redis: Any = None


def invalidate_redis_client() -> None:
    """Réinitialise pour retenter (ex. après erreur réseau transitoire)."""
    global _redis
    _redis = None


def get_redis():
    """Client Upstash REST ou None."""
    global _redis

    if _redis is False:
        return None
    if _redis is not None:
        return _redis

    url = (os.getenv("UPSTASH_REDIS_REST_URL") or "").strip()
    token = (os.getenv("UPSTASH_REDIS_REST_TOKEN") or "").strip()

    if not url or not token:
        logger.info("Upstash non configuré : UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN")
        _redis = False
        return None

    if url.startswith("redis://") or url.startswith("rediss://"):
        logger.error(
            "UPSTASH_REDIS_REST_URL est en redis:// (TCP). "
            "Ce backend attend l’URL REST Upstash (https://….upstash.io), pas REDIS_URL Railway."
        )
        _redis = False
        return None

    try:
        from upstash_redis import Redis

        r = Redis(url=url, token=token)
        _verify_connection(r)
        _redis = r
        logger.info("Upstash Redis REST : ping OK")
        return _redis
    except Exception as e:
        logger.warning(
            "Upstash Redis indisponible : %s. "
            "Vérifie l’URL HTTPS et le token dans le dashboard Upstash.",
            e,
        )
        _redis = False
        return None


def _verify_connection(r: Any) -> None:
    if hasattr(r, "ping"):
        r.ping()
    else:
        r.get("__sahten_redis_check__")


def redis_connection_error(exc: BaseException) -> bool:
    """True si l’exception ressemble à un souci DNS / réseau."""
    if isinstance(exc, OSError):
        return True
    name = type(exc).__name__.lower()
    if "gaierror" in name or "connection" in name:
        return True
    msg = str(exc).lower()
    return (
        "name or service not known" in msg
        or "errno -2" in msg
        or "failed to resolve" in msg
        or "temporary failure" in msg
    )

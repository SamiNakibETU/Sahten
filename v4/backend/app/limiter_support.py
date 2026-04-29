"""Rate limiting (SlowAPI) — clé IP derrière proxy (X-Forwarded-For)."""

from __future__ import annotations

from slowapi import Limiter
from starlette.requests import Request

__all__ = ["limiter", "real_ip_key"]


def real_ip_key(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()[:45]
    if request.client:
        return request.client.host
    return "unknown"


limiter = Limiter(key_func=real_ip_key)

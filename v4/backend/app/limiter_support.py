"""Rate limiting (SlowAPI) — clé IP derrière proxy (X-Forwarded-For).

Sécurité : on prend le DERNIER IP de la chaîne X-Forwarded-For, car c'est
celui ajouté par le proxy de confiance (Railway Edge, Cloudflare…). Un
attaquant peut injecter des IPs arbitraires au début de la chaîne — le proxy
ajoute toujours le vrai IP à la fin. Utiliser le premier IP = rate-limit
bypassable trivially.
"""

from __future__ import annotations

from slowapi import Limiter
from starlette.requests import Request

__all__ = ["limiter", "real_ip_key"]


def real_ip_key(request: Request) -> str:
    # Railway / Cloudflare ajoutent l'IP client en dernière position.
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[-1][:45]
    if request.client:
        return request.client.host
    return "unknown"


limiter = Limiter(key_func=real_ip_key)

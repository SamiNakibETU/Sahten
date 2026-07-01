"""Rate limiting (SlowAPI) — clé = vrai IP client derrière proxy(s) de confiance.

X-Forwarded-For s'écrit `client, proxy1, proxy2, …` : chaque proxy AJOUTE (à
droite) l'IP de la connexion qu'il reçoit. Les N entrées les plus à DROITE sont
donc posées par NOS proxys de confiance ; le vrai IP client est à la position
`-N` (N = `SAHTEN_TRUSTED_PROXY_HOPS`).

- 1 proxy (Railway edge, ou AWS ALB seul) -> le client est le DERNIER IP.
- 2 proxys (AWS CloudFront -> ALB)         -> le client est l'AVANT-DERNIER
  (le dernier = IP partagée de l'edge CloudFront ; l'utiliser mettrait TOUS les
  utilisateurs dans le même quota).

Un attaquant peut injecter des IPs arbitraires au DÉBUT de la chaîne, jamais à
la fin : lire à la position `-N` (pas la première) évite le contournement.
"""

from __future__ import annotations

from slowapi import Limiter
from starlette.requests import Request

__all__ = ["limiter", "real_ip_key"]


def real_ip_key(request: Request) -> str:
    from .settings import get_settings  # import tardif : évite un cycle au boot

    hops = max(1, get_settings().trusted_proxy_hops)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            # Vrai client = N-ième en partant de la droite (N = proxys de confiance).
            # Si la chaîne est plus courte que prévu, on prend l'entrée la plus à
            # gauche disponible (meilleur effort, jamais au-delà des bornes).
            idx = -hops if len(parts) >= hops else 0
            return parts[idx][:45]
    if request.client:
        return request.client.host
    return "unknown"


limiter = Limiter(key_func=real_ip_key)

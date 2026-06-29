"""Agrégation analytics / traces en Redis (mêmes clés que le MVP, client async).

Observabilité production-grade :
- Traces de requête (TTL 30 jours)
- Latence P50 / P95 / P99 via sorted-set Redis (1 000 derniers échantillons)
- Buckets horaires (7 jours glissants) pour courbes de charge
- Compteurs d'erreurs par type
- Métriques intent / modèle / coût
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from .sessions import redis_client

log = structlog.get_logger(__name__)

_TRACE_LIST = "sahten:traces:recent"
_TRACE_MAX = 499
_LATENCY_KEY = "sahten:latency:samples"
_LATENCY_MAX = 1000  # Nombre max d'échantillons pour les percentiles


async def record_chat_trace(
    *,
    request_id: str,
    session_id: str,
    user_message: str,
    response_html: str,
    intent: str,
    confidence: float,
    recipe_count: int,
    model_used: str,
    timings_ms: dict[str, int],
    is_base2_fallback: bool = False,
    cost_breakdown: dict[str, Any] | None = None,
    error: bool = False,
    error_type: str | None = None,
    answer_strategy: str | None = None,
) -> None:
    """Enregistre une trace + compteurs métier (modèle, intent, requêtes).

    Observabilité ajoutée :
    - Latence P50/P95/P99 via sorted-set Redis
    - Bucket horaire (expiration 7 jours) pour courbes de charge
    - Compteur d'erreurs par type
    """
    r = await redis_client()
    if r is None:
        return
    try:
        now_utc = datetime.now(timezone.utc)
        latency_ms = int(timings_ms.get("total_ms", 0))
        cost_usd = 0.0
        if cost_breakdown:
            cost_usd = float(cost_breakdown.get("estimated_usd", 0.0) or 0.0)

        trace: dict[str, Any] = {
            "timestamp": now_utc.isoformat(),
            "request_id": request_id,
            "session_id": session_id,
            "user_message": user_message,
            "response_html": response_html,
            "response_type": intent,
            "intent": intent,
            "confidence": confidence,
            "recipe_count": recipe_count,
            "model_used": model_used,
            "is_base2_fallback": is_base2_fallback,
            "answer_strategy": answer_strategy,
            "latency_ms": latency_ms,
            "timings_ms": timings_ms,
            "cost_usd": round(cost_usd, 6),
            "cost_breakdown": cost_breakdown,
            "error": error,
        }
        raw = json.dumps(trace, ensure_ascii=False)

        # Bucket horaire — clé : sahten:hourly:YYYY-MM-DDTHH:count (TTL 7 j)
        hour_key = "sahten:hourly:" + now_utc.strftime("%Y-%m-%dT%H") + ":count"

        pipe = r.pipeline(transaction=False)
        pipe.set(f"trace:{request_id}", raw, ex=60 * 60 * 24 * 30)
        pipe.lpush(_TRACE_LIST, raw)
        pipe.ltrim(_TRACE_LIST, 0, _TRACE_MAX)
        pipe.incr("sahten:metrics:total_requests")

        # Latence P50/P95/P99 : sorted-set avec score = latence (ms)
        # member = timestamp:request_id pour l'unicité
        member = f"{time.time_ns()}:{request_id}"
        pipe.zadd(_LATENCY_KEY, {member: float(latency_ms)})
        pipe.zremrangebyrank(_LATENCY_KEY, 0, -(_LATENCY_MAX + 1))

        # Bucket horaire (7 jours)
        pipe.incr(hour_key)
        pipe.expire(hour_key, 7 * 24 * 3600)

        if error:
            pipe.incr("sahten:metrics:errors:total_count")
            if error_type:
                safe_type = error_type.replace(":", "_")[:40]
                pipe.incr(f"sahten:metrics:errors:{safe_type}:count")

        if is_base2_fallback:
            pipe.incr("sahten:metrics:base2_fallback_count")

        mk = model_used.replace(".", "_").replace("-", "_")
        pipe.incr(f"sahten:metrics:model:{mk}:count")
        if intent:
            pipe.incr(f"sahten:metrics:intent:{intent}:count")
        if answer_strategy:
            sk = answer_strategy.replace(":", "_")[:40]
            pipe.incr(f"sahten:metrics:strategy:{sk}:count")
        if cost_usd > 0:
            micros = int(round(cost_usd * 1_000_000))
            pipe.incrby("sahten:metrics:cost_usd_micros", micros)
            pipe.incr("sahten:metrics:cost_tracked_requests")

        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        log.warning("analytics.trace_failed", error=str(exc))


async def get_latency_percentiles() -> dict[str, float | None]:
    """Calcule P50 / P95 / P99 sur les derniers _LATENCY_MAX échantillons."""
    r = await redis_client()
    if r is None:
        return {"p50": None, "p95": None, "p99": None, "count": 0}
    try:
        count = await r.zcard(_LATENCY_KEY)
        if count == 0:
            return {"p50": None, "p95": None, "p99": None, "count": 0}
        # Redis ZRANGEBYSCORE donne les scores triés : on lit tout d'un coup.
        # Pour les percentiles on lit les scores directement.
        samples_raw = await r.zrange(_LATENCY_KEY, 0, -1, withscores=True)
        scores = sorted(float(s) for _, s in samples_raw)
        n = len(scores)

        def _percentile(p: float) -> float:
            idx = max(0, min(n - 1, int(p / 100.0 * n)))
            return round(scores[idx], 1)

        return {
            "p50": _percentile(50),
            "p95": _percentile(95),
            "p99": _percentile(99),
            "count": n,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("analytics.latency_percentile_failed", error=str(exc))
        return {"p50": None, "p95": None, "p99": None, "count": 0}


async def get_hourly_buckets(hours: int = 24) -> list[dict[str, Any]]:
    """Retourne les compteurs horaires des `hours` dernières heures."""
    r = await redis_client()
    if r is None:
        return []
    now = datetime.now(timezone.utc)
    result: list[dict[str, Any]] = []
    for h in range(hours - 1, -1, -1):
        # Calcul de l'heure passée
        ts_h = now.replace(minute=0, second=0, microsecond=0)
        offset_s = h * 3600
        ts_h_unix = ts_h.timestamp() - offset_s
        from datetime import datetime as dt
        hour_str = dt.fromtimestamp(ts_h_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H")
        key = f"sahten:hourly:{hour_str}:count"
        try:
            count = int(await r.get(key) or 0)
        except Exception:  # noqa: BLE001
            count = 0
        result.append({"hour": hour_str, "requests": count})
    return result


async def record_widget_event(
    *,
    event_type: str,
    session_id: str | None,
    request_id: str | None,
    recipe_url: str | None = None,
    recipe_title: str | None = None,
    intent: str | None = None,
    model_used: str | None = None,
) -> None:
    r = await redis_client()
    if r is None:
        return
    try:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "request_id": request_id,
            "session_id": session_id,
            "recipe_url": recipe_url,
            "recipe_title": recipe_title,
            "intent": intent,
            "model_used": model_used,
        }
        raw = json.dumps(event, ensure_ascii=False)
        pipe = r.pipeline(transaction=False)
        pipe.lpush(f"sahten:events:{event_type}", raw)
        pipe.ltrim(f"sahten:events:{event_type}", 0, 4999)
        pipe.incr(f"sahten:events:{event_type}:count")
        if event_type == "impression" and recipe_url:
            pipe.hincrby("sahten:recipe:impressions", recipe_url, 1)
        if event_type == "click" and recipe_url:
            pipe.hincrby("sahten:recipe:clicks", recipe_url, 1)
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        log.warning("analytics.event_failed", error=str(exc))


async def record_feedback_rating(
    *,
    request_id: str | None,
    session_id: str | None,
    rating: str,
    reason: str | None,
) -> None:
    r = await redis_client()
    if r is None:
        return
    try:
        pipe = r.pipeline(transaction=False)
        pipe.incr("sahten:events:feedback:count")
        if rating == "positive":
            pipe.incr("sahten:feedback:positive_count")
        else:
            pipe.incr("sahten:feedback:negative_count")
            if reason:
                pipe.lpush(
                    "sahten:feedback:negative_reasons",
                    json.dumps(
                        {
                            "reason": reason[:500],
                            "request_id": request_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        ensure_ascii=False,
                    ),
                )
                pipe.ltrim("sahten:feedback:negative_reasons", 0, 199)
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        log.warning("analytics.feedback_failed", error=str(exc))


async def get_traces(limit: int) -> dict[str, Any]:
    r = await redis_client()
    if r is None:
        return {"traces": [], "count": 0, "logging_backend": "none"}
    try:
        raw_traces = await r.lrange(_TRACE_LIST, 0, max(0, limit - 1))
        traces: list[dict[str, Any]] = []
        for raw in raw_traces:
            try:
                t = json.loads(raw) if isinstance(raw, str) else raw
                traces.append(t)
            except (json.JSONDecodeError, TypeError):
                continue
        return {
            "traces": traces,
            "count": len(traces),
            "logging_backend": "redis",
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("analytics.traces_read_failed", error=str(exc))
        return {"traces": [], "count": 0, "logging_backend": "error"}


async def get_feedback_stats() -> dict[str, Any]:
    r = await redis_client()
    if r is None:
        return {"status": "no_redis", "positive": 0, "negative": 0}
    try:
        positive = int(await r.get("sahten:feedback:positive_count") or 0)
        negative = int(await r.get("sahten:feedback:negative_count") or 0)
        total = positive + negative
        recent_raw = await r.lrange("sahten:feedback:negative_reasons", 0, 9)
        recent_negative_reasons: list[dict[str, Any]] = []
        for row in recent_raw:
            try:
                recent_negative_reasons.append(
                    json.loads(row) if isinstance(row, str) else row
                )
            except (json.JSONDecodeError, TypeError):
                continue
        return {
            "status": "ok",
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": round(positive / total * 100, 1) if total > 0 else 0,
            "recent_negative_reasons": recent_negative_reasons,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "redis_unreachable", "message": str(exc)}


async def get_analytics() -> dict[str, Any]:
    r = await redis_client()
    now = datetime.now(timezone.utc).isoformat()
    if r is None:
        return {
            "status": "no_redis",
            "message": "Redis requis pour les analytics (REDIS_URL).",
            "timestamp": now,
        }
    try:
        impressions = int(await r.get("sahten:events:impression:count") or 0)
        clicks = int(await r.get("sahten:events:click:count") or 0)
        feedback_count = int(await r.get("sahten:events:feedback:count") or 0)
        positive = int(await r.get("sahten:feedback:positive_count") or 0)
        negative = int(await r.get("sahten:feedback:negative_count") or 0)
        total_feedback = positive + negative
        ctr = round(clicks / impressions * 100, 2) if impressions > 0 else 0
        satisfaction_rate = (
            round(positive / total_feedback * 100, 1) if total_feedback > 0 else 0
        )
        total_requests = int(await r.get("sahten:metrics:total_requests") or 0)
        cost_micros = int(await r.get("sahten:metrics:cost_usd_micros") or 0)
        cost_tracked = int(await r.get("sahten:metrics:cost_tracked_requests") or 0)
        cost_usd_total = round(cost_micros / 1_000_000.0, 4)
        avg_cost_usd = (
            round(cost_usd_total / cost_tracked, 6) if cost_tracked > 0 else 0.0
        )
        base2_fallbacks = int(await r.get("sahten:metrics:base2_fallback_count") or 0)
        fallback_rate = (
            round(base2_fallbacks / total_requests * 100, 1)
            if total_requests > 0
            else 0
        )
        trace_count = await r.llen(_TRACE_LIST)
        top_clicks_raw = await r.hgetall("sahten:recipe:clicks") or {}
        top_recipes = sorted(
            [{"url": k, "clicks": int(v)} for k, v in top_clicks_raw.items()],
            key=lambda x: -x["clicks"],
        )[:10]

        model_stats: dict[str, int] = {}
        cursor = 0
        while True:
            cursor, keys = await r.scan(
                cursor=cursor, match="sahten:metrics:model:*:count", count=80
            )
            for key in keys:
                k = key.decode() if isinstance(key, bytes) else str(key)
                cnt = int(await r.get(k) or 0)
                if cnt > 0:
                    slug = k.replace("sahten:metrics:model:", "").replace(":count", "")
                    model_stats[slug] = cnt
            if cursor == 0:
                break

        intent_stats: dict[str, int] = {}
        intent_keys = [
            "recipe_specific",
            "recipe_by_ingredient",
            "recipe_by_mood",
            "recipe_by_category",
            "menu_composition",
            "multi_recipe",
            "greeting",
            "clarification",
            "off_topic",
            "redirect",
            "error",
            "unknown",
        ]
        for ik in intent_keys:
            c = int(await r.get(f"sahten:metrics:intent:{ik}:count") or 0)
            if c > 0:
                intent_stats[ik] = c

        # ── Ventilation par stratégie de réponse (observabilité du routage) ──
        strategy_stats: dict[str, int] = {}
        cursor = 0
        while True:
            cursor, keys = await r.scan(
                cursor=cursor, match="sahten:metrics:strategy:*:count", count=80
            )
            for key in keys:
                k = key.decode() if isinstance(key, bytes) else str(key)
                cnt = int(await r.get(k) or 0)
                if cnt > 0:
                    slug = k.replace("sahten:metrics:strategy:", "").replace(":count", "")
                    strategy_stats[slug] = cnt
            if cursor == 0:
                break

        # ── Latence P50/P95/P99 ───────────────────────────────────────────
        latency = await get_latency_percentiles()

        # ── Erreurs ───────────────────────────────────────────────────────
        total_errors = int(await r.get("sahten:metrics:errors:total_count") or 0)
        error_rate = (
            round(total_errors / total_requests * 100, 2) if total_requests > 0 else 0.0
        )

        return {
            "status": "ok",
            "timestamp": now,
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
                "error_rate_pct": error_rate,
            },
            "fallback": {
                "total_requests": total_requests,
                "base2_count": base2_fallbacks,
            },
            "costs": {
                "total_usd": cost_usd_total,
                "tracked_requests": cost_tracked,
                "avg_usd_per_request": avg_cost_usd,
            },
            "feedback": {
                "positive": positive,
                "negative": negative,
                "total": total_feedback,
            },
            "latency_ms": latency,
            "errors": {
                "total": total_errors,
                "rate_pct": error_rate,
            },
            "models": model_stats,
            "intents": intent_stats,
            "strategies": strategy_stats,
            "top_recipes": top_recipes,
            "quality": {
                "exact_match_count": int(
                    await r.get("sahten:metrics:exact_match_count") or 0
                ),
                "proven_alternative_count": int(
                    await r.get("sahten:metrics:proven_alternative_count") or 0
                ),
                "recipe_not_found_count": int(
                    await r.get("sahten:metrics:recipe_not_found_count") or 0
                ),
                "safety_block_count": int(
                    await r.get("sahten:metrics:safety_block_count") or 0
                ),
            },
            "response_types": {},
            "routing": {},
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "redis_unreachable", "message": str(exc), "timestamp": now}

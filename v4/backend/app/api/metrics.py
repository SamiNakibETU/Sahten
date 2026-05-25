"""GET /api/metrics — endpoint Prometheus text-format (admin protégé).

Expose les compteurs Redis au format Prometheus pour intégration
avec Grafana / UptimeRobot / Railway Metrics ou tout scraper compatible.

Usage : curl -H "X-Sahten-Admin-Token: <token>" https://<host>/api/metrics
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from .. import analytics_store
from ..auth_deps import require_admin_token
from ..sessions import redis_client

router = APIRouter(prefix="/api", tags=["metrics"])


def _gauge(name: str, value: float | int | None, labels: str = "") -> str:
    if value is None:
        return ""
    label_str = f"{{{labels}}}" if labels else ""
    return f"{name}{label_str} {value}\n"


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_admin_token)],
    summary="Métriques Prometheus (scraping)",
    description=(
        "Format text/plain compatible Prometheus. "
        "Protégé par X-Sahten-Admin-Token."
    ),
)
async def prometheus_metrics() -> PlainTextResponse:
    """Expose les métriques opérationnelles au format Prometheus."""
    r = await redis_client()
    lines: list[str] = []

    # ── Helper Redis ────────────────────────────────────────────────────────
    async def _int(key: str) -> int:
        try:
            return int(await r.get(key) or 0) if r else 0
        except Exception:  # noqa: BLE001
            return 0

    # ── Métriques de base ───────────────────────────────────────────────────
    total_requests = await _int("sahten:metrics:total_requests")
    total_errors = await _int("sahten:metrics:errors:total_count")
    base2_count = await _int("sahten:metrics:base2_fallback_count")
    cost_micros = await _int("sahten:metrics:cost_usd_micros")
    cost_tracked = await _int("sahten:metrics:cost_tracked_requests")
    impressions = await _int("sahten:events:impression:count")
    clicks = await _int("sahten:events:click:count")
    positive = await _int("sahten:feedback:positive_count")
    negative = await _int("sahten:feedback:negative_count")

    lines.append("# HELP sahten_requests_total Total de requêtes RAG traitées")
    lines.append("# TYPE sahten_requests_total counter")
    lines.append(_gauge("sahten_requests_total", total_requests))

    lines.append("# HELP sahten_errors_total Total d'erreurs pipeline")
    lines.append("# TYPE sahten_errors_total counter")
    lines.append(_gauge("sahten_errors_total", total_errors))

    error_rate = round(total_errors / total_requests * 100, 2) if total_requests else 0.0
    lines.append("# HELP sahten_error_rate_pct Taux d'erreur (%)")
    lines.append("# TYPE sahten_error_rate_pct gauge")
    lines.append(_gauge("sahten_error_rate_pct", error_rate))

    lines.append("# HELP sahten_base2_fallback_total Fallbacks base2")
    lines.append("# TYPE sahten_base2_fallback_total counter")
    lines.append(_gauge("sahten_base2_fallback_total", base2_count))

    cost_usd = round(cost_micros / 1_000_000.0, 6)
    avg_cost = round(cost_usd / cost_tracked, 6) if cost_tracked else 0.0
    lines.append("# HELP sahten_cost_usd_total Coût API total estimé (USD)")
    lines.append("# TYPE sahten_cost_usd_total counter")
    lines.append(_gauge("sahten_cost_usd_total", cost_usd))

    lines.append("# HELP sahten_avg_cost_usd_per_request Coût moyen par requête (USD)")
    lines.append("# TYPE sahten_avg_cost_usd_per_request gauge")
    lines.append(_gauge("sahten_avg_cost_usd_per_request", avg_cost))

    lines.append("# HELP sahten_widget_impressions_total Impressions widget")
    lines.append("# TYPE sahten_widget_impressions_total counter")
    lines.append(_gauge("sahten_widget_impressions_total", impressions))

    lines.append("# HELP sahten_widget_clicks_total Clics sur recettes")
    lines.append("# TYPE sahten_widget_clicks_total counter")
    lines.append(_gauge("sahten_widget_clicks_total", clicks))

    ctr = round(clicks / impressions * 100, 2) if impressions else 0.0
    lines.append("# HELP sahten_ctr_pct Click-through rate (%)")
    lines.append("# TYPE sahten_ctr_pct gauge")
    lines.append(_gauge("sahten_ctr_pct", ctr))

    lines.append("# HELP sahten_feedback_positive_total Feedbacks positifs")
    lines.append("# TYPE sahten_feedback_positive_total counter")
    lines.append(_gauge("sahten_feedback_positive_total", positive))

    lines.append("# HELP sahten_feedback_negative_total Feedbacks négatifs")
    lines.append("# TYPE sahten_feedback_negative_total counter")
    lines.append(_gauge("sahten_feedback_negative_total", negative))

    total_feedback = positive + negative
    satisfaction = round(positive / total_feedback * 100, 1) if total_feedback else 0.0
    lines.append("# HELP sahten_satisfaction_pct Taux de satisfaction (%)")
    lines.append("# TYPE sahten_satisfaction_pct gauge")
    lines.append(_gauge("sahten_satisfaction_pct", satisfaction))

    # ── Latence P50/P95/P99 ────────────────────────────────────────────────
    latency = await analytics_store.get_latency_percentiles()
    lines.append("# HELP sahten_latency_ms Latence pipeline (ms) — percentiles")
    lines.append("# TYPE sahten_latency_ms gauge")
    for pct, key in [("50", "p50"), ("95", "p95"), ("99", "p99")]:
        val = latency.get(key)
        if val is not None:
            lines.append(_gauge("sahten_latency_ms", val, f'quantile="{pct}"'))

    # ── Métriques par modèle ───────────────────────────────────────────────
    if r:
        lines.append("# HELP sahten_model_requests_total Requêtes par modèle LLM")
        lines.append("# TYPE sahten_model_requests_total counter")
        try:
            cursor = 0
            while True:
                cursor, keys = await r.scan(
                    cursor=cursor, match="sahten:metrics:model:*:count", count=50
                )
                for key in keys:
                    k = key.decode() if isinstance(key, bytes) else str(key)
                    cnt = int(await r.get(k) or 0)
                    slug = k.replace("sahten:metrics:model:", "").replace(":count", "")
                    if cnt > 0:
                        lines.append(
                            _gauge("sahten_model_requests_total", cnt, f'model="{slug}"')
                        )
                if cursor == 0:
                    break
        except Exception:  # noqa: BLE001
            pass

    # ── Métriques par intent ───────────────────────────────────────────────
    intent_keys = [
        "recipe", "recipe_specific", "recipe_by_ingredient", "recipe_by_mood",
        "recipe_by_category", "menu_composition", "multi_recipe",
        "greeting", "clarification", "off_topic", "redirect", "error", "unknown",
    ]
    lines.append("# HELP sahten_intent_requests_total Requêtes par intent détecté")
    lines.append("# TYPE sahten_intent_requests_total counter")
    for ik in intent_keys:
        cnt = await _int(f"sahten:metrics:intent:{ik}:count")
        if cnt > 0:
            lines.append(_gauge("sahten_intent_requests_total", cnt, f'intent="{ik}"'))

    # ── Corpus / santé ─────────────────────────────────────────────────────
    lines.append("# HELP sahten_traces_in_redis Nombre de traces en mémoire Redis")
    lines.append("# TYPE sahten_traces_in_redis gauge")
    if r:
        try:
            trace_count = await r.llen("sahten:traces:recent")
            lines.append(_gauge("sahten_traces_in_redis", trace_count))
        except Exception:  # noqa: BLE001
            pass

    output = "".join(lines)
    return PlainTextResponse(
        content=output,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )

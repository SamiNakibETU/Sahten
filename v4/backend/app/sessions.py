"""Persistance d'historique de chat (Redis si dispo, mémoire sinon).

Schéma Redis :
    sahten:session:<session_id>            -> JSON {created_at, ...}
    sahten:session:<session_id>:messages   -> LIST de JSON {role, ts, ...payload}
    sahten:sessions                        -> SORTED SET (score = last_seen_ts)

L'API publique reste minimale : `record_turn`, `list_sessions`,
`get_session_messages`. En cas d'absence de Redis (env local sans broker),
on bascule sur un store en mémoire process-local — suffisant pour les tests
et pour ne pas bloquer le démarrage.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from typing import Any

import structlog

from .settings import get_settings

log = structlog.get_logger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)


def _html_to_plain(html: str, *, max_len: int = 4500) -> str:
    """Texte lisible pour le prompt LLM (réponses assistant déjà rendues en HTML).

    Assez long pour inclure cartes recette + relance (follow_up) en fin de tour :
    une coupure trop agressive cassait la « mémoire » conversationnelle.
    """
    if not html:
        return ""
    t = _HTML_TAG_RE.sub(" ", html)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t


async def conversation_block_for_llm(
    session_id: str,
    *,
    max_messages: int = 24,
    max_chars: int = 14000,
) -> str:
    """Derniers tours user/assistant (hors le message en cours), pour le grounding."""
    msgs = await get_session_messages(session_id)
    if not msgs:
        return ""
    lines: list[str] = []
    for m in msgs[-max_messages:]:
        role = m.get("role")
        if role == "user":
            tx = (m.get("text") or "").strip()
            if tx:
                if len(tx) > 4000:
                    tx = tx[:3999] + "…"
                lines.append(f"Utilisateur : {tx}")
        elif role == "assistant":
            tx = _html_to_plain(m.get("html") or "")
            if tx:
                lines.append(f"Assistant : {tx}")
    block = "\n".join(lines).strip()
    if len(block) > max_chars:
        block = "…\n" + block[-max_chars:]
    return block


_MEMORY_MESSAGES: dict[str, list[dict[str, Any]]] = defaultdict(list)
_MEMORY_LAST_SEEN: dict[str, float] = {}

_SESSION_KEY = "sahten:session:{sid}"
_MESSAGES_KEY = "sahten:session:{sid}:messages"
_INDEX_KEY = "sahten:sessions"
_MAX_MESSAGES_PER_SESSION = 200
_DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 jours


_redis_client: Any = None
_redis_unavailable = False


async def _get_redis() -> Any | None:
    """Connexion Redis async paresseuse. Retourne None si indisponible."""
    global _redis_client, _redis_unavailable
    if _redis_client is not None:
        return _redis_client
    if _redis_unavailable:
        return None
    try:
        from redis.asyncio import Redis  # type: ignore[import-not-found]

        url = str(get_settings().redis_url)
        client = Redis.from_url(url, decode_responses=True)
        await client.ping()
        _redis_client = client
        return _redis_client
    except Exception as exc:  # noqa: BLE001
        _redis_unavailable = True
        log.warning("sessions.redis_unavailable", error=str(exc))
        return None


async def record_turn(
    session_id: str,
    *,
    user_message: str,
    assistant_html: str,
    request_id: str,
    intent: str | None = None,
    confidence: float | None = None,
    sources: list[dict[str, Any]] | None = None,
    timings_ms: dict[str, int] | None = None,
    model_used: str | None = None,
) -> None:
    """Enregistre un tour user → assistant dans la session."""
    ts = time.time()
    user_payload = {"role": "user", "ts": ts, "text": user_message}
    bot_payload = {
        "role": "assistant",
        "ts": ts,
        "html": assistant_html,
        "request_id": request_id,
        "intent": intent,
        "confidence": confidence,
        "sources": sources or [],
        "timings_ms": timings_ms or {},
        "model_used": model_used,
    }
    redis = await _get_redis()
    if redis is None:
        bucket = _MEMORY_MESSAGES[session_id]
        bucket.extend([user_payload, bot_payload])
        if len(bucket) > _MAX_MESSAGES_PER_SESSION:
            del bucket[: len(bucket) - _MAX_MESSAGES_PER_SESSION]
        _MEMORY_LAST_SEEN[session_id] = ts
        return

    try:
        msg_key = _MESSAGES_KEY.format(sid=session_id)
        meta_key = _SESSION_KEY.format(sid=session_id)
        pipe = redis.pipeline(transaction=False)
        pipe.rpush(msg_key, json.dumps(user_payload, ensure_ascii=False))
        pipe.rpush(msg_key, json.dumps(bot_payload, ensure_ascii=False))
        pipe.ltrim(msg_key, -_MAX_MESSAGES_PER_SESSION, -1)
        pipe.expire(msg_key, _DEFAULT_TTL_SECONDS)
        pipe.zadd(_INDEX_KEY, {session_id: ts})
        pipe.hset(meta_key, mapping={"last_seen": str(ts)})
        pipe.expire(meta_key, _DEFAULT_TTL_SECONDS)
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        log.warning("sessions.record_turn_failed", session_id=session_id, error=str(exc))
        bucket = _MEMORY_MESSAGES[session_id]
        bucket.extend([user_payload, bot_payload])
        _MEMORY_LAST_SEEN[session_id] = ts


async def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Liste les sessions les plus récentes (id + last_seen + n_messages)."""
    redis = await _get_redis()
    if redis is None:
        items = sorted(
            _MEMORY_LAST_SEEN.items(), key=lambda x: x[1], reverse=True
        )[:limit]
        return [
            {
                "session_id": sid,
                "last_seen": ts,
                "n_messages": len(_MEMORY_MESSAGES.get(sid, [])),
            }
            for sid, ts in items
        ]
    try:
        raw = await redis.zrevrange(_INDEX_KEY, 0, limit - 1, withscores=True)
        out: list[dict[str, Any]] = []
        for sid, score in raw:
            n = await redis.llen(_MESSAGES_KEY.format(sid=sid))
            out.append(
                {"session_id": sid, "last_seen": float(score), "n_messages": int(n)}
            )
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("sessions.list_failed", error=str(exc))
        return []


async def recent_article_external_ids(
    session_id: str,
    *,
    max_assistant_turns: int = 6,
    max_ids: int = 24,
) -> list[int]:
    """IDs d'articles déjà cités dans les réponses récentes (diversité RAG)."""
    msgs = await get_session_messages(session_id)
    seen: list[int] = []
    assistant_turns = 0
    for m in reversed(msgs):
        if m.get("role") != "assistant":
            continue
        assistant_turns += 1
        if assistant_turns > max_assistant_turns:
            break
        for src in m.get("sources") or []:
            aid = src.get("article_external_id")
            if aid is None:
                continue
            try:
                i = int(aid)
            except (TypeError, ValueError):
                continue
            if i not in seen:
                seen.append(i)
            if len(seen) >= max_ids:
                return seen
    return seen


async def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    redis = await _get_redis()
    if redis is None:
        return list(_MEMORY_MESSAGES.get(session_id, []))
    try:
        raw = await redis.lrange(_MESSAGES_KEY.format(sid=session_id), 0, -1)
        return [json.loads(item) for item in raw]
    except Exception as exc:  # noqa: BLE001
        log.warning("sessions.fetch_failed", session_id=session_id, error=str(exc))
        return []


async def healthcheck() -> dict[str, Any]:
    redis = await _get_redis()
    if redis is None:
        return {"backend": "memory", "ok": True, "n_sessions": len(_MEMORY_LAST_SEEN)}
    try:
        await redis.ping()
        n = await redis.zcard(_INDEX_KEY)
        return {"backend": "redis", "ok": True, "n_sessions": int(n)}
    except Exception as exc:  # noqa: BLE001
        return {"backend": "redis", "ok": False, "error": str(exc)}


__all__ = [
    "record_turn",
    "list_sessions",
    "get_session_messages",
    "recent_article_external_ids",
    "conversation_block_for_llm",
    "healthcheck",
]

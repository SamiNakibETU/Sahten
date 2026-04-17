"""
Session manager with TTL + LRU and conversation memory.
Sessions are persisted to Upstash Redis when available so they survive
process restarts and work across multiple workers.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from typing import Any, Optional
import re
import threading

logger = logging.getLogger(__name__)

_SESSION_TTL_SECONDS = 7200  # 2 h

# Défauts si Settings non utilisés (tests hors app)
_DEFAULT_MAX_HISTORY_TURNS = 4
_DEFAULT_MAX_TURN_TEXT_CHARS = 280


@dataclass
class ConversationTurn:
    user_message: str
    bot_response_summary: str
    intent: str
    primary_dish: str | None
    ingredients: list[str]
    recipe_url: str | None
    recipe_titles: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class SessionState:
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    recipes_proposed: list[str] = field(default_factory=list)
    ingredients_mentioned: set[str] = field(default_factory=set)
    preferences: dict[str, Any] = field(default_factory=dict)

    def add_turn(
        self,
        user_message: str,
        intent: str,
        primary_dish: str | None = None,
        ingredients: list[str] | None = None,
        recipe_url: str | None = None,
        response_summary: str = "",
        recipe_titles: list[str] | None = None,
    ) -> None:
        titles = [t.strip() for t in (recipe_titles or []) if t and str(t).strip()]
        turn = ConversationTurn(
            user_message=user_message,
            bot_response_summary=response_summary,
            intent=intent,
            primary_dish=primary_dish,
            ingredients=ingredients or [],
            recipe_url=recipe_url,
            recipe_titles=titles,
        )
        self.conversation_history.append(turn)
        if recipe_url and recipe_url not in self.recipes_proposed:
            self.recipes_proposed.append(recipe_url)
        if ingredients:
            self.ingredients_mentioned.update({ing.lower() for ing in ingredients if ing})
        self.last_activity = datetime.now(UTC)

    def has_history(self) -> bool:
        """Return True if at least one conversation turn exists."""
        return bool(self.conversation_history)

    def get_last_turn(self) -> ConversationTurn | None:
        return self.conversation_history[-1] if self.conversation_history else None

    def should_exclude(self, url: str | None) -> bool:
        return bool(url and url in self.recipes_proposed)

    def get_context_for_continuation(self) -> dict[str, Any]:
        """Contexte pour la requête en cours (historique déjà enregistré, sans le message actuel)."""
        try:
            from ..core.config import get_settings

            s = get_settings()
            max_turns = int(getattr(s, "session_max_turns", _DEFAULT_MAX_HISTORY_TURNS) or _DEFAULT_MAX_HISTORY_TURNS)
            max_chars = int(getattr(s, "session_turn_max_chars", _DEFAULT_MAX_TURN_TEXT_CHARS) or _DEFAULT_MAX_TURN_TEXT_CHARS)
        except Exception:
            max_turns = _DEFAULT_MAX_HISTORY_TURNS
            max_chars = _DEFAULT_MAX_TURN_TEXT_CHARS

        last = self.get_last_turn()
        recent_turns: list[dict[str, str]] = []
        for turn in self.conversation_history[-max_turns:]:
            u = (turn.user_message or "").strip()
            if u:
                if len(u) > max_chars:
                    u = u[: max_chars - 1] + "…"
                recent_turns.append({"role": "user", "text": u})
            asst = (turn.bot_response_summary or "").strip()
            if asst:
                if len(asst) > max_chars:
                    asst = asst[: max_chars - 1] + "…"
                recent_turns.append({"role": "assistant", "text": asst})

        # Titres de fiches proposés (ordre chronologique, dédupliqués)
        recent_recipe_titles: list[str] = []
        seen_t: set[str] = set()
        for turn in self.conversation_history[-max_turns:]:
            for t in turn.recipe_titles:
                key = t.strip().lower()
                if key and key not in seen_t:
                    seen_t.add(key)
                    recent_recipe_titles.append(t.strip())

        base: dict[str, Any] = {
            "recent_turns": recent_turns,
            "has_prior_turns": bool(self.conversation_history),
            "recent_recipe_titles": recent_recipe_titles[-8:],
            "recipes_proposed_count": len(self.recipes_proposed),
        }
        if not last:
            return base
        base.update(
            {
                "last_intent": last.intent,
                "last_dish": last.primary_dish,
                "last_ingredients": last.ingredients,
                "last_recipe_url": last.recipe_url,
                "all_ingredients": list(self.ingredients_mentioned),
                "preferences": self.preferences,
            }
        )
        return base


def _redis_key(session_id: str) -> str:
    return f"sahten:session:{session_id}"


def _session_to_redis(session: SessionState) -> str:
    """Serialize the deduplication-critical fields to JSON for Redis."""
    turns = []
    for t in session.conversation_history:
        turns.append({
            "recipe_url": t.recipe_url,
            "recipe_titles": t.recipe_titles,
            "intent": t.intent,
            "primary_dish": t.primary_dish,
            "ingredients": t.ingredients,
        })
    return json.dumps({
        "recipes_proposed": session.recipes_proposed,
        "turns": turns,
    })


def _restore_from_redis(session: SessionState, raw: str) -> None:
    """Restore deduplication state from Redis payload into an existing SessionState."""
    try:
        data = json.loads(raw)
        proposed = data.get("recipes_proposed") or []
        for url in proposed:
            if url and url not in session.recipes_proposed:
                session.recipes_proposed.append(url)
        for turn_data in (data.get("turns") or []):
            url = turn_data.get("recipe_url")
            titles = turn_data.get("recipe_titles") or []
            if url or titles:
                existing_urls = {t.recipe_url for t in session.conversation_history}
                if url not in existing_urls:
                    synthetic = ConversationTurn(
                        user_message="",
                        bot_response_summary="",
                        intent=turn_data.get("intent") or "",
                        primary_dish=turn_data.get("primary_dish"),
                        ingredients=turn_data.get("ingredients") or [],
                        recipe_url=url,
                        recipe_titles=titles,
                    )
                    session.conversation_history.append(synthetic)
    except Exception as exc:
        logger.warning("Failed to restore session from Redis: %s", exc)


class SessionManager:
    """LRU + TTL in-memory session manager with optional Redis persistence.

    Redis is used to persist the deduplication state (recipes_proposed +
    recipe_titles) across process restarts and multiple workers.
    In-memory cache avoids a Redis round-trip on every request.
    """

    def __init__(self, max_sessions: int = 2000, ttl_minutes: int = 30) -> None:
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.ttl = timedelta(minutes=ttl_minutes)

    # ── Redis helpers (fire-and-forget, never raise) ─────────────────────

    def _redis_load(self, session_id: str) -> Optional[str]:
        try:
            from ..core.redis_client import get_redis
            r = get_redis()
            if r is None:
                return None
            return r.get(_redis_key(session_id))
        except Exception as exc:
            logger.debug("Redis session load error: %s", exc)
            return None

    def _redis_save(self, session: SessionState) -> None:
        try:
            from ..core.redis_client import get_redis
            r = get_redis()
            if r is None:
                return
            payload = _session_to_redis(session)
            r.set(_redis_key(session.session_id), payload, ex=_SESSION_TTL_SECONDS)
        except Exception as exc:
            logger.debug("Redis session save error: %s", exc)

    # ── Public API ───────────────────────────────────────────────────────

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            self._cleanup_expired()
            if session_id in self._sessions:
                session = self._sessions.pop(session_id)
                session.last_activity = datetime.now(UTC)
                self._sessions[session_id] = session
                return session
            # Not in memory — create fresh and try to restore from Redis
            session = SessionState(session_id=session_id)
            raw = self._redis_load(session_id)
            if raw:
                _restore_from_redis(session, raw)
                logger.debug("Session %s restored from Redis (%d urls)", session_id, len(session.recipes_proposed))
            self._sessions[session_id] = session
            return session

    def save(self, session: SessionState) -> None:
        with self._lock:
            session.last_activity = datetime.now(UTC)
            if session.session_id in self._sessions:
                self._sessions.pop(session.session_id)
            self._sessions[session.session_id] = session
        # Persist to Redis outside the lock (non-blocking for in-memory ops)
        self._redis_save(session)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "ttl_minutes": self.ttl.total_seconds() / 60,
            }

    def get_context_hints(self, session_id: str | None) -> dict[str, Any]:
        if not session_id:
            return {}
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return {}
            last_turn = session.get_last_turn()
            return {
                "recipes_proposed": list(session.recipes_proposed),
                "ingredients_mentioned": list(session.ingredients_mentioned),
                "last_intent": last_turn.intent if last_turn else None,
                "turn_count": len(session.conversation_history),
            }

    def _cleanup_expired(self) -> None:
        now = datetime.now(UTC)
        expired = [
            sid for sid, session in self._sessions.items() if now - session.last_activity > self.ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)


CONTINUATION_PATTERNS = [
    r"^oui\b",
    r"^non\b",
    r"^ok\b",
    r"^d'accord\b",
    r"^une autre",
    r"^autre chose",
    r"^plutot",
    r"^version",
    r"^avec\b",
    r"^sans\b",
    r"^et (?:si|avec|pour)",
    r"^et sans\b",
    r"^la meme",
    r"^celle",
    r"^pareil",
    r"^encore",
    r"^aussi",
    r"^sinon",
    r"^peut-?etre",
    r"^je (?:veux|voudrais|prefere)",
    r"^(?:vegetarien|vegetalien|vegan|sans viande|sans gluten)",
]


def format_session_hints_for_analyzer(session_context: dict[str, Any]) -> str:
    """Inject conversation context into the QueryAnalyzer prompt.

    Enables implicit continuation resolution:
    "encore une autre" -> reuse last intent and ingredients.
    """
    if not session_context or not session_context.get("has_prior_turns"):
        return ""

    lines: list[str] = [
        "CONTEXTE DE CONVERSATION (meme session) — utilise ces informations pour"
        " interpreter les requetes courtes comme encore une autre, une autre recette,"
        " non, autre chose, sans viande, etc. :",
    ]

    last_intent = session_context.get("last_intent")
    last_dish = session_context.get("last_dish")
    last_ingredients = session_context.get("last_ingredients") or []
    all_ingredients = list(session_context.get("all_ingredients") or [])

    if last_intent:
        lines.append(f"- Dernier intent : {last_intent}")
    if last_dish:
        lines.append(f"- Dernier plat demande : {last_dish}")
    if last_ingredients:
        lines.append(f"- Derniers ingredients demandes : {', '.join(last_ingredients[:5])}")
    elif all_ingredients:
        lines.append(f"- Ingredients mentionnes dans la conversation : {', '.join(all_ingredients[:5])}")

    titles = session_context.get("recent_recipe_titles") or []
    if titles:
        lines.append(f"- Recettes deja proposees : {' ; '.join(titles[:6])}")

    lines.append(
        "REGLE : si la requete est une continuation implicite (encore une autre,"
        " non, une autre), conserve le meme intent et les memes"
        " ingredients/plat que le dernier tour."
    )
    return "\n".join(lines)


def format_recent_turns_for_retrieval(session_context: dict[str, Any], *, max_chars: int = 900) -> str:
    """
    Résume le fil récent pour le retrieveur / reranker (reformulation implicite multi-tours).
    """
    turns = session_context.get("recent_turns") or []
    if not turns:
        return ""
    lines: list[str] = []
    for t in turns:
        role = t.get("role", "")
        text = (t.get("text") or "").strip()
        if not text:
            continue
        label = "Utilisateur" if role == "user" else "Assistant"
        lines.append(f"{label}: {text}")
    try:
        from ..core.config import get_settings

        mc = int(getattr(get_settings(), "session_retrieval_context_max_chars", max_chars) or max_chars)
        if mc > 0:
            max_chars = mc
    except Exception:
        pass

    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        return out[: max_chars - 1] + "…"
    return out


def is_continuation(message: str) -> bool:
    """Detect whether a short reply continues the previous discussion."""
    if not message:
        return False
    message_lower = message.strip().lower()
    normalized = (
        message_lower.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("ù", "u")
        .replace("û", "u")
    )
    return any(re.match(pattern, normalized) for pattern in CONTINUATION_PATTERNS)


__all__ = [
    "SessionManager",
    "SessionState",
    "ConversationTurn",
    "format_recent_turns_for_retrieval",
    "format_session_hints_for_analyzer",
    "is_continuation",
]

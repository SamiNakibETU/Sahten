"""
Conversation premium quality tests.

These tests validate the new control-plane/generation-plane behavior:
- response keeps conversational mention of user query
- blocks are populated for richer rendering
- proven alternatives include concrete shared ingredients context
"""

from __future__ import annotations

import pytest

from app.bot import reload_bot
from app.core.config import get_settings


@pytest.mark.asyncio
async def test_alternative_response_mentions_query_and_recipe(monkeypatch):
    """Alternative narrative must explicitly mention requested dish and proposed recipe."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    bot = reload_bot()
    response, _, _ = await bot.chat("recette boeuf bourguignon", debug=True, session_id="conv-test-1")

    assert response.response_type == "not_found_with_alternative"
    assert response.conversation_blocks, "Conversation blocks should be present"
    assistant_blocks = [b.text.lower() for b in response.conversation_blocks if b.block_type == "assistant_message"]
    assert assistant_blocks, "assistant_message block expected"
    combined = " ".join(assistant_blocks)
    assert "boeuf bourguignon" in combined
    assert any((r.title or "").lower().split()[0] in combined for r in response.recipes if r.title)
    assert "partage au moins un ingrédient" not in combined


@pytest.mark.asyncio
async def test_continuation_context_adds_follow_up_block(monkeypatch):
    """Continuation query should trigger a follow-up conversational block."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    bot = reload_bot()
    session_id = "conv-test-2"
    await bot.chat("recette boeuf bourguignon", debug=False, session_id=session_id)
    response, _, _ = await bot.chat("et sans viande ?", debug=False, session_id=session_id)

    follow_ups = [b for b in response.conversation_blocks if b.block_type == "follow_up_question"]
    assert follow_ups, "Follow-up question block expected on continuation"


@pytest.mark.asyncio
async def test_trace_contains_timings_and_compact_assistant_block(monkeypatch):
    """Trace meta should expose timings and assistant message should stay compact."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    bot = reload_bot()
    response, _, trace_meta = await bot.chat("recette fajitas", debug=True, session_id="conv-test-3")

    assert "timings_ms" in trace_meta
    assert "total" in trace_meta["timings_ms"]
    assistant_blocks = [b.text for b in response.conversation_blocks if b.block_type == "assistant_message"]
    assert assistant_blocks, "assistant_message block expected"
    assert all(len(b) <= 500 for b in assistant_blocks)


"""Tests du chunker hybride sur de vrais payloads WhiteBeard.

Hypothèse cruciale pour le RAG conversationnel : sur le taboulé de Kamal
Mouzawak (1227694), on doit obtenir ≥ 1 chunk par commandement, par
ingrédient et par étape, plus un chunk d'ancrage. C'est la granularité
qui permet à des requêtes type « quel est le 6ᵉ commandement du
taboulé ? » de retomber sur le bon fragment isolé.
"""

from __future__ import annotations

from collections import Counter

from backend.app.ingestion.mapper import map_article
from backend.app.rag.chunker import chunk_article


def _chunks(payload):
    m = map_article(payload)
    return m, chunk_article(
        article_external_id=m.external_id,
        article_title=m.title,
        sections=m.sections,
    )


def test_taboule_produces_anchor_chunk(taboule_payload):
    _, chunks = _chunks(taboule_payload)
    anchors = [c for c in chunks if c.kind == "anchor"]
    assert len(anchors) == 1
    a = anchors[0]
    assert "Titre" in a.text
    assert "taboul" in a.text.lower()


def test_taboule_one_chunk_per_commandment(taboule_payload):
    _, chunks = _chunks(taboule_payload)
    history_chunks = [c for c in chunks if c.kind == "recipe_history"]
    # Les 9 commandements de Kamal Mouzawak.
    assert len(history_chunks) >= 9, f"expected >=9 history chunks, got {len(history_chunks)}"
    # Chaque chunk a un item_index unique et incrémental.
    indices = [c.metadata.get("item_index") for c in history_chunks]
    assert sorted(indices) == list(range(len(indices)))
    # On doit retrouver le 6ᵉ commandement (oignon + sel + poivre doux).
    sixth = next(
        c for c in history_chunks if c.metadata.get("item_index") == 5
    )
    assert "oignon" in sixth.text.lower() or "poivre doux" in sixth.text.lower()


def test_taboule_one_chunk_per_ingredient(taboule_payload):
    _, chunks = _chunks(taboule_payload)
    ing = [c for c in chunks if c.kind == "ingredients_list"]
    assert len(ing) >= 8, f"expected >=8 ingredient chunks, got {len(ing)}"
    persil = [c for c in ing if "persil" in c.text.lower()]
    assert persil, "ingrédient persil manquant"


def test_taboule_one_chunk_per_step(taboule_payload):
    _, chunks = _chunks(taboule_payload)
    steps = [c for c in chunks if c.kind == "recipe_steps"]
    assert len(steps) >= 5, f"expected >=5 step chunks, got {len(steps)}"


def test_taboule_includes_chef_bio_chunk(taboule_payload):
    _, chunks = _chunks(taboule_payload)
    bio = [c for c in chunks if c.kind == "chef_bio"]
    assert bio, "chef_bio chunk manquant"
    assert any("souk el-tayeb" in c.text.lower() for c in bio)


def test_taboule_total_chunk_count_meets_target(taboule_payload):
    """Pour la conversation, on veut une granularité riche : au moins 12 chunks."""
    _, chunks = _chunks(taboule_payload)
    by_kind = Counter(c.kind for c in chunks)
    assert sum(by_kind.values()) >= 18, (
        f"expected >=18 chunks for taboulé, got {sum(by_kind.values())}; breakdown={dict(by_kind)}"
    )


def test_chunk_headers_are_well_formed(taboule_payload):
    """Les chunks doivent commencer par un header `[titre | kind | ...]`."""
    _, chunks = _chunks(taboule_payload)
    for c in chunks:
        first_line = c.text.split("\n", 1)[0]
        assert first_line.startswith("[") and first_line.endswith("]")
        assert "taboul" in first_line.lower()


def test_no_truncated_word_at_chunk_start(taboule_payload):
    """Régression du bug initial : pas de chunks commençant par fragment de mot."""
    _, chunks = _chunks(taboule_payload)
    for c in chunks:
        body = c.text.split("\n\n", 1)[-1].lstrip()
        if not body:
            continue
        # Premier caractère doit être lettre maj/min, chiffre, tiret, ou ponctuation d'ouverture
        assert body[0].isalnum() or body[0] in "«\"'-(", (
            f"chunk démarre par fragment suspect : {body[:60]!r}"
        )

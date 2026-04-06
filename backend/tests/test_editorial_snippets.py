"""Extraits éditoriaux depuis search_text."""

from app.rag.editorial_snippets import (
    extract_editorial_snippets,
    extract_recipe_lead,
    extract_story_snippet,
)


def test_extract_recipe_lead_taboule_sample():
    text = (
        "Le vrai taboulé Carla HENOUD On ne plaisante pas avec le taboulé ! "
        "Triez le persil. 2 bottes de persil plat 1 c. à soupe de bourghol"
    )
    lead = extract_recipe_lead(text, max_chars=500)
    assert "taboulé" in lead.lower()


def test_extract_story_snippet_mere():
    text = (
        "La recette X. L'ultime plaisir est de le déguster avec des feuilles "
        "comme aimait le faire ma mère. Ensuite la suite."
    )
    sn = extract_story_snippet(text, cited_passage=None, max_chars=200)
    assert "mère" in sn.lower() or "plaisir" in sn.lower()


def test_extract_editorial_snippets_combined():
    search = (
        "Titre Chef Une tarte riche en saveurs pour faire durer l'été. "
        "Pour la pâte 115 g de beurre. « Ma famille en est fière » raconte le chef."
    )
    lead, story = extract_editorial_snippets(search, "Une tarte riche", combined_max=800)
    assert lead
    assert len(lead) + len(story) <= 805


def test_format_session_hints_for_analyzer():
    from app.rag.session_manager import format_session_hints_for_analyzer

    s = format_session_hints_for_analyzer({})
    assert s == ""

    h = format_session_hints_for_analyzer(
        {
            "has_prior_turns": True,
            "recent_recipe_titles": ["Le taboulé"],
            "recipes_proposed_count": 1,
        }
    )
    assert "taboulé" in h.lower() or "Taboulé" in h
    assert "1" in h

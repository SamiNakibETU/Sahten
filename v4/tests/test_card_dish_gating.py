"""Gating de pertinence des cartes recette par plat demandé (anti carte hors-sujet)."""

from backend.app.rag.pipeline import (
    _card_title_matches_requested_dish,
    _requested_dish_terms,
)


def test_requested_dish_terms_detects_manouche() -> None:
    terms = _requested_dish_terms("recette manouche")
    assert terms
    assert any("manaiche" in t or "manaiichs" in t or "manouche" in t for t in terms)


def test_requested_dish_terms_empty_for_plain_ingredient_query() -> None:
    assert _requested_dish_terms("recette au poulet pour ce soir") == []


def test_card_dropped_when_title_off_topic() -> None:
    requested = _requested_dish_terms("recette manouche")
    # le bug : carte « Lahm bi aajine » sur une demande de manouche
    assert _card_title_matches_requested_dish("Le Tripoli-style Lahm bi aajine", requested) is False
    assert _card_title_matches_requested_dish("Les pâtes au kishk et pesto de zaatar", requested) is False


def test_card_kept_when_title_matches_dish() -> None:
    requested = _requested_dish_terms("recette manouche")
    assert _card_title_matches_requested_dish("Les manaïichs du Chouf de Salim Azzam", requested) is True


def test_no_gate_when_no_dish_requested() -> None:
    # pas de plat précis -> aucune contrainte, on garde la carte
    assert _card_title_matches_requested_dish("N'importe quelle fiche", []) is True

"""Gating de pertinence des cartes recette par plat demandé (anti carte hors-sujet)."""

from backend.app.llm.response_generator import GroundedAnswer, GroundedSentence
from backend.app.rag.pipeline import (
    _card_title_matches_requested_dish,
    _drop_speculative_ingredients,
    _ensure_recipe_card,
    _primary_dish_canonical,
    _requested_dish_terms,
)
from backend.app.rag.reranker import RerankedHit
from backend.app.rag.retriever import Hit


def _rh(title: str, chunk_id: int = 101, chef: str = "Salim Azzam", score: float = 0.8) -> RerankedHit:
    h = Hit(
        chunk_id=chunk_id, article_id=1, article_external_id=1474718, article_title=title,
        article_url="", cover_image_url=None, section_kind="recipe_steps", chunk_text="...",
        score_lex=None, score_vec=None, score_rrf=0.5, metadata={"featured_chef": chef},
    )
    return RerankedHit(hit=h, rerank_score=score)


def _ans(conf: float, card=None) -> GroundedAnswer:
    return GroundedAnswer(
        answer_sentences=[GroundedSentence(text="Le manouche est une galette...", source_chunk_ids=[101])],
        recipe_card=card, recipe_card_secondary=None, chef_card=None, follow_up="", confidence=conf,
    )


def test_ensure_card_synthesizes_when_relevant_article_found() -> None:
    rh = _rh("Les manaïichs du Chouf de Salim Azzam")
    out = _ensure_recipe_card(_ans(0.85), [rh], "recette manouche", 0.2)
    assert out.recipe_card is not None
    assert "mana" in out.recipe_card.title.lower()
    assert out.recipe_card.chef == "Salim Azzam"


def test_ensure_card_respects_abstention_low_confidence() -> None:
    rh = _rh("Les manaïichs du Chouf de Salim Azzam")
    out = _ensure_recipe_card(_ans(0.2), [rh], "recette de sushi", 0.2)
    assert out.recipe_card is None


def test_ensure_card_noop_when_card_exists() -> None:
    from backend.app.llm.response_generator import RecipeCard
    existing = RecipeCard(title="Déjà là")
    out = _ensure_recipe_card(_ans(0.9, card=existing), [_rh("X")], "recette manouche", 0.2)
    assert out.recipe_card is existing


def test_ensure_card_skips_when_dish_named_but_no_match() -> None:
    rh = _rh("Le sfouf libanais")  # ne matche pas "manouche"
    out = _ensure_recipe_card(_ans(0.85), [rh], "recette manouche", 0.2)
    assert out.recipe_card is None


def test_ensure_card_for_named_dish_even_low_confidence() -> None:
    # plat nommé + article correspondant -> carte même si confiance moyenne
    rh = _rh("Les manaïichs du Chouf de Salim Azzam")
    out = _ensure_recipe_card(_ans(0.35), [rh], "recette manouche", 0.2)
    assert out.recipe_card is not None
    assert "mana" in out.recipe_card.title.lower()


def test_primary_dish_canonical_strips_conversational_noise() -> None:
    assert _primary_dish_canonical("je voudrais un hommos classique") == "houmous"
    assert _primary_dish_canonical("je veux du hommos") == "houmous"
    assert _primary_dish_canonical("recette manouche") == "manaiche"
    assert _primary_dish_canonical("manakish") == "manaiche"


def test_primary_dish_canonical_none_when_no_dish() -> None:
    assert _primary_dish_canonical("une recette au poulet pour ce soir") is None


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


# ── Garde anti-ingrédients spéculatifs (bug manouche -> zaatar/pain-pita) ────

def test_drop_invented_ingredients_for_named_dish() -> None:
    # 'manouche' est un plat connu ; l'utilisateur n'a pas tapé zaatar/pain-pita
    assert _drop_speculative_ingredients(["zaatar", "pain-pita"], "recette manouche") == []


def test_drop_all_ingredients_for_named_dish() -> None:
    # plat nommé -> tous les filtres d'ingrédients retirés (le plat est la cible).
    # Même un ingrédient tapé ne doit pas filtrer (l'article du plat l'a déjà).
    assert _drop_speculative_ingredients(["concombre"], "recette de fattouche au concombre") == []
    assert _drop_speculative_ingredients(["fromage"], "recette de manakish au fromage") == []


def test_no_drop_when_no_dish_named() -> None:
    # pas de plat connu nommé -> on ne touche pas aux slugs (ex. requête ingrédient)
    assert _drop_speculative_ingredients(["poulet"], "recette au poulet") == ["poulet"]


def test_drop_empty_is_noop() -> None:
    assert _drop_speculative_ingredients([], "manouche") == []

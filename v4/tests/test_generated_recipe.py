"""Fallback dernier recours : matching base2 fuzzy + recette générée (hors OLJ)."""

from backend.app.llm.query_understanding import QueryPlan
from backend.app.rag.pipeline import (
    _build_generated_recipe_answer,
    _card_title_matches_requested_dish,
    _fallback_dish_name,
    _match_base2_recipe,
    _olj_has_dish_text,
    _requested_dish_terms,
)
from backend.app.rag.recipe_generator import normalize_dish
from backend.app.rag.reranker import RerankedHit
from backend.app.rag.retriever import Hit


def _mk_hit(*, aid: int, title: str, url: str, chunk_id: int) -> RerankedHit:
    return RerankedHit(
        hit=Hit(
            chunk_id=chunk_id,
            article_id=aid,
            article_external_id=aid,
            article_title=title,
            article_url=url,
            cover_image_url=None,
            section_kind="recipe_summary",
            chunk_text=title,
            score_lex=0.0,
            score_vec=0.0,
            score_rrf=0.0,
            metadata={},
        ),
        rerank_score=0.5,
    )


def _plan(intent: str = "recipe", ingredient_slugs: list[str] | None = None) -> QueryPlan:
    return QueryPlan(
        rewritten_query="x", intent=intent, ingredient_slugs=ingredient_slugs or []
    )


def test_base2_fuzzy_matches_bare_and_phrased() -> None:
    assert (_match_base2_recipe("fattouche") or {}).get("name") == "Salade fattouche"
    assert (_match_base2_recipe("recette de fattouche") or {}).get("name") == "Salade fattouche"
    # plat absent de base2 -> None (ira en génération)
    assert _match_base2_recipe("je veux un knefe") is None
    # requête ingrédient -> pas de faux match
    assert _match_base2_recipe("recette avec du concombre") is None


def test_fallback_dish_name_only_for_real_dish_requests() -> None:
    assert _fallback_dish_name("knefe", _plan()) == "knefe"
    assert _fallback_dish_name("recette de baklava", _plan()) == "baklava"
    # requête ingrédient (marqueur « avec ») -> pas de génération
    assert _fallback_dish_name("recette avec du concombre", _plan(ingredient_slugs=["concombre"])) is None
    # intention non-recette -> pas de génération
    assert _fallback_dish_name("qui est ce chef", _plan(intent="chef_bio")) is None
    # plat nu que le LLM a (à tort) mis comme son propre ingrédient -> on génère quand même
    assert _fallback_dish_name("katayef", _plan(ingredient_slugs=["katayef"])) == "katayef"


def test_manouche_title_matches_despite_curly_apostrophe() -> None:
    # le titre OLJ a une apostrophe COURBE (mana'ichs) ; l'alias une DROITE
    req = _requested_dish_terms("recette de manouche")
    assert req  # manouche est un plat connu
    assert _card_title_matches_requested_dish(
        "Les mana’ichs du Chouf de Salim Azzam", req
    )


def test_olj_has_dish_text() -> None:
    reranked = [_mk_hit(aid=1, title="Le knefe de X", url="u", chunk_id=1)]
    assert _olj_has_dish_text("knefe", reranked) is True
    assert _olj_has_dish_text("baklava", reranked) is False


def test_build_generated_recipe_answer_labels_and_suggests() -> None:
    generated = {
        "name": "Knefe",
        "serves": "4 personnes",
        "prep": "20 min",
        "cook": "15 min",
        "difficulty": "moyenne",
        "ingredients": ["500 g de kataifi", "300 g de fromage akkawi"],
        "steps": ["Beurrer le moule.", "Cuire 15 min."],
        "note": "Servir tiède.",
    }
    reranked = [
        _mk_hit(
            aid=2,
            title="Le maamoul de chef Amani",
            url="https://www.lorientlejour.com/cuisine-liban-a-table/1477552/maamoul.html",
            chunk_id=20,
        )
    ]
    out = _build_generated_recipe_answer(
        user_query="knefe", generated=generated, reranked=reranked
    )
    text = " ".join(s.text for s in out.answer_sentences)
    assert "hors carnets OLJ" in text  # étiquetage non négociable
    assert "kataifi" in text and "Beurrer le moule" in text  # contenu généré présent
    # suggestion OLJ rendue en carte (avec chunk source valide)
    assert out.recipe_card is not None
    assert out.recipe_card.title == "Le maamoul de chef Amani"


def test_normalize_dish() -> None:
    assert normalize_dish("Knéfé") == "knefe"
    assert normalize_dish("Baba Ghannouj'") == "baba ghannouj"

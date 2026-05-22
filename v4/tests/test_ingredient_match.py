from backend.app.llm.query_understanding import QueryPlan
from backend.app.rag.ingredient_match import (
    chunk_confirms_ingredient,
    extract_ingredient_slugs_from_text,
    filter_hits_by_ingredient_slugs,
    is_short_follow_up,
    supplement_ingredient_slugs,
    wants_another_recipe,
)
from backend.app.rag.retriever import Hit


def _hit(aid: int, chunk_id: int, kind: str, text: str, title: str = "T") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        article_id=aid,
        article_external_id=aid,
        article_title=title,
        article_url=f"https://example.com/{aid}",
        cover_image_url=None,
        section_kind=kind,
        chunk_text=text,
        score_lex=0.5,
        score_vec=0.5,
        score_rrf=0.5,
        metadata={},
    )


def test_extract_concombre_from_user_query() -> None:
    slugs = extract_ingredient_slugs_from_text("recette avec du concombre")
    assert "concombre" in slugs


def test_supplement_ingredient_slugs_from_history() -> None:
    plan = QueryPlan(rewritten_query="autre idée", intent="recipe")
    out = supplement_ingredient_slugs(
        plan,
        "non avec du concombre",
        "Utilisateur : recette avec du concombre\nAssistant : mehche selek",
    )
    assert "concombre" in out.ingredient_slugs


def test_filter_hits_keeps_only_ingredient_list_matches() -> None:
    good = _hit(1, 10, "ingredients_list", "2 concombres", "Fattouche")
    bad = _hit(2, 20, "ingredients_list", "blettes, riz", "Mehche selek")
    bad_step = _hit(
        3,
        30,
        "recipe_steps",
        "Servir avec yaourt au concombre",
        "Kibbeh",
    )
    out = filter_hits_by_ingredient_slugs(
        [good, bad, bad_step], ["concombre"]
    )
    aids = {h.article_external_id for h in out}
    assert aids == {1}


def test_chunk_confirms_ingredient_strict_on_steps_only() -> None:
    assert not chunk_confirms_ingredient(
        "recipe_steps",
        "Servir avec yaourt au concombre",
        "concombre",
    )
    assert chunk_confirms_ingredient(
        "ingredients_list",
        "2 concombres moyens",
        "concombre",
    )


def test_short_follow_up_detection() -> None:
    assert is_short_follow_up("oui")
    assert is_short_follow_up("une autre")
    assert not is_short_follow_up("recette avec du concombre")


def test_wants_another_recipe() -> None:
    assert wants_another_recipe("une autre")
    assert wants_another_recipe("encore une autre recette")
    assert not wants_another_recipe("recette houmous")


def test_fuzzy_typo_tomatge() -> None:
    slugs = extract_ingredient_slugs_from_text("recette avec de la tomatge")
    assert "tomate" in slugs

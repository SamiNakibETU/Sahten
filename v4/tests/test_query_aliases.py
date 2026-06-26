from backend.app.llm.query_understanding import QueryPlan
from backend.app.rag.pipeline import (
    _canonicalize_dish_aliases,
    _expand_query_with_aliases,
    _expand_search_q_with_ingredients,
    _retrieval_fallback_queries,
)


def test_expand_query_with_aliases_for_houmous() -> None:
    out = _expand_query_with_aliases("recette houmous")
    low = out.lower()
    assert "houmous" in low
    assert "hommos" in low
    assert "hummus" in low


def test_retrieval_fallback_queries_include_hommos_variant() -> None:
    plan = QueryPlan(
        rewritten_query="recette houmous",
        intent="recipe",
        ingredient_slugs=["houmous"],
    )
    queries = _retrieval_fallback_queries("recette hoummous", plan)
    blob = " | ".join(queries).lower()
    assert "hommos" in blob
    assert "hummus" in blob


def test_expand_search_query_with_ingredient_aliases() -> None:
    plan = QueryPlan(
        rewritten_query="recette hoummous",
        intent="recipe",
        ingredient_slugs=["houmous"],
    )
    out = _expand_search_q_with_ingredients("recette hoummous", plan).lower()
    assert "hommos" in out


# ── Canonicalisation des translittérations de plats (bug manouche & co.) ────

def test_canonicalize_manouche_variants_to_indexed_form() -> None:
    for q in ("recette manouche", "recette manouché", "man'ouché", "manakish"):
        out = _canonicalize_dish_aliases(q).lower()
        assert "manaiche" in out, f"{q!r} -> {out!r}"
        assert "manouche" not in out and "manakish" not in out


def test_canonicalize_taboule_english_spellings() -> None:
    for q in ("recette de tabbouleh", "tabbouli", "taboule"):
        assert "taboulé" in _canonicalize_dish_aliases(q).lower()


def test_canonicalize_kofta_and_mouloukhieh() -> None:
    assert "kafta" in _canonicalize_dish_aliases("recette de kofta").lower()
    assert "kafta" in _canonicalize_dish_aliases("köfte").lower()
    assert "mouloukhiyé" in _canonicalize_dish_aliases("molokheya").lower()


def test_canonicalize_is_noop_for_plain_query() -> None:
    assert _canonicalize_dish_aliases("recette de poulet") == "recette de poulet"


def test_expand_query_canonicalizes_then_keeps_houmous_group() -> None:
    # non-régression : la canonicalisation ne casse pas l'expansion houmous
    out = _expand_query_with_aliases("recette houmous").lower()
    assert "houmous" in out and "hommos" in out and "hummus" in out
    # et la canonicalisation manouche s'applique dans le même chemin
    assert "manaiche" in _expand_query_with_aliases("recette manouche").lower()


def test_manouche_full_pipeline_query_paths() -> None:
    plan = QueryPlan(rewritten_query="recette manouche", intent="recipe")
    blob = " | ".join(_retrieval_fallback_queries("recette manouche", plan)).lower()
    assert "manaiche" in blob


# ── Canonicalisation des ingrédients translittérés ──────────────────────────

def test_canonicalize_ingredient_transliterations() -> None:
    assert "zaatar" in _canonicalize_dish_aliases("recette au za'atar").lower()
    assert "sumac" in _canonicalize_dish_aliases("avec du sumak").lower()
    assert "freekeh" in _canonicalize_dish_aliases("recette de frikeh").lower()


def test_ingredient_canon_does_not_hijack_houmous_dish() -> None:
    # pois-chiche est volontairement exclu : 'houmous' reste un PLAT, pas 'pois chiche'
    out = _canonicalize_dish_aliases("recette houmous").lower()
    assert "pois" not in out
    assert "houmous" in out

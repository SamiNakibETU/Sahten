"""Lecture du graphe de cooccurrence ingrédients (fondation métadonnées/graphe)."""

from backend.app.rag.cooccurrence import related_ingredients, related_with_scores


def test_graph_loads_and_has_neighbors() -> None:
    # aubergine est un ingrédient fréquent du corpus -> doit avoir des voisins
    nb = related_ingredients("aubergine", k=8)
    assert nb, "aucun voisin pour aubergine (graphe non chargé ?)"


def test_aubergine_neighbors_are_culinary() -> None:
    nb = set(related_ingredients("aubergine", k=8))
    # cooccurrences culinaires réelles attendues
    assert nb & {"poivron", "courgette", "tomate"}, nb


def test_unknown_ingredient_returns_empty() -> None:
    assert related_ingredients("xyz-pas-un-ingredient") == []


def test_scores_are_sorted_desc() -> None:
    scored = related_with_scores("aubergine", k=8)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)

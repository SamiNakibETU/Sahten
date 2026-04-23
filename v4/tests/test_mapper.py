"""Tests du mapping payload WhiteBeard -> MappedArticle.

Vérité terrain : on charge les vrais payloads téléchargés via
``scripts/audit_whitebeard.py`` et on s'assure que le mapping refactor
n'oublie aucune section utile au RAG (ingrédients, étapes,
commandements, astuce, bio chef, metadata structurées).
"""

from __future__ import annotations

from backend.app.ingestion.mapper import map_article


def test_taboule_basic_fields(taboule_payload):
    m = map_article(taboule_payload)
    assert m.external_id == 1227694
    assert m.url.endswith(".html")
    assert "taboulé" in m.title.lower() or "taboule" in m.title.lower()
    assert m.summary  # description éditoriale (et non plus HTML d'ingrédients)
    assert "<ul>" not in m.summary, "summary doit rester du texte propre"
    assert m.first_published_at is not None
    assert m.is_premium is False


def test_taboule_body_html_contains_all_sections(taboule_payload):
    m = map_article(taboule_payload)
    assert m.body_html is not None
    body = m.body_html.lower()
    # Les balises sémantiques injectées par _assemble_body_html.
    for kind in (
        "data-kind=\"recipe_meta\"",
        "data-kind=\"recipe_summary\"",
        "data-kind=\"recipe_history\"",
        "data-kind=\"ingredients_list\"",
        "data-kind=\"recipe_steps\"",
        "data-kind=\"chef_astuce\"",
        "data-kind=\"chef_bio\"",
    ):
        assert kind in body, f"section manquante dans body_html : {kind}"
    # Contenu : commandements + astuce + bio chef.
    assert "commandement" in body
    assert "persil" in body  # ingrédient & astuce
    assert "souk el-tayeb" in body  # bio chef Kamal Mouzawak
    assert "kamal mouzawak" in body


def test_taboule_chef_promoted_to_author(taboule_payload):
    m = map_article(taboule_payload)
    names = {a.name for a in m.authors}
    assert "Kamal Mouzawak" in names
    chef = next(a for a in m.authors if a.name == "Kamal Mouzawak")
    assert chef.role == "featured_chef"
    assert chef.biography_text and "Souk el-Tayeb" in chef.biography_text
    assert chef.department == "cuisine"
    assert chef.image_url and chef.image_url.startswith("https://")


def test_taboule_keywords_normalized(taboule_payload):
    m = map_article(taboule_payload)
    names = {k.name.lower() for k in m.keywords}
    # Keywords sous le nouveau format `{ "data": { "name": ... } }`
    assert any("recette" in n for n in names)
    assert any("liban" in n or "moyen-orient" in n or "méditerran" in n for n in names)
    # Pas de doublons (slug unique).
    slugs = [k.slug for k in m.keywords]
    assert len(slugs) == len(set(slugs))


def test_taboule_categories_with_parent(taboule_payload):
    m = map_article(taboule_payload)
    names = {c.name for c in m.categories}
    # On doit retrouver la sous-catégorie ET la racine "Recettes".
    assert "Recettes" in names


def test_taboule_cover_from_attachments(taboule_payload):
    m = map_article(taboule_payload)
    assert m.cover_image_url and m.cover_image_url.startswith("https://")


def test_spaghettis_minimal_recipe_still_complete(spaghettis_payload):
    """Recette sans astuce ni history doit quand même produire les sections de base."""
    m = map_article(spaghettis_payload)
    assert m.body_html is not None
    body = m.body_html.lower()
    for kind in (
        "data-kind=\"recipe_meta\"",
        "data-kind=\"ingredients_list\"",
        "data-kind=\"recipe_steps\"",
        "data-kind=\"chef_bio\"",
    ):
        assert kind in body
    # Pas d'astuce/history pour cette recette → pas de section vide.
    assert "data-kind=\"chef_astuce\"" not in body
    assert "data-kind=\"recipe_history\"" not in body


def test_status_ok_when_full_payload(taboule_payload):
    m = map_article(taboule_payload)
    assert m.ingestion_status == "ok"

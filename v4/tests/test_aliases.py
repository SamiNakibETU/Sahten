"""Enrichissement d'alias à l'indexation (solution systémique translittérations)."""

from backend.app.rag.aliases import article_alias_text, transliterate_variants


def test_translit_variants_cover_common_spellings() -> None:
    v = transliterate_variants("manouché")
    assert "manouche" in v  # accent retiré


def test_manouche_article_links_via_title_token() -> None:
    # le titre indexé "mana'ichs" doit lier le groupe manouche -> graphies utilisateur
    txt = article_alias_text("Les mana'ichs du Chouf de Salim Azzam")
    assert "manouche" in txt
    assert "manakish" in txt


def test_taboule_article_gets_user_spellings() -> None:
    txt = article_alias_text("Le « vrai » taboulé de Kamal Mouzawak")
    assert "tabbouleh" in txt or "taboule" in txt


def test_no_alias_for_unknown_dish() -> None:
    assert article_alias_text("Un plat inconnu xyzzy sans correspondance") == ""

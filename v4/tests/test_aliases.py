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


def test_sfiha_links_via_lahm_bi_ajin_title() -> None:
    # le plat "sfiha" est indexé sous le titre "Lahm bi ajin" -> graphies rattachées
    txt = article_alias_text("Lahm bi ajin tripolitaine d'Alan Geaam")
    assert "sfiha" in txt
    assert "sfeeha" in txt
    # faux ami : sfouf (gâteau) ne doit PAS être rattaché ici
    assert "sfouf" not in txt


def test_warak_enab_links_via_title() -> None:
    txt = article_alias_text("Warak enab (Feuilles de vigne farcies) de Tara Khattar")
    assert "dolma" in txt
    assert "warak" in txt


def test_chawarma_and_falafel_spelling_variants() -> None:
    assert "shawarma" in article_alias_text("Le chawarma de poulet de John Achkar")
    assert "falafil" in article_alias_text("Les falafels « maison » d'Andrée Maalouf")


def test_no_alias_for_unknown_dish() -> None:
    assert article_alias_text("Un plat inconnu xyzzy sans correspondance") == ""

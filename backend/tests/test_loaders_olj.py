"""
Focused tests for OLJ article loading/indexing robustness.
"""

from app.data.loaders import load_olj_articles
from app.data.content_index import ContentIndex
from app.data.link_index import LinkIndex

TABOULE_URL = "https://www.lorientlejour.com/cuisine-liban-a-table/1227694/le-vrai-taboule-de-kamal-mouzawak.html"


def test_taboule_article_is_loaded_and_count_is_high_enough():
    """Ensure the taboulé recipe exists in the loader output."""
    articles = load_olj_articles()

    assert len(articles) >= 120, "Expect at least 120 OLJ recipes after refreshed scrape"

    matches = [article for article in articles if article.url == TABOULE_URL]
    assert matches, "Taboulé article missing from loaded OLJ recipes"

    taboule_article = matches[0]
    assert "taboul" in taboule_article.title.lower()
    assert taboule_article.main_ingredients, "Taboulé article should expose derived ingredients"


def test_taboule_article_present_in_content_and_link_index():
    """The taboulé article must exist in both indexes."""
    articles = load_olj_articles()

    # Content index coverage
    content_index = ContentIndex()
    content_index.add_olj_articles(articles)
    content_index.build()

    content_match = None
    for doc in content_index.documents:
        if doc.metadata.get("url") == TABOULE_URL:
            content_match = doc
            break
    assert content_match is not None, "Taboulé article missing from content index"

    # Link index coverage
    link_index = LinkIndex()
    link_index.add_articles(articles)
    link_index.build()

    link_match = None
    for doc in link_index.documents:
        if doc.article.url == TABOULE_URL:
            link_match = doc.article
            break
    assert link_match is not None, "Taboulé article missing from link index"


def test_new_dataset_carries_derived_metadata():
    """Representative recipes contain derived metadata for hybrid retrieval."""
    articles = load_olj_articles()

    maamoul = next(
        article
        for article in articles
        if "maamoul" in article.title.lower()
    )

    assert maamoul.course == "dessert"
    assert maamoul.diet in {"vege", "viande", "poisson"}
    assert maamoul.main_ingredients, "Expected extracted main ingredients"
    assert maamoul.editorial_score > 0
    assert maamoul.recency_score > 0
    assert maamoul.doc_text, "Doc text should not be empty"

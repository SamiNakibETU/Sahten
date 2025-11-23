"""
Tests for data loaders and indexes
"""

import pytest
from app.data.loaders import (
    load_olj_articles,
    load_structured_recipes,
    load_golden_examples,
    data_cache,
)
from app.data.content_index import ContentIndex
from app.data.link_index import LinkIndex
from app.data.culinary_graph import culinary_graph


def test_load_olj_articles():
    """Test loading OLJ articles from Base 1"""
    articles = load_olj_articles()

    assert len(articles) > 0, "Should load at least one article"

    # Check first article structure
    article = articles[0]
    assert article.article_id is not None
    assert article.title is not None
    assert article.url.startswith("https://www.lorientlejour.com")
    assert article.normalized_title is not None


def test_load_structured_recipes():
    """Test loading structured recipes from Base 2"""
    recipes = load_structured_recipes()

    assert len(recipes) > 0, "Should load at least one recipe"

    # Check first recipe structure
    recipe = recipes[0]
    assert recipe.recipe_id is not None
    assert recipe.name is not None
    assert len(recipe.ingredients) > 0
    assert len(recipe.steps) > 0


def test_load_golden_examples():
    """Test loading golden examples"""
    examples = load_golden_examples()

    assert len(examples) > 0, "Should load at least one example"

    # Check first example
    example = examples[0]
    assert example.id is not None
    assert example.user_query is not None
    assert example.response is not None


def test_data_cache():
    """Test data cache singleton"""
    # First load
    articles1 = data_cache.get_olj_articles()
    recipes1 = data_cache.get_structured_recipes()
    examples1 = data_cache.get_golden_examples()

    # Second load (should be cached)
    articles2 = data_cache.get_olj_articles()
    recipes2 = data_cache.get_structured_recipes()
    examples2 = data_cache.get_golden_examples()

    # Should be the same instances
    assert articles1 is articles2
    assert recipes1 is recipes2
    assert examples1 is examples2


def test_content_index():
    """Test content index building and search"""
    articles = data_cache.get_olj_articles()
    recipes = data_cache.get_structured_recipes()

    index = ContentIndex()
    index.add_olj_articles(articles)
    index.add_structured_recipes(recipes)
    index.build()

    assert index.is_built
    assert len(index) > 0

    # Test search
    results = index.search("taboulé", top_k=5)
    assert len(results) > 0

    # Each result should be (document, score)
    doc, score = results[0]
    assert doc.content is not None
    assert 0 <= score <= 1


def test_link_index():
    """Test link index for article resolution"""
    articles = data_cache.get_olj_articles()

    index = LinkIndex()
    index.add_articles(articles)
    index.build()

    assert index.is_built
    assert len(index) > 0

    # Test exact match (if "taboulé" article exists)
    exact = index.find_exact_match("taboulé")
    # May or may not find exact match depending on data

    # Test similarity search
    results = index.find_best_match("taboulé", top_k=3)
    if results:
        article, score, strategy = results[0]
        assert article.url.startswith("https://www.lorientlejour.com")
        assert 0 <= score <= 1
        assert strategy in ["exact", "high_similarity", "moderate_similarity", "low_similarity"]

    # Test fallback
    fallbacks = index.get_fallback_articles(strategy="recent", count=3)
    assert len(fallbacks) <= 3
    for article in fallbacks:
        assert article.url.startswith("https://www.lorientlejour.com")


def test_culinary_graph():
    """Test culinary knowledge graph"""
    # Test finding dishes
    hummus = culinary_graph.find_dish("hummus")
    assert hummus is not None
    assert hummus.category == "mezze_cold"

    # Test variations
    houmous = culinary_graph.find_dish("houmous")
    assert houmous is not None  # Should find via variation

    # Test Lebanese check
    assert culinary_graph.is_lebanese_dish("taboulé")

    # Test category
    category = culinary_graph.get_dish_category("hummus")
    assert category == "mezze_cold"

    # Test ingredients
    ingredients = culinary_graph.get_key_ingredients("hummus")
    assert "pois chiches" in ingredients


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
End-to-end pipeline tests
Tests the complete RAG pipeline from query to response
"""

import pytest
from app.rag.pipeline import RAGPipeline
from app.data.loaders import data_cache
from app.data.content_index import ContentIndex
from app.data.link_index import LinkIndex
from app.models.llm_client import MockLLMClient
from app.models.config import settings


@pytest.fixture(scope="module")
def pipeline():
    """Create complete pipeline"""
    # Load data
    articles = data_cache.get_olj_articles()
    recipes = data_cache.get_structured_recipes()

    # Build indexes
    content_index = ContentIndex()
    content_index.add_olj_articles(articles)
    content_index.add_structured_recipes(recipes)
    content_index.build()

    link_index = LinkIndex()
    link_index.add_articles(articles)
    link_index.build()

    # Create pipeline with mock LLM
    return RAGPipeline(
        content_index=content_index,
        link_index=link_index,
        llm_client=MockLLMClient(),
    )


def test_pipeline_greeting(pipeline):
    """Test pipeline with greeting"""
    response = pipeline.process("Bonjour", debug=True)

    assert response.html is not None
    assert "<p>" in response.html
    assert response.scenario_id in [4, 7]  # Greeting or non-French
    # Should have French greeting words
    assert any(word in response.html.lower() for word in ["bonjour", "salut", "français", "sahtein"])


def test_pipeline_recipe_query(pipeline):
    """Test pipeline with recipe query"""
    response = pipeline.process("Je veux la recette du hummus", debug=True)

    assert response.html is not None
    assert "<p>" in response.html
    # Should have some URL if OLJ article found
    if response.primary_url:
        assert response.primary_url.startswith(settings.allowed_url_domain)


def test_pipeline_off_topic(pipeline):
    """Test pipeline with off-topic query"""
    response = pipeline.process("Parle-moi de la politique française", debug=True)

    assert response.html is not None
    # Should redirect to cooking or be detected as non-French
    assert response.scenario_id in [6, 3, 7]  # Off-topic, fallback, or non-French


def test_pipeline_non_french(pipeline):
    """Test pipeline with non-French query"""
    response = pipeline.process("Hello, how are you?", debug=True)

    assert response.html is not None
    assert response.scenario_id == 7  # Non-French
    # Should respond in French
    assert "français" in response.html.lower()


def test_pipeline_ingredient_query(pipeline):
    """Test pipeline with ingredient query"""
    response = pipeline.process("J'ai des pois chiches, que faire?", debug=True)

    assert response.html is not None
    assert "<p>" in response.html
    # Should suggest something


def test_pipeline_html_validity(pipeline):
    """Test that all responses contain valid HTML"""
    queries = [
        "Bonjour",
        "recette de taboulé",
        "comment faire du kebbeh",
        "mezze froid",
    ]

    for query in queries:
        response = pipeline.process(query, debug=False)

        # Should have HTML tags
        assert "<p>" in response.html or "<a" in response.html

        # Should not have Markdown
        assert "**" not in response.html
        assert not response.html.startswith("#")


def test_pipeline_url_safety(pipeline):
    """Test that all URLs are from allowed domain"""
    queries = [
        "recette de hummus",
        "taboulé libanais",
        "dessert oriental",
    ]

    for query in queries:
        response = pipeline.process(query, debug=False)

        if response.primary_url:
            assert response.primary_url.startswith(settings.allowed_url_domain), \
                f"Invalid URL for query '{query}': {response.primary_url}"

        # Check URLs in HTML
        import re
        urls = re.findall(r'https?://[^\s<>"]+', response.html)
        for url in urls:
            assert url.startswith(settings.allowed_url_domain), \
                f"Invalid URL in HTML for query '{query}': {url}"


def test_pipeline_debug_mode(pipeline):
    """Test pipeline debug mode"""
    response = pipeline.process("recette de hummus", debug=True)

    assert response.debug_info is not None
    assert "classification" in response.debug_info
    assert "query_plan" in response.debug_info


def test_pipeline_error_handling(pipeline):
    """Test pipeline handles errors gracefully"""
    # Empty query
    response = pipeline.process("", debug=False)
    assert response.html is not None

    # Very long query
    long_query = "recette " * 100
    response = pipeline.process(long_query, debug=False)
    assert response.html is not None


def test_pipeline_consistency(pipeline):
    """Test that same query produces consistent results"""
    query = "recette de taboulé"

    response1 = pipeline.process(query, debug=False)
    response2 = pipeline.process(query, debug=False)

    # Should have same scenario
    assert response1.scenario_id == response2.scenario_id

    # Should have same primary URL (if any)
    assert response1.primary_url == response2.primary_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

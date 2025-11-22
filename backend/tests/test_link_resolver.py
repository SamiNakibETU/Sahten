"""
Tests for LinkResolver
Ensures ultra-precise URL handling with no generation
"""

import pytest
from app.rag.link_resolver import LinkResolver
from app.rag.classifier_agent import ClassifierAgent
from app.rag.query_planner_agent import QueryPlannerAgent
from app.data.loaders import data_cache
from app.data.link_index import LinkIndex
from app.models.llm_client import MockLLMClient
from app.models.config import settings


@pytest.fixture(scope="module")
def link_index():
    """Create and build link index"""
    articles = data_cache.get_olj_articles()

    index = LinkIndex()
    index.add_articles(articles)
    index.build()

    return index


@pytest.fixture
def resolver(link_index):
    """Create link resolver"""
    return LinkResolver(link_index)


@pytest.fixture
def classifier():
    """Create classifier"""
    return ClassifierAgent(llm_client=MockLLMClient())


@pytest.fixture
def planner():
    """Create planner"""
    return QueryPlannerAgent()


def test_resolve_with_exact_match(resolver, classifier, planner):
    """Test resolution with exact dish name match"""
    # Use a common Lebanese dish that likely exists in the data
    query = "recette de taboulé"
    classification = classifier.classify(query)
    plan = planner.plan(classification, query)

    result = resolver.resolve(plan)

    # Should find some article (exact or similar)
    if result.primary_article:
        # Must be a valid OLJ URL
        assert result.primary_article.url.startswith(settings.allowed_url_domain)
        assert resolver.validate_url(result.primary_article.url)


def test_resolve_no_link_for_greeting(resolver, classifier, planner):
    """Test that greetings get a fallback article to showcase OLJ content"""
    query = "Bonjour"
    classification = classifier.classify(query)
    plan = planner.plan(classification, query)

    result = resolver.resolve(plan)

    # After P0 fix: greeting should get fallback article to showcase OLJ
    assert result.primary_article is not None
    assert result.strategy == "greeting_fallback"
    assert result.confidence == 0.5
    assert resolver.validate_url(result.primary_article.url)


def test_resolve_fallback_when_no_match(resolver, classifier, planner):
    """Test fallback strategy when no match is found"""
    # Query for something unlikely to match
    query = "recette de pizza hawaïenne"  # Not Lebanese
    classification = classifier.classify(query)
    plan = planner.plan(classification, query)

    result = resolver.resolve(plan)

    # Should use fallback strategy
    assert "fallback" in result.strategy or result.primary_article is None
    if result.primary_article:
        assert resolver.validate_url(result.primary_article.url)


def test_all_urls_are_valid(resolver, classifier, planner):
    """Test that all resolved URLs are from allowed domain"""
    queries = [
        "recette de hummus",
        "kebbeh libanais",
        "mezze froid",
        "dessert libanais",
    ]

    for query in queries:
        classification = classifier.classify(query)
        plan = planner.plan(classification, query)
        result = resolver.resolve(plan)

        if result.primary_article:
            assert resolver.validate_url(result.primary_article.url), \
                f"Invalid URL for query '{query}': {result.primary_article.url}"

        for suggested in result.suggested_articles:
            assert resolver.validate_url(suggested.url), \
                f"Invalid suggested URL for query '{query}': {suggested.url}"


def test_validate_url_rejects_invalid(resolver):
    """Test URL validation rejects invalid domains"""
    # Valid URL
    assert resolver.validate_url("https://www.lorientlejour.com/cuisine-liban-a-table/1234/recipe.html")

    # Invalid URLs
    assert not resolver.validate_url("https://example.com/recipe.html")
    assert not resolver.validate_url("http://www.lorientlejour.com/recipe.html")  # Wrong protocol
    assert not resolver.validate_url("")
    assert not resolver.validate_url("not-a-url")


def test_suggested_articles_are_relevant(resolver, classifier, planner):
    """Test that suggested articles are relevant and not duplicates"""
    query = "recette de taboulé"
    classification = classifier.classify(query)
    plan = planner.plan(classification, query)

    result = resolver.resolve(plan)

    if result.primary_article and result.suggested_articles:
        # Suggested articles should not include primary
        primary_id = result.primary_article.article_id
        suggested_ids = [art.article_id for art in result.suggested_articles]

        assert primary_id not in suggested_ids, "Suggested articles should not include primary"

        # No duplicates in suggested
        assert len(suggested_ids) == len(set(suggested_ids)), "No duplicate suggested articles"


def test_resolve_from_candidates(resolver):
    """Test resolution from retrieval candidates"""
    from app.models.schemas import RetrievalCandidate, QueryPlan

    # Get a real article from the index
    articles = list(resolver.link_index._article_by_id.values())
    if not articles:
        pytest.skip("No articles in index")

    test_article = articles[0]

    # Create a mock retrieval candidate
    candidate = RetrievalCandidate(
        source="olj",
        content="test content",
        score=0.9,
        metadata={"article_id": test_article.article_id},
        article_id=test_article.article_id,
    )

    # Create a mock query plan
    plan = QueryPlan(
        need_type="recipe_by_name",
        primary_dish=None,
        ingredients=[],
        constraints=[],
        language="fr",
        retrieval_query="test",
        link_query=None,
    )

    result = resolver.resolve(plan, retrieval_candidates=[candidate])

    assert result.primary_article is not None
    assert result.primary_article.article_id == test_article.article_id
    assert result.strategy == "from_retrieval"


def test_confidence_scores_are_valid(resolver, classifier, planner):
    """Test that confidence scores are in valid range [0, 1]"""
    queries = [
        "recette de hummus",
        "mezze libanais",
        "dessert oriental",
    ]

    for query in queries:
        classification = classifier.classify(query)
        plan = planner.plan(classification, query)
        result = resolver.resolve(plan)

        assert 0 <= result.confidence <= 1, f"Confidence out of range for '{query}': {result.confidence}"


def test_get_article_by_url(resolver):
    """Test getting article by exact URL"""
    # Get a real article
    articles = list(resolver.link_index._article_by_id.values())
    if not articles:
        pytest.skip("No articles in index")

    test_article = articles[0]
    url = test_article.url

    # Should find the article
    found = resolver.get_article_by_url(url)
    assert found is not None
    assert found.article_id == test_article.article_id

    # Should not find invalid URL
    not_found = resolver.get_article_by_url("https://example.com/fake.html")
    assert not_found is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

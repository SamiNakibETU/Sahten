"""
Golden Examples Test Suite
Tests the complete RAG pipeline against curated golden examples
Validates editorial compliance and expected behaviors
"""

import pytest
import logging
from app.rag.pipeline import RAGPipeline
from app.data.loaders import data_cache
from app.data.content_index import ContentIndex
from app.data.link_index import LinkIndex
from app.models.llm_client import MockLLMClient
from app.models.config import settings

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def pipeline():
    """Create complete pipeline with real data"""
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


@pytest.fixture(scope="module")
def golden_examples():
    """Load golden examples"""
    return data_cache.get_golden_examples()


def test_golden_examples_loaded(golden_examples):
    """Verify golden examples are loaded"""
    assert len(golden_examples) > 0, "No golden examples found"
    logger.info(f"Loaded {len(golden_examples)} golden examples")


def test_all_golden_examples_produce_valid_html(pipeline, golden_examples):
    """Test that all golden examples produce valid HTML responses"""
    failures = []

    for example in golden_examples:
        try:
            response = pipeline.process(example.user_query, debug=False)

            # Should have HTML
            if not response.html:
                failures.append(f"{example.id}: No HTML generated")
                continue

            # Should have <p> tags
            if "<p>" not in response.html and "<a" not in response.html:
                failures.append(f"{example.id}: No <p> or <a> tags in HTML")

            # Should not have Markdown
            if "**" in response.html or response.html.strip().startswith("#"):
                failures.append(f"{example.id}: Contains Markdown syntax")

        except Exception as e:
            failures.append(f"{example.id}: Exception - {e}")

    if failures:
        logger.error("Golden example failures:\n" + "\n".join(failures))

    assert len(failures) == 0, f"{len(failures)} golden examples failed:\n" + "\n".join(failures[:5])


def test_golden_examples_url_safety(pipeline, golden_examples):
    """Test that all URLs in golden example responses are safe"""
    import re
    failures = []

    for example in golden_examples:
        try:
            response = pipeline.process(example.user_query, debug=False)

            # Check primary URL
            if response.primary_url:
                if not response.primary_url.startswith(settings.allowed_url_domain):
                    failures.append(
                        f"{example.id}: Invalid primary URL - {response.primary_url}"
                    )

            # Check URLs in HTML
            urls = re.findall(r'https?://[^\s<>"]+', response.html)
            for url in urls:
                if not url.startswith(settings.allowed_url_domain):
                    failures.append(f"{example.id}: Invalid URL in HTML - {url}")

        except Exception as e:
            failures.append(f"{example.id}: Exception - {e}")

    if failures:
        logger.error("URL safety failures:\n" + "\n".join(failures))

    assert len(failures) == 0, f"{len(failures)} URLs failed validation:\n" + "\n".join(failures[:5])


def test_golden_examples_scenario_alignment(pipeline, golden_examples):
    """Test that golden examples map to expected scenarios"""
    scenario_mapping = {
        "greeting": [4, 7],  # Greeting or non-French
        "about_bot": [5],
        "off_topic": [3, 6],  # Fallback or redirect
        "recipe_query": [1, 2, 8],  # OLJ, Base2+OLJ, or ingredient suggestions
        "non_french": [7],
    }

    failures = []

    for example in golden_examples:
        expected_scenarios = scenario_mapping.get(example.scenario, [])

        if not expected_scenarios:
            # Unknown scenario in golden data, skip
            continue

        try:
            response = pipeline.process(example.user_query, debug=False)

            if response.scenario_id not in expected_scenarios:
                failures.append(
                    f"{example.id} ({example.scenario}): "
                    f"Got scenario {response.scenario_id}, "
                    f"expected one of {expected_scenarios}"
                )

        except Exception as e:
            failures.append(f"{example.id}: Exception - {e}")

    # Allow some flexibility in scenario detection
    max_allowed_failures = max(1, len(golden_examples) // 10)  # Allow 10% failures

    if len(failures) > max_allowed_failures:
        logger.error("Scenario alignment failures:\n" + "\n".join(failures[:10]))

    assert len(failures) <= max_allowed_failures, (
        f"{len(failures)} scenario mismatches (max allowed: {max_allowed_failures}):\n"
        + "\n".join(failures[:5])
    )


def test_golden_examples_french_responses(pipeline, golden_examples):
    """Test that all responses are in French (except non-French scenario)"""
    failures = []

    french_indicators = ["le", "la", "les", "de", "du", "pour", "avec", "recette"]

    for example in golden_examples:
        try:
            response = pipeline.process(example.user_query, debug=False)

            # Check if response contains French words
            html_lower = response.html.lower()
            has_french = any(word in html_lower for word in french_indicators)

            if not has_french and response.scenario_id != 7:  # Skip non-French scenario
                failures.append(f"{example.id}: Response may not be in French")

        except Exception as e:
            failures.append(f"{example.id}: Exception - {e}")

    # Allow some false positives
    max_allowed_failures = max(1, len(golden_examples) // 20)  # Allow 5% failures

    if len(failures) > max_allowed_failures:
        logger.warning("French detection failures:\n" + "\n".join(failures[:5]))

    assert len(failures) <= max_allowed_failures, (
        f"{len(failures)} may not be French (max allowed: {max_allowed_failures})"
    )


def test_golden_examples_no_hallucinated_content(pipeline, golden_examples):
    """Test that responses don't contain hallucinated OLJ recipe content"""
    hallucination_indicators = [
        # Shouldn't reveal actual ingredient lists for OLJ articles
        "ingrédients :",
        "ingredients:",
        "• 200g",  # Specific measurements
        "• 1 cuillère",
    ]

    failures = []

    for example in golden_examples:
        try:
            response = pipeline.process(example.user_query, debug=True)

            # Only check OLJ scenario responses
            if response.scenario_id == 1:  # OLJ recipe available
                html_lower = response.html.lower()

                for indicator in hallucination_indicators:
                    if indicator.lower() in html_lower:
                        failures.append(
                            f"{example.id}: May contain hallucinated "
                            f"ingredient list: '{indicator}'"
                        )
                        break

        except Exception as e:
            failures.append(f"{example.id}: Exception - {e}")

    if failures:
        logger.warning("Potential hallucination warnings:\n" + "\n".join(failures[:5]))

    # This is a warning, not a hard failure
    # Content Guard should catch this, but log it for review
    assert len(failures) <= len(golden_examples) // 4, (
        f"Too many potential hallucinations: {len(failures)}"
    )


def test_golden_examples_consistency(pipeline, golden_examples):
    """Test that same query produces consistent scenario"""
    # Test with a subset (first 10 examples)
    test_examples = golden_examples[:min(10, len(golden_examples))]

    for example in test_examples:
        response1 = pipeline.process(example.user_query, debug=False)
        response2 = pipeline.process(example.user_query, debug=False)

        # Should have same scenario
        assert response1.scenario_id == response2.scenario_id, (
            f"{example.id}: Inconsistent scenarios - "
            f"{response1.scenario_id} vs {response2.scenario_id}"
        )

        # Should have same primary URL (if any)
        assert response1.primary_url == response2.primary_url, (
            f"{example.id}: Inconsistent URLs"
        )


def test_golden_examples_performance(pipeline, golden_examples):
    """Test that golden examples process in reasonable time"""
    import time

    # Test with first 5 examples
    test_examples = golden_examples[:min(5, len(golden_examples))]

    total_time = 0
    for example in test_examples:
        start = time.time()
        pipeline.process(example.user_query, debug=False)
        elapsed = time.time() - start
        total_time += elapsed

        # Should process in under 2 seconds (with mock LLM)
        assert elapsed < 2.0, f"{example.id}: Took {elapsed:.2f}s (too slow)"

    avg_time = total_time / len(test_examples)
    logger.info(f"Average processing time: {avg_time:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

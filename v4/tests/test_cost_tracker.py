"""Tests du suivi de coûts API."""

from backend.app.cost_tracker import (
    RequestCostAccumulator,
    cost_tracker_scope,
    estimate_cohere_search_units,
    get_request_cost,
)


def test_openai_chat_cost_gpt41_mini() -> None:
    acc = RequestCostAccumulator()
    acc.record_openai_chat(
        "query_analyzer",
        model="gpt-4.1-mini",
        prompt_tokens=2000,
        completion_tokens=150,
    )
    # 2000*0.40 + 150*1.60 = 800 + 240 = 1040 / 1M
    assert acc.openai_chat["query_analyzer"]["usd"] == 0.00104
    assert acc.estimated_usd == 0.00104


def test_openai_embedding_cost() -> None:
    acc = RequestCostAccumulator()
    acc.record_openai_embedding(model="text-embedding-3-small", tokens=1000)
    assert acc.openai_embeddings["usd"] == 0.00002
    assert acc.estimated_usd == 0.00002


def test_cohere_rerank_search_units() -> None:
    docs = ["x" * 2000] * 120  # ~500 tok each -> 120 counted docs -> 2 search units
    assert estimate_cohere_search_units(docs) == 2


def test_cohere_rerank_cost() -> None:
    acc = RequestCostAccumulator()
    acc.record_cohere_rerank(
        "chunk_rerank",
        model="rerank-multilingual-v3.0",
        documents=["doc"] * 50,
        search_units=1,
    )
    assert acc.cohere_rerank["chunk_rerank"]["usd"] == 0.002
    assert acc.estimated_usd == 0.002


def test_cost_tracker_scope_isolated() -> None:
    assert get_request_cost() is None
    with cost_tracker_scope() as acc:
        assert get_request_cost() is acc
        acc.record_openai_chat(
            "generation",
            model="gpt-4.1-mini",
            prompt_tokens=100,
            completion_tokens=50,
        )
    assert get_request_cost() is None


def test_to_dict_includes_breakdown() -> None:
    acc = RequestCostAccumulator()
    acc.record_openai_chat(
        "generation",
        model="gpt-4.1-mini",
        prompt_tokens=5000,
        completion_tokens=600,
    )
    acc.record_openai_embedding(model="text-embedding-3-small", tokens=30)
    acc.record_cohere_rerank(
        "article_rerank",
        model="rerank-multilingual-v3.0",
        documents=["a", "b"],
        search_units=1,
    )
    d = acc.to_dict()
    assert "estimated_usd" in d
    assert d["openai_chat"]["generation"]["prompt_tokens"] == 5000
    assert d["openai_embeddings"]["tokens"] == 30
    assert d["cohere_rerank"]["article_rerank"]["search_units"] == 1
    assert d["estimated_usd"] > 0

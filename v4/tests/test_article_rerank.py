from backend.app.rag.pipeline import _apply_article_rerank, _article_rerank_documents
from backend.app.rag.reranker import RerankedHit
from backend.app.rag.retriever import Hit


def _mk_hit(
    *,
    chunk_id: int,
    article_external_id: int,
    title: str,
    section_kind: str,
    text: str,
) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        article_id=article_external_id,
        article_external_id=article_external_id,
        article_title=title,
        article_url=f"https://example.com/{article_external_id}",
        cover_image_url=None,
        section_kind=section_kind,
        chunk_text=text,
        score_lex=0.0,
        score_vec=0.0,
        score_rrf=0.0,
        metadata={},
    )


def test_article_rerank_documents_group_by_article() -> None:
    reranked = [
        RerankedHit(
            hit=_mk_hit(
                chunk_id=1,
                article_external_id=100,
                title="Article A",
                section_kind="recipe_summary",
                text="A summary",
            ),
            rerank_score=0.9,
        ),
        RerankedHit(
            hit=_mk_hit(
                chunk_id=2,
                article_external_id=100,
                title="Article A",
                section_kind="recipe_meta",
                text="A meta",
            ),
            rerank_score=0.8,
        ),
        RerankedHit(
            hit=_mk_hit(
                chunk_id=3,
                article_external_id=200,
                title="Article B",
                section_kind="recipe_summary",
                text="B summary",
            ),
            rerank_score=0.7,
        ),
    ]
    docs = _article_rerank_documents(
        reranked, max_articles=10, max_chunks_per_article=2
    )
    assert len(docs) == 2
    assert docs[0].article_external_id == 100
    assert "[recipe_summary] A summary" in docs[0].chunk_text
    assert "[recipe_meta] A meta" in docs[0].chunk_text


def test_apply_article_rerank_reorders_chunks_by_article_rank() -> None:
    base = [
        RerankedHit(
            hit=_mk_hit(
                chunk_id=10,
                article_external_id=100,
                title="Article A",
                section_kind="recipe_summary",
                text="A",
            ),
            rerank_score=0.95,
        ),
        RerankedHit(
            hit=_mk_hit(
                chunk_id=20,
                article_external_id=200,
                title="Article B",
                section_kind="recipe_summary",
                text="B",
            ),
            rerank_score=0.90,
        ),
    ]
    article_ranked = [
        RerankedHit(
            hit=_mk_hit(
                chunk_id=20,
                article_external_id=200,
                title="Article B",
                section_kind="article_doc",
                text="B doc",
            ),
            rerank_score=0.99,
        ),
        RerankedHit(
            hit=_mk_hit(
                chunk_id=10,
                article_external_id=100,
                title="Article A",
                section_kind="article_doc",
                text="A doc",
            ),
            rerank_score=0.80,
        ),
    ]
    out = _apply_article_rerank(base, article_ranked, keep_top_articles=2)
    assert [r.hit.article_external_id for r in out] == [200, 100]

from backend.app.rag.pipeline import _post_rank_recipe_proximity
from backend.app.rag.reranker import RerankedHit
from backend.app.rag.retriever import Hit


def _mk_hit(
    *,
    title: str,
    text: str,
    section_kind: str = "recipe_summary",
    article_external_id: int = 1,
) -> Hit:
    return Hit(
        chunk_id=article_external_id,
        article_id=article_external_id,
        article_external_id=article_external_id,
        article_title=title,
        article_url="https://example.com/article",
        cover_image_url=None,
        section_kind=section_kind,
        chunk_text=text,
        score_lex=0.0,
        score_vec=0.0,
        score_rrf=0.0,
        metadata={},
    )


def test_post_rank_recipe_proximity_prefers_savory_for_couscous() -> None:
    cookie = RerankedHit(
        hit=_mk_hit(
            title="Cookies à la mode libanaise",
            text="Dessert sucré au chocolat.",
            section_kind="dessert",
            article_external_id=10,
        ),
        rerank_score=0.95,
    )
    savory = RerankedHit(
        hit=_mk_hit(
            title="Moghrabieh traditionnel",
            text="Plat salé à base de semoule et bouillon.",
            section_kind="recipe_meta",
            article_external_id=11,
        ),
        rerank_score=0.72,
    )

    out = _post_rank_recipe_proximity("recette de couscous", [cookie, savory])
    assert out[0].hit.article_external_id == 11


def test_post_rank_recipe_proximity_noop_for_other_queries() -> None:
    first = RerankedHit(
        hit=_mk_hit(
            title="Cookies à la mode libanaise",
            text="Dessert sucré",
            article_external_id=20,
        ),
        rerank_score=0.9,
    )
    second = RerankedHit(
        hit=_mk_hit(
            title="Moghrabieh traditionnel",
            text="Plat salé",
            article_external_id=21,
        ),
        rerank_score=0.8,
    )

    out = _post_rank_recipe_proximity("qui est Carla Rebeiz ?", [first, second])
    assert [h.hit.article_external_id for h in out] == [20, 21]

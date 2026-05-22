from backend.app.llm.response_generator import CARNETS_PHRASE
from backend.app.rag.pipeline import (
    _build_base2_last_resort_answer,
    _match_base2_recipe,
)
from backend.app.rag.reranker import RerankedHit
from backend.app.rag.retriever import Hit


def _mk_hit(
    *,
    aid: int,
    title: str,
    url: str,
    chunk_id: int,
    section_kind: str = "recipe_summary",
    score: float = 0.7,
) -> RerankedHit:
    return RerankedHit(
        hit=Hit(
            chunk_id=chunk_id,
            article_id=aid,
            article_external_id=aid,
            article_title=title,
            article_url=url,
            cover_image_url=None,
            section_kind=section_kind,
            chunk_text=title,
            score_lex=0.0,
            score_vec=0.0,
            score_rrf=0.0,
            metadata={},
        ),
        rerank_score=score,
    )


def test_match_base2_recipe_exact_houmous() -> None:
    recipe = _match_base2_recipe("recette hoummous")
    assert recipe is not None
    assert recipe.get("name") == "Houmous"


def test_build_base2_last_resort_answer_includes_olj_suggestion() -> None:
    recipe = _match_base2_recipe("recette houmous")
    assert recipe is not None
    reranked = [
        _mk_hit(
            aid=2,
            title="Le hommos au cumin",
            url="https://www.lorientlejour.com/cuisine-liban-a-table/1495322/le-hommos.html",
            chunk_id=20,
            section_kind="recipe_meta",
        )
    ]
    out = _build_base2_last_resort_answer(
        user_query="recette houmous",
        base2_recipe=recipe,
        reranked=reranked,
    )
    assert out.answer_sentences[0].text == CARNETS_PHRASE
    assert any("Ingrédients (résumé)" in s.text for s in out.answer_sentences)
    assert out.recipe_card is not None
    assert out.recipe_card.title == "Le hommos au cumin"

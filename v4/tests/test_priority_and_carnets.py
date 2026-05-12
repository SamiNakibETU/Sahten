from backend.app.llm.response_generator import (
    GroundedAnswer,
    GroundedSentence,
    _enforce_carnets_phrase,
)
from backend.app.rag.pipeline import _apply_source_priority
from backend.app.rag.reranker import RerankedHit
from backend.app.rag.retriever import Hit


def _mk_hit(
    *,
    aid: int,
    title: str,
    url: str,
    chunk_id: int,
    score: float,
) -> RerankedHit:
    hit = Hit(
        chunk_id=chunk_id,
        article_id=aid,
        article_external_id=aid,
        article_title=title,
        article_url=url,
        cover_image_url=None,
        section_kind="recipe_summary",
        chunk_text=title,
        score_lex=0.0,
        score_vec=0.0,
        score_rrf=0.0,
        metadata={},
    )
    return RerankedHit(hit=hit, rerank_score=score)


def test_source_priority_prefers_canonical_olj_by_default() -> None:
    non_canonical = _mk_hit(
        aid=1,
        title="Recette externe",
        url="https://external.example/recette",
        chunk_id=10,
        score=0.99,
    )
    canonical = _mk_hit(
        aid=2,
        title="Le hommos au cumin",
        url="https://www.lorientlejour.com/cuisine-liban-a-table/1495322/le-hommos.html",
        chunk_id=20,
        score=0.70,
    )
    out = _apply_source_priority([non_canonical, canonical], "recette houmous")
    assert out[0].hit.article_external_id == 2


def test_enforce_carnets_phrase_on_recipe_refusal() -> None:
    answer = GroundedAnswer(
        answer_sentences=[
            GroundedSentence(
                text="Je n'ai pas de fiche de houmous dans les extraits consultés.",
                source_chunk_ids=[],
            )
        ],
        follow_up="Souhaitez-vous une alternative ?",
        confidence=0.2,
    )
    out = _enforce_carnets_phrase(answer, "recette houmous")
    assert out.answer_sentences[0].text == "Désolé, je n'ai pas cette recette dans mes carnets"

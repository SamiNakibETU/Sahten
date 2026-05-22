from backend.app.llm.response_generator import (
    CARNETS_PHRASE,
    GroundedAnswer,
    GroundedSentence,
    RecipeCard,
    _enforce_carnets_phrase,
    _polish_user_facing_tone,
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
    assert out.answer_sentences[0].text == CARNETS_PHRASE


def test_polish_user_facing_tone_rewrites_meta_pivot() -> None:
    answer = GroundedAnswer(
        answer_sentences=[
            GroundedSentence(text=CARNETS_PHRASE, source_chunk_ids=[]),
            GroundedSentence(
                text=(
                    "La fiche « La Mouloukhiyé de Tara Khattar » est celle qui "
                    "se rapproche le plus de votre demande parmi les extraits "
                    "disponibles."
                ),
                source_chunk_ids=[1],
            ),
        ],
        recipe_card=RecipeCard(
            title="La Mouloukhiyé de Tara Khattar",
            chef="Tara Khattar",
            duration_min=None,
            serves=None,
            ingredients=[],
            steps=[],
            source_chunk_ids=[1],
        ),
        follow_up="Souhaitez-vous aussi ouvrir la fiche « Autre plat » ?",
        confidence=0.4,
    )
    out = _polish_user_facing_tone(answer)
    pivot = out.answer_sentences[1].text.lower()
    assert "fiche" not in pivot
    assert "extrait" not in pivot
    assert "mouloukhiyé" in pivot
    assert "je peux aussi vous proposer" in out.follow_up.lower()
    assert "autre plat" in out.follow_up.lower()

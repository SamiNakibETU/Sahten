from backend.app.llm.response_generator import (
    ChefCard,
    GroundedAnswer,
    GroundedSentence,
    RecipeCard,
)
from backend.app.rag.html_renderer import render_answer_html
from backend.app.rag.reranker import RerankedHit
from backend.app.rag.retriever import Hit


def _mk_hit() -> RerankedHit:
    hit = Hit(
        chunk_id=1,
        article_id=10,
        article_external_id=999,
        article_title="Les atayefs bil achta d'Andrée Maalouf",
        article_url="https://www.lorientlejour.com/cuisine-liban-a-table/1504187/les-atayefs-bil-achta-dandree-maalouf.html",
        cover_image_url=None,
        section_kind="recipe_summary",
        chunk_text="Atayefs bil achta, dessert libanais.",
        score_lex=1.0,
        score_vec=1.0,
        score_rrf=1.0,
        metadata={"featured_chef": "Andrée Maalouf"},
    )
    return RerankedHit(hit=hit, rerank_score=0.9)


def test_render_answer_html_avoids_duplicate_chef_link_when_same_article() -> None:
    rh = _mk_hit()
    answer = GroundedAnswer(
        answer_sentences=[
            GroundedSentence(text="Voici une proposition.", source_chunk_ids=[1])
        ],
        recipe_card=RecipeCard(
            title="Les atayefs bil achta d'Andrée Maalouf",
            chef="Andrée Maalouf",
            source_chunk_ids=[1],
        ),
        chef_card=ChefCard(
            name="Andrée Maalouf",
            biography="Andrée Maalouf, entre tradition et modernité.",
            works=["Les atayefs bil achta d'Andrée Maalouf"],
            source_chunk_ids=[1],
        ),
        follow_up="Souhaitez-vous une autre suggestion ?",
        confidence=0.9,
    )
    html = render_answer_html(answer, [rh])
    assert html.count("https://www.lorientlejour.com/cuisine-liban-a-table/1504187/les-atayefs-bil-achta-dandree-maalouf.html") == 1

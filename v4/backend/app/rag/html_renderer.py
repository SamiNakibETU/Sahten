"""Rendu HTML d'une réponse RAG v4 vers le format attendu par le widget v3.

Le frontend `frontend/js/sahten.js` consomme un champ `html` qu'il sanitize via
DOMPurify. On reformate ici la sortie structurée du pipeline (`GroundedAnswer`
+ sources rerankées) en HTML lisible et accessible, sans dépendance externe
(pas de Jinja, pas de markdown — escape manuel pour rester safe).
"""

from __future__ import annotations

import html as _html

from ..llm.response_generator import ChefCard, GroundedAnswer, RecipeCard
from .reranker import RerankedHit


def _article_url_for_chunks(
    chunk_ids: list[int], hits: list[RerankedHit]
) -> tuple[str | None, str | None]:
    """URL + titre d’article à partir des chunk_ids de carte (ou 1er hit en secours)."""
    want = set(chunk_ids)
    for h in hits:
        if h.hit.chunk_id in want:
            return h.hit.article_url, h.hit.article_title
    if hits:
        h0 = hits[0]
        return h0.hit.article_url, h0.hit.article_title
    return None, None


def _escape(value: str | None) -> str:
    if value is None:
        return ""
    return _html.escape(str(value), quote=True)


def _render_recipe(card: RecipeCard, *, article_url: str | None, article_title: str | None) -> str:
    parts: list[str] = ['<article class="sahten-recipe-card">']
    parts.append(f'<header><h2>{_escape(card.title)}</h2>')
    meta_bits: list[str] = []
    if card.chef:
        meta_bits.append(f"par <strong>{_escape(card.chef)}</strong>")
    if card.duration_min:
        meta_bits.append(f"{int(card.duration_min)} min")
    if card.serves:
        meta_bits.append(f"pour {_escape(card.serves)}")
    if meta_bits:
        parts.append(f'<p class="sahten-recipe-meta">{" · ".join(meta_bits)}</p>')
    parts.append("</header>")

    if article_url:
        label = _escape(article_title or "Lire l’article sur L’Orient-Le Jour")
        parts.append(
            f'<p class="sahten-recipe-link"><a href="{_escape(article_url)}" '
            f'target="_blank" rel="noopener noreferrer">{label}</a></p>'
        )
        parts.append(
            '<p class="sahten-recipe-teaser">Les ingrédients détaillés et la '
            "préparation pas à pas se trouvent dans la fiche sur "
            "<strong>L’Orient-Le Jour</strong> — ouvrez le lien ci-dessus.</p>"
        )
    else:
        parts.append(
            '<p class="sahten-recipe-teaser"><em>Consultez les sources en bas de '
            "réponse pour retrouver la recette complète.</em></p>"
        )

    parts.append("</article>")
    return "".join(parts)


def _render_chef(card: ChefCard, *, article_url: str | None, article_title: str | None) -> str:
    parts: list[str] = ['<article class="sahten-chef-card">']
    parts.append(f"<header><h2>{_escape(card.name)}</h2>")
    if card.role:
        parts.append(f'<p class="sahten-chef-role">{_escape(card.role)}</p>')
    parts.append("</header>")
    if article_url:
        label = _escape(article_title or "Lire sur L’Orient-Le Jour")
        parts.append(
            f'<p class="sahten-chef-link"><a href="{_escape(article_url)}" '
            f'target="_blank" rel="noopener noreferrer">{label}</a></p>'
        )
    if card.biography:
        parts.append(f"<p>{_escape(card.biography)}</p>")
    if card.works:
        parts.append('<section><h3>Œuvres / titres</h3><ul>')
        for w in card.works:
            parts.append(f"<li>{_escape(w)}</li>")
        parts.append("</ul></section>")
    parts.append("</article>")
    return "".join(parts)


def _render_sources(hits: list[RerankedHit], used_chunk_ids: set[int]) -> str:
    """Liste compacte des articles cités, dédupliqués par article_external_id.

    Si des `used_chunk_ids` sont fournis mais qu’aucun hit ne correspond (IDs
    incohérents entre phrases et rerank), on retombe sur les meilleurs hits pour
    garder au moins un lien cliquable.
    """
    if not hits:
        return ""
    seen: dict[int, RerankedHit] = {}
    for h in hits:
        if used_chunk_ids and h.hit.chunk_id not in used_chunk_ids:
            continue
        key = h.hit.article_external_id
        if key in seen:
            continue
        seen[key] = h
    if not seen and used_chunk_ids:
        for h in hits:
            key = h.hit.article_external_id
            if key in seen:
                continue
            seen[key] = h
            if len(seen) >= 5:
                break
    if not seen:
        return ""
    parts: list[str] = ['<section class="sahten-sources"><h3>Sources L\'Orient-Le Jour</h3><ul>']
    for h in seen.values():
        url = _escape(h.hit.article_url)
        title = _escape(h.hit.article_title)
        parts.append(
            f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a></li>'
        )
    parts.append("</ul></section>")
    return "".join(parts)


def render_answer_html(
    answer: GroundedAnswer,
    hits: list[RerankedHit],
    *,
    follow_up_override: str | None = None,
) -> str:
    """Construit l'HTML final à afficher dans le widget chat.

    - Phrases du `answer.answer_sentences` concaténées en paragraphes.
    - `recipe_card` ou `chef_card` rendues si présentes.
    - Section sources (déduplication par article).
    - `follow_up` proposé en italique en bas.
    """
    used_ids: set[int] = set()
    sentences_html: list[str] = []
    for sent in answer.answer_sentences:
        used_ids.update(sent.source_chunk_ids)
        sentences_html.append(_escape(sent.text))
    if answer.recipe_card is not None:
        used_ids.update(answer.recipe_card.source_chunk_ids)
    if answer.chef_card is not None:
        used_ids.update(answer.chef_card.source_chunk_ids)

    recipe_url: str | None = None
    recipe_link_title: str | None = None
    chef_url: str | None = None
    chef_link_title: str | None = None
    if answer.recipe_card is not None:
        recipe_url, recipe_link_title = _article_url_for_chunks(
            answer.recipe_card.source_chunk_ids, hits
        )
    if answer.chef_card is not None:
        chef_url, chef_link_title = _article_url_for_chunks(
            answer.chef_card.source_chunk_ids, hits
        )

    parts: list[str] = ['<div class="sahten-narrative">']
    if sentences_html:
        parts.append("<p>" + " ".join(sentences_html) + "</p>")
    else:
        parts.append(
            "<p><em>Je n'ai rien trouvé d'assez solide dans les archives pour vous "
            "répondre. Pourriez-vous reformuler ou préciser un chef, un plat ou "
            "un ingrédient ?</em></p>"
        )

    if answer.recipe_card is not None:
        parts.append(
            _render_recipe(
                answer.recipe_card,
                article_url=recipe_url,
                article_title=recipe_link_title,
            )
        )
    if answer.chef_card is not None:
        parts.append(
            _render_chef(
                answer.chef_card,
                article_url=chef_url,
                article_title=chef_link_title,
            )
        )

    parts.append(_render_sources(hits, used_ids))

    follow = follow_up_override if follow_up_override is not None else (answer.follow_up or "")
    if follow.strip():
        parts.append(
            f'<p class="sahten-followup"><em>{_escape(follow.strip())}</em></p>'
        )

    parts.append("</div>")
    return "".join(parts)


__all__ = ["render_answer_html"]

"""Rendu HTML d'une réponse RAG v4 vers le format attendu par le widget v3.

Le frontend `frontend/js/sahten.js` consomme un champ `html` qu'il sanitize via
DOMPurify. On reformate ici la sortie structurée du pipeline (`GroundedAnswer`
+ sources rerankées) en HTML lisible et accessible, sans dépendance externe
(pas de Jinja, pas de markdown — escape manuel pour rester safe).
"""

from __future__ import annotations

import html as _html
from typing import Any
from urllib.parse import urlparse, urlunparse

from ..llm.response_generator import ChefCard, GroundedAnswer, RecipeCard
from .reranker import RerankedHit
from .retriever import Hit


def _int_id(raw: int | float) -> int:
    """Évite les mismatches d’appartenance (ex. numpy / types SQL)."""
    return int(raw)


def _norm_article_title(title: str | None) -> str:
    if not title:
        return ""
    return " ".join(str(title).strip().lower().split())


def _norm_article_url(url: str | None) -> str | None:
    """Clé de comparaison stable (schéma / hôte / chemin, sans fragment)."""
    if not url:
        return None
    p = urlparse(str(url).strip())
    if not p.scheme or not p.netloc:
        return str(url).strip().rstrip("/")
    path = (p.path or "").rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", "", ""))


def _resolve_card_article(
    chunk_ids: list[int], hits: list[RerankedHit]
) -> tuple[str | None, str | None, set[int]]:
    """URL + titre affichés sur la carte recette/chef + IDs à ne pas répéter en Sources.

    Aligné sur un seul parcours : d’abord premier hit dont le chunk est cité par la carte,
    sinon secours sur le premier hit reranké (même logique qu’avant, mais IDs toujours cohérents).
    """
    want = set(chunk_ids)
    ids_from_chunks: set[int] = {
        _int_id(h.hit.article_external_id) for h in hits if h.hit.chunk_id in want
    }
    for h in hits:
        if h.hit.chunk_id in want:
            hh = h.hit
            return hh.article_url, hh.article_title, ids_from_chunks
    if hits:
        hh = hits[0].hit
        return hh.article_url, hh.article_title, {_int_id(hh.article_external_id)}
    return None, None, set()


def _escape(value: str | None) -> str:
    if value is None:
        return ""
    return _html.escape(str(value), quote=True)


def _safe_http_image_url(url: str | None) -> str | None:
    """Vignette article : uniquement http(s), jamais javascript:/data:."""
    if not url:
        return None
    s = str(url).strip()
    if s.startswith("//"):
        s = "https:" + s
    if s.lower().startswith(("https://", "http://")):
        return s
    return None


def _absolute_image_url(raw: str | None, article_url: str | None) -> str | None:
    """Absolu https ou chemin relatif résolu avec le domaine de l’article."""
    u = _safe_http_image_url(raw)
    if u:
        return u
    if not raw or not article_url:
        return None
    s = str(raw).strip()
    if not s.startswith("/"):
        return None
    p = urlparse(str(article_url).strip())
    if not p.scheme or not p.netloc:
        return None
    return _safe_http_image_url(f"{p.scheme}://{p.netloc}{s}")


def _cover_url_from_primary_hit(
    primary_hit: RerankedHit | None,
    article_url: str | None,
) -> str | None:
    """Couverture SQL + métadonnées chunk ; URLs relatives OLJ via l’URL article."""
    if not primary_hit:
        return None
    h: Hit = primary_hit.hit
    candidates: list[str | None] = [h.cover_image_url]
    meta = h.metadata
    if isinstance(meta, dict):
        for key in ("cover_image_url", "image_url", "thumb_url", "og_image"):
            v = meta.get(key)
            if isinstance(v, str):
                candidates.append(v)
    for raw in candidates:
        u = _absolute_image_url(raw, article_url)
        if u:
            return u
    return None


def _chef_from_chunk_metadata(meta: Any) -> str | None:
    if not isinstance(meta, dict):
        return None
    for key in ("featured_chef", "chef_name", "chef", "author_name", "author"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _recipe_category_label(section_kind: str) -> str:
    """Libellé type « PLAT PRINCIPAL » à partir du kind de chunk."""
    k = (section_kind or "").strip().lower()
    if k in ("recipe_steps", "main_course", "plat", "plat_principal"):
        return "PLAT PRINCIPAL"
    if "dessert" in k:
        return "DESSERT"
    if "entree" in k or "entrée" in k or "starter" in k:
        return "ENTRÉE"
    if "salad" in k or "salade" in k:
        return "SALADE"
    if "ingredient" in k:
        return "RECETTE"
    return "RECETTE"


def _primary_hit_for_card(chunk_ids: list[int], hits: list[RerankedHit]) -> RerankedHit | None:
    """Premier hit reranké aligné sur la carte (même logique que la résolution d’URL)."""
    want = set(chunk_ids)
    for h in hits:
        if h.hit.chunk_id in want:
            return h
    return hits[0] if hits else None


def _render_recipe(
    card: RecipeCard,
    *,
    article_url: str | None,
    article_title: str | None,
    primary_hit: RerankedHit | None,
) -> str:
    """Carte type aperçu OLJ : vignette + bandeau marque, titre unique, lien sur toute la carte."""
    cover = _cover_url_from_primary_hit(primary_hit, article_url)
    sk = primary_hit.hit.section_kind if primary_hit else ""
    cat = _recipe_category_label(sk)
    display_alt = article_title or card.title

    parts: list[str] = [
        '<article class="sahten-recipe-card sahten-recipe-card--preview">'
    ]
    if article_url:
        parts.append(
            f'<a class="sahten-recipe-card__link" href="{_escape(article_url)}" '
            'target="_blank" rel="noopener noreferrer">'
        )
    else:
        parts.append('<div class="sahten-recipe-card__nolink">')

    if cover:
        parts.append(
            '<div class="sahten-recipe-card__media">'
            f'<img src="{_escape(cover)}" alt="{_escape(display_alt)}" '
            'loading="lazy" decoding="async" width="120" height="120" />'
            "</div>"
        )
    else:
        parts.append(
            '<div class="sahten-recipe-card__media sahten-recipe-card__media--placeholder" '
            'aria-hidden="true"></div>'
        )

    parts.append('<div class="sahten-recipe-card__body">')
    parts.append(f'<p class="sahten-recipe-card__category">{_escape(cat)}</p>')
    parts.append(f'<h2 class="sahten-recipe-card__title">{_escape(card.title)}</h2>')
    if card.chef:
        parts.append(
            f'<p class="sahten-recipe-card__byline">Par {_escape(card.chef)}</p>'
        )
    meta_bits: list[str] = []
    if card.duration_min:
        meta_bits.append(f"{int(card.duration_min)} min")
    if card.serves:
        meta_bits.append(f"pour {_escape(card.serves)}")
    if meta_bits:
        parts.append(
            f'<p class="sahten-recipe-card__meta">{" · ".join(meta_bits)}</p>'
        )
    parts.append("</div>")

    if article_url:
        parts.append("</a>")
    else:
        parts.append("</div>")

    if not article_url:
        parts.append(
            '<p class="sahten-recipe-teaser"><em>Consultez les sources en bas de '
            "réponse pour retrouver la recette complète.</em></p>"
        )

    parts.append("</article>")
    return "".join(parts)


def _render_source_article_card(rh: RerankedHit) -> str:
    """Même carte aperçu que la recette principale, pour chaque source citée."""
    h = rh.hit
    chef = _chef_from_chunk_metadata(h.metadata)
    card = RecipeCard(
        title=h.article_title or "Article",
        chef=chef,
        duration_min=None,
        serves=None,
        ingredients=[],
        steps=[],
        source_chunk_ids=[h.chunk_id],
    )
    return _render_recipe(
        card,
        article_url=h.article_url or None,
        article_title=h.article_title,
        primary_hit=rh,
    )


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


def _render_sources(
    hits: list[RerankedHit],
    used_chunk_ids: set[int],
    *,
    skip_article_external_ids: set[int] | None = None,
    skip_article_urls: set[str] | None = None,
    skip_article_title_norms: set[str] | None = None,
) -> str:
    """Liste compacte des articles cités, dédupliqués par article_external_id.

    `skip_article_external_ids` : articles déjà liés dans la carte recette / chef
    (évite le même lien deux fois avec un style différent).
    `skip_article_urls` : filet de sécurité (même URL que le lien principal).
    `skip_article_title_norms` : filet si même titre / URL légerement différente en base.

    Si des `used_chunk_ids` sont fournis mais qu’aucun hit ne correspond (IDs
    incohérents entre phrases et rerank), on retombe sur les meilleurs hits pour
    garder au moins un lien cliquable.
    """
    skip = {_int_id(x) for x in (skip_article_external_ids or set())}
    skip_urls = skip_article_urls or set()
    skip_titles = skip_article_title_norms or set()
    if not hits:
        return ""
    # Aucun chunk cité dans la réponse structurée : ne pas lister les hits RAG
    # (sinon 4–5 articles hors sujet quand le modèle dit « rien trouvé »).
    if not used_chunk_ids:
        return ""
    seen: dict[int, RerankedHit] = {}
    for h in hits:
        if used_chunk_ids and h.hit.chunk_id not in used_chunk_ids:
            continue
        key = _int_id(h.hit.article_external_id)
        if key in skip:
            continue
        nu = _norm_article_url(h.hit.article_url)
        if nu and nu in skip_urls:
            continue
        tnorm = _norm_article_title(h.hit.article_title)
        if tnorm and tnorm in skip_titles:
            continue
        if key in seen:
            continue
        seen[key] = h
    if not seen and used_chunk_ids:
        for h in hits:
            key = _int_id(h.hit.article_external_id)
            if key in skip:
                continue
            nu = _norm_article_url(h.hit.article_url)
            if nu and nu in skip_urls:
                continue
            tnorm = _norm_article_title(h.hit.article_title)
            if tnorm and tnorm in skip_titles:
                continue
            if key in seen:
                continue
            seen[key] = h
            if len(seen) >= 5:
                break
    if not seen:
        return ""
    parts: list[str] = [
        '<section class="sahten-sources sahten-sources--cards">'
        '<h3>Sources L\'Orient-Le Jour</h3>'
        '<div class="sahten-sources__grid" role="list">'
    ]
    for rh in seen.values():
        parts.append(
            '<div class="sahten-sources__item" role="listitem">'
            f"{_render_source_article_card(rh)}"
            "</div>"
        )
    parts.append("</div></section>")
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
    if answer.recipe_card_secondary is not None:
        used_ids.update(answer.recipe_card_secondary.source_chunk_ids)
    if answer.chef_card is not None:
        used_ids.update(answer.chef_card.source_chunk_ids)

    recipe_url: str | None = None
    recipe_link_title: str | None = None
    primary_recipe_hit: RerankedHit | None = None
    recipe_url_secondary: str | None = None
    recipe_link_title_secondary: str | None = None
    primary_recipe_hit_secondary: RerankedHit | None = None
    chef_url: str | None = None
    chef_link_title: str | None = None
    skip_sources_ids: set[int] = set()
    skip_sources_urls: set[str] = set()
    skip_sources_title_norms: set[str] = set()
    if answer.recipe_card is not None:
        primary_recipe_hit = _primary_hit_for_card(
            answer.recipe_card.source_chunk_ids, hits
        )
        recipe_url, recipe_link_title, rids = _resolve_card_article(
            answer.recipe_card.source_chunk_ids, hits
        )
        skip_sources_ids |= rids
        nu = _norm_article_url(recipe_url)
        if nu:
            skip_sources_urls.add(nu)
        tn = _norm_article_title(recipe_link_title)
        if tn:
            skip_sources_title_norms.add(tn)
    if answer.recipe_card_secondary is not None:
        primary_recipe_hit_secondary = _primary_hit_for_card(
            answer.recipe_card_secondary.source_chunk_ids, hits
        )
        recipe_url_secondary, recipe_link_title_secondary, rids2 = _resolve_card_article(
            answer.recipe_card_secondary.source_chunk_ids, hits
        )
        skip_sources_ids |= rids2
        nu2 = _norm_article_url(recipe_url_secondary)
        if nu2:
            skip_sources_urls.add(nu2)
        tn2 = _norm_article_title(recipe_link_title_secondary)
        if tn2:
            skip_sources_title_norms.add(tn2)
    if answer.chef_card is not None:
        chef_url, chef_link_title, cids = _resolve_card_article(
            answer.chef_card.source_chunk_ids, hits
        )
        skip_sources_ids |= cids
        nu = _norm_article_url(chef_url)
        if nu:
            skip_sources_urls.add(nu)
        tn = _norm_article_title(chef_link_title)
        if tn:
            skip_sources_title_norms.add(tn)

    parts: list[str] = ['<div class="sahten-narrative">']
    follow_pre = (follow_up_override if follow_up_override is not None else (answer.follow_up or "")).strip()
    has_cards = (
        answer.recipe_card is not None
        or answer.recipe_card_secondary is not None
        or answer.chef_card is not None
    )
    if sentences_html:
        parts.append("<p>" + " ".join(sentences_html) + "</p>")
    elif has_cards or follow_pre:
        # Le modèle a pu filtrer les phrases (citations invalides) mais proposer
        # encore une carte ou une relance — éviter le message d'échec générique.
        parts.append(
            "<p><em>Voici une piste issue des archives ; le détail est dans la fiche ci-dessous.</em></p>"
        )
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
                primary_hit=primary_recipe_hit,
            )
        )
    if answer.recipe_card_secondary is not None:
        parts.append(
            _render_recipe(
                answer.recipe_card_secondary,
                article_url=recipe_url_secondary,
                article_title=recipe_link_title_secondary,
                primary_hit=primary_recipe_hit_secondary,
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

    # Cartes recette avec liens : ne pas dupliquer en liste « Sources » si chaque
    # carte affichée a une URL résolue.
    recipe_cards_cover = (
        (answer.recipe_card is None or bool(recipe_url))
        and (answer.recipe_card_secondary is None or bool(recipe_url_secondary))
        and (
            answer.recipe_card is not None
            or answer.recipe_card_secondary is not None
        )
    )
    if not recipe_cards_cover:
        parts.append(
            _render_sources(
                hits,
                used_ids,
                skip_article_external_ids=skip_sources_ids,
                skip_article_urls=skip_sources_urls,
                skip_article_title_norms=skip_sources_title_norms,
            )
        )

    if follow_pre:
        parts.append(
            f'<p class="sahten-followup"><em>{_escape(follow_pre)}</em></p>'
        )

    parts.append("</div>")
    return "".join(parts)


__all__ = ["render_answer_html"]

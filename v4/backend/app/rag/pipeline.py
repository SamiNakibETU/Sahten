"""Pipeline RAG complet de bout-en-bout.

    user_query
        -> QueryAnalyzer (LLM) + SessionFocus (LLM, si historique)  # plan + fil
        -> HybridRetriever (pgvector + tsvector)   # RRF, élargissement corpus
        -> Reranker (Cohere ou BGE local)          # cross-encoder
        -> ResponseGenerator (LLM)                  # réponse + grounding
        -> validate_grounding                      # filtre des phrases non sourcées
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..cost_tracker import cost_tracker_scope, get_request_cost
from .. import sessions as sessions_store  # noqa: E402 — import circulaire évité par lazy
from ..llm.response_generator import (
    CARNETS_PHRASE,
    GroundedAnswer,
    GroundedSentence,
    RecipeCard,
    ResponseGenerator,
    _chef_name_from_metadata,
    validate_grounding,
)
from ..llm.models_config import resolve_llm_model
from ..llm.query_understanding import QueryAnalyzer, QueryPlan
from ..llm.session_focus import SessionFocus, SessionFocusAnalyzer
from ..settings import get_settings
from .embeddings import OpenAIEmbeddings
from .ingredient_match import (
    filter_hits_by_ingredient_slugs,
    filter_hits_by_ingredient_text,
    filter_reranked_by_ingredient_slugs,
    rerank_by_ingredient_centrality,
    slug_search_terms,
    supplement_ingredient_slugs,
    wants_another_recipe,
    ingredient_display_name,
)
from .recipe_generator import get_or_generate_recipe
from .reranker import Reranker, build_default_reranker
from .retriever import HybridRetriever, Hit
from .reranker import RerankedHit

log = structlog.get_logger(__name__)


def _interleave_hits_by_article(hits: list[Hit]) -> list[Hit]:
    """Mélange les extraits : un chunk par article en tour à tour, ordre d’abord
    d’apparition dans `hits` (souvent score RRF décroissant). Aide le reranker
    à voir plusieurs fiches, pas 10 chunks d’une seule (ex. seulement « sauce »)."""
    if not hits or len(hits) < 2:
        return hits

    by_art: dict[int, list[Hit]] = defaultdict(list)
    for h in hits:
        by_art[h.article_external_id].append(h)
    order: list[int] = []
    seen: set[int] = set()
    for h in hits:
        a = h.article_external_id
        if a not in seen:
            seen.add(a)
            order.append(a)
    buckets = [by_art[aid] for aid in order]
    out: list[Hit] = []
    i = 0
    while any(i < len(b) for b in buckets):
        for b in buckets:
            if i < len(b):
                out.append(b[i])
        i += 1
    return out


_OBJECTION_USER = re.compile(
    r"(?i)(n'y a|n'y en a|pas de|pas d'|ne contient|dedans|ingrédient|\?\?)",
)
_KNOWN_ING_RE = re.compile(
    r"(?i)\b(concombre|tomate|tomates|yaourt|poulet|citron|menthe|persil|aubergine|courgette|halloumi|boulgour|concombres)\b",
)


def _merge_objection_boost(
    user_query: str,
    conversation_history: str | None,
    focus: SessionFocus | None,
) -> SessionFocus | None:
    """Renfort déterministe si l'LLM session a raté l'objection (recherche trop large)."""
    if not (conversation_history or "").strip():
        return focus
    uq = (user_query or "").strip()
    if not _OBJECTION_USER.search(uq):
        return focus
    hist = conversation_history or ""
    m = _KNOWN_ING_RE.search(hist) or _KNOWN_ING_RE.search(uq)
    extra = (m.group(0).lower() if m else "").strip()
    lowh = hist.lower()
    if "fattouche" in lowh or "fattouch" in lowh:
        extra = f"{extra} fattouche salade".strip()
    if "taboul" in lowh:
        extra = f"{extra} taboulé".strip()
    if not extra:
        return focus
    if focus is None:
        return SessionFocus(
            search_boost_phrase=extra,
            thread_summary=(
                "[Objection] Fil ingrédient repris depuis l'historique ; "
                "prioriser une fiche où l'ingrédient apparaît au texte."
            ),
            user_wants_different_article=True,
            suggest_broaden_corpus_search=False,
        )
    boost = f"{(focus.search_boost_phrase or '').strip()} {extra}".strip()
    return focus.model_copy(
        update={
            "search_boost_phrase": boost,
            "suggest_broaden_corpus_search": False,
            "user_wants_different_article": True,
            "thread_summary": (
                (focus.thread_summary or "").strip()
                + " [Objection] Maintenir l'ingrédient du fil ; prioriser une fiche où il apparaît au texte."
            )[:400],
        }
    )


def _merge_follow_up_boost(
    user_query: str,
    focus: SessionFocus | None,
) -> SessionFocus | None:
    """Renfort si l'utilisateur demande explicitement une autre recette."""
    if not wants_another_recipe(user_query):
        return focus
    if focus is None:
        return SessionFocus(
            user_wants_different_article=True,
            suggest_broaden_corpus_search=False,
            thread_summary="[Relance] L'utilisateur veut une autre recette sur le même fil.",
        )
    return focus.model_copy(
        update={
            "user_wants_different_article": True,
            "suggest_broaden_corpus_search": False,
        }
    )


def _build_no_alternate_ingredient_answer(slugs: list[str]) -> GroundedAnswer:
    ing = ingredient_display_name(slugs[0])
    return GroundedAnswer(
        answer_sentences=[
            GroundedSentence(
                text=(
                    f"Je n'ai pas d'autre recette à base de {ing} "
                    f"à vous proposer pour l'instant."
                ),
                source_chunk_ids=[],
            )
        ],
        recipe_card=None,
        recipe_card_secondary=None,
        chef_card=None,
        follow_up=(
            "Souhaitez-vous un autre ingrédient ou le nom d'un plat libanais précis ?"
        ),
        confidence=0.12,
    )


async def _probe_corpus_searchable(
    retriever: HybridRetriever,
    session: AsyncSession,
) -> bool:
    """True si au moins un chunk est retrievable sans filtre structuré."""
    try:
        probe = await retriever.search(
            session,
            "recette libanaise",
            chef_slugs=[],
            ingredient_slugs=[],
            category_slugs=[],
            keyword_slugs=[],
            final_limit=1,
        )
        return bool(probe)
    except Exception:  # noqa: BLE001
        return False


def _dedupe_merge_hits(primary: list[Hit], secondary: list[Hit], *, cap: int = 100) -> list[Hit]:
    """Préserve l’ordre (fusionner sans doublon de chunk) pour alimenter le reranker."""
    seen: set[int] = {h.chunk_id for h in primary}
    out = list(primary)
    for h in secondary:
        if h.chunk_id in seen:
            continue
        seen.add(h.chunk_id)
        out.append(h)
        if len(out) >= cap:
            break
    return out


def _article_rerank_documents(
    reranked: list[RerankedHit],
    *,
    max_articles: int,
    max_chunks_per_article: int,
) -> list[Hit]:
    """Construit des documents article-level pour une 2e passe reranker."""
    if not reranked:
        return []
    by_article: dict[int, list[RerankedHit]] = defaultdict(list)
    article_order: list[int] = []
    for r in reranked:
        aid = int(r.hit.article_external_id)
        if aid not in by_article:
            article_order.append(aid)
        by_article[aid].append(r)
    docs: list[Hit] = []
    for aid in article_order[: max(1, max_articles)]:
        items = sorted(by_article[aid], key=lambda x: x.rerank_score, reverse=True)[
            : max(1, max_chunks_per_article)
        ]
        top = items[0].hit
        text_parts: list[str] = []
        for rr in items:
            sk = (rr.hit.section_kind or "").strip()
            chunk = (rr.hit.chunk_text or "").strip()
            if not chunk:
                continue
            text_parts.append(f"[{sk}] {chunk}" if sk else chunk)
        doc_text = "\n".join(text_parts)[:6000]
        docs.append(
            Hit(
                chunk_id=top.chunk_id,
                article_id=top.article_id,
                article_external_id=top.article_external_id,
                article_title=top.article_title,
                article_url=top.article_url,
                cover_image_url=top.cover_image_url,
                section_kind="article_doc",
                chunk_text=doc_text,
                score_lex=top.score_lex,
                score_vec=top.score_vec,
                score_rrf=top.score_rrf,
                metadata=top.metadata or {},
            )
        )
    return docs


def _apply_article_rerank(
    reranked: list[RerankedHit],
    article_ranked: list[RerankedHit],
    *,
    keep_top_articles: int,
) -> list[RerankedHit]:
    if not reranked or not article_ranked:
        return reranked
    article_order = [int(r.hit.article_external_id) for r in article_ranked]
    article_pos = {aid: i for i, aid in enumerate(article_order)}
    keep = set(article_order[: max(1, keep_top_articles)])
    filtered = [r for r in reranked if int(r.hit.article_external_id) in keep]
    if not filtered:
        filtered = reranked
    return sorted(
        filtered,
        key=lambda r: (
            article_pos.get(int(r.hit.article_external_id), 10**9),
            -float(r.rerank_score),
        ),
    )


def _norm_text(s: str) -> str:
    t = unicodedata.normalize("NFKD", s or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^\w\s]+", " ", t.lower())
    return " ".join(t.split())


def _is_canonical_olj_url(url: str | None) -> bool:
    u = (url or "").lower()
    return "lorientlejour.com/cuisine-liban-a-table/" in u


def _apply_source_priority(
    reranked: list[RerankedHit],
    user_query: str,
) -> list[RerankedHit]:
    """Priorise OLJ indexé; non-canonique seulement sur demande exacte.

    Règle produit demandée :
    - priorité au corpus OLJ indexé (base principale),
    - une fiche non canonique ne remonte que si l’utilisateur la demande
      de façon quasi exacte (titre/fiche précis).
    """
    if len(reranked) < 2:
        return reranked
    qn = _norm_text(user_query or "")
    qn = re.sub(r"\b(recette|fiche|de|du|des|la|le|les)\b", " ", qn)
    qn = " ".join(qn.split())

    def _is_exact_title_match(title: str) -> bool:
        tn = _norm_text(title or "")
        if not qn or len(qn) < 6 or not tn:
            return False
        return qn in tn or tn in qn

    def _bucket(r: RerankedHit) -> tuple[int, float]:
        canonical = _is_canonical_olj_url(r.hit.article_url)
        exact_noncanonical = (not canonical) and _is_exact_title_match(
            r.hit.article_title or ""
        )
        if canonical:
            return (0, -float(r.rerank_score))
        if exact_noncanonical:
            return (1, -float(r.rerank_score))
        return (2, -float(r.rerank_score))

    return sorted(reranked, key=_bucket)


_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    # Variantes orthographiques courantes (FR/EN/translittération).
    ("houmous", "hoummous", "hummus", "hommos", "hommous"),
    ("moghrabieh", "moghrabie", "moghrabié", "moghrabiyé"),
)

# Alias arabes / levantins pour chaque ingrédient connu.
# Utilisés pour enrichir la requête de recherche quand un slug ingrédient est actif,
# car beaucoup d'articles OLJ mélangent le français et l'arabe translittéré dans leurs chunks.
_INGREDIENT_ARABIC_ALIASES: dict[str, tuple[str, ...]] = {
    "concombre": ("khyar", "khiar", "fattouche", "fattoush", "fattouch", "fattush"),
    "tomate": ("banadoura", "bandoura", "tomates fraîches"),
    "aubergine": ("batinjane", "batinghane", "moutabbal", "mouttabbal"),
    "courgette": ("kousa", "kusa", "mehche kousa"),
    "poulet": ("djej", "dajaj", "djeij"),
    "pois-chiche": ("hommos", "houmous", "hummus", "hoummous"),
    "citron": ("laymoun", "hamad", "jus de citron"),
    "persil": ("baqle", "baqlé"),
    "menthe": ("naana", "naané"),
}

# Regex pour détecter une rupture de fil ingrédient dans la question courante.
# Ex : "sans poivron", "pas forcément de/du X", "sans aucun X", "rien avec X"
_NEGATIVE_INGREDIENT_RE = re.compile(
    r"(?i)\b(?:sans|pas de|pas du|pas d'|pas forcément|sans forcément|sans aucun|"
    r"ne veux pas|je ne veux pas|éviter|autre chose|pas d'ingrédient)\b",
)

# Détecte les requêtes de type « humeur / ambiance » sans ingrédient précis.
# Ces requêtes doivent réinitialiser le contexte ingrédient issu de l'historique.
_MOOD_QUERY_RE = re.compile(
    r"(?i)\b(?:simple|simplement|facile|rapide|express|léger|légère|soir|ce soir|"
    r"pour ce midi|ce midi|vite fait|vite-fait|rapide à faire|facile à faire|"
    r"ce weekend|pour demain|plat du soir|repas du soir|dîner rapide|dîner simple)\b",
)

# En image Docker, seul v4/data est copié (/app/data). On cherche donc d'abord
# data/Data_base_2.json ; fallback racine du repo pour le dev local.
_BASE2_JSON_PATH = Path(__file__).resolve().parents[3] / "data" / "Data_base_2.json"
if not _BASE2_JSON_PATH.is_file():
    _BASE2_JSON_PATH = Path(__file__).resolve().parents[3].parent / "Data_base_2.json"
_BASE2_RECIPE_ALIASES: tuple[tuple[str, ...], ...] = (
    ("houmous", "hoummous", "hummus", "hommos", "hommous"),
    ("moghrabieh", "moghrabié", "moghrabie", "moghrabiyé"),
)
_BASE2_TOKEN_STOPWORDS = {
    "recette",
    "recettes",
    "de",
    "du",
    "des",
    "la",
    "le",
    "les",
    "d",
    "l",
    "a",
    "au",
    "aux",
    "base",
    "faire",
    "comment",
    "pour",
}
_BASE2_RECIPES_CACHE: list[dict[str, Any]] | None = None


def _contains_term(text: str, term: str) -> bool:
    return bool(re.search(rf"(?i)\b{re.escape(term)}\b", text))


# ── Canonicalisation des translittérations de plats (donnée auditable) ──────
# Source : data/aliases_dishes.json. On REMPLACE la graphie utilisateur par la
# forme réellement indexée (méthode 'canonicalize_replace'), car l'empirie montre
# qu'ajouter laisse le mauvais token tirer ailleurs (ex. 'tabbouleh taboulé'
# échoue, mais 'taboulé' seul rang 1). Voir docs/alias-validation-report.md.
_DISH_ALIASES_PATH = Path(__file__).resolve().parents[3] / "data" / "aliases_dishes.json"
_INGREDIENT_ALIASES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "aliases_ingredients.json"
)


def _build_canon_rules() -> list[tuple[re.Pattern[str], str]]:
    """Règles variante→forme indexée, depuis les datasets plats + ingrédients."""
    pairs: list[tuple[str, str]] = []
    # Plats : cible = canonical_indexed.
    try:
        dishes = json.loads(_DISH_ALIASES_PATH.read_text(encoding="utf-8"))
        for dish in dishes.get("dishes", []):
            canon = (dish.get("canonical_indexed") or "").strip()
            if not canon:
                continue
            for variant in dish.get("user_variants", []):
                v = (variant or "").strip()
                if v and v.lower() != canon.lower():
                    pairs.append((v, canon))
    except Exception as exc:  # noqa: BLE001
        log.warning("rag.pipeline.dish_aliases_load_failed", error=str(exc))
    # Ingrédients : cible = inject[0] (forme présente dans les chunks). On saute
    # « pois-chiche » dont les graphies (hommos/houmous) sont gérées comme PLAT.
    try:
        ings = json.loads(_INGREDIENT_ALIASES_PATH.read_text(encoding="utf-8"))
        for ing in ings.get("ingredients", []):
            if ing.get("canonical") == "pois-chiche":
                continue
            inject = ing.get("inject") or []
            target = (inject[0] if inject else (ing.get("canonical") or "")).strip()
            if not target:
                continue
            for variant in ing.get("variants", []):
                v = (variant or "").strip()
                if v and v.lower() != target.lower():
                    pairs.append((v, target))
    except Exception as exc:  # noqa: BLE001
        log.warning("rag.pipeline.ingredient_aliases_load_failed", error=str(exc))
    # Plus longues d'abord (multi-mots avant mono-mot) pour éviter les chevauchements.
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    rules: list[tuple[re.Pattern[str], str]] = []
    for v, canon in pairs:
        vn = re.escape(v.replace("’", "'"))
        pat = re.compile(rf"(?i)(?<![\wàâäéèêëïîôùûç']){vn}(?![\wàâäéèêëïîôùûç'])")
        rules.append((pat, canon))
    return rules


_DISH_CANON_RULES = _build_canon_rules()


def _canonicalize_dish_aliases(q: str) -> str:
    """Remplace les graphies translittérées de plats par la forme indexée."""
    if not q:
        return q
    s = q.replace("’", "'")
    for pat, canon in _DISH_CANON_RULES:
        s = pat.sub(canon, s)
    return s


def _norm_match(s: str) -> str:
    # Normalise les apostrophes (courbe ’ vs droite ') PUIS les supprime, pour que
    # « mana'ich » (alias) matche « mana’ichs » (titre) — sinon faux négatif qui
    # déclenche à tort le fallback (bug manouche -> recette générée).
    t = unicodedata.normalize("NFKD", (s or "").lower()).replace("’", "'")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t.replace("'", "")


def _load_dish_entries() -> list[tuple[tuple[str, ...], str, str | None]]:
    """Par plat : (termes normalisés >=4 car. pour la détection, forme canonique
    indexée, pin_query optionnel). pin_query = requête distinctive et stable qui
    pointe DÉTERMINISTE l'article du plat (plat+chef), pour les translittérations
    où le retrieval vectoriel est instable."""
    out: list[tuple[tuple[str, ...], str, str | None]] = []
    try:
        data = json.loads(_DISH_ALIASES_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return out
    for dish in data.get("dishes", []):
        canon = (dish.get("canonical_indexed") or "").strip()
        pin = (dish.get("pin_query") or "").strip() or None
        terms: set[str] = set()
        if canon:
            terms.add(canon)
        for key in ("indexed_forms", "user_variants"):
            for v in dish.get(key, []) or []:
                if isinstance(v, str):
                    terms.add(v)
        norm = {_norm_match(t) for t in terms if len(_norm_match(t)) >= 4}
        if norm and canon:
            out.append((tuple(sorted(norm, key=len, reverse=True)), canon, pin))
    return out


_DISH_ENTRIES = _load_dish_entries()


def _requested_dish_terms(user_query: str) -> list[str]:
    """Termes des plats explicitement nommés dans la requête (vide si aucun)."""
    qn = _norm_match(user_query)
    found: list[str] = []
    for terms, _canon, _pin in _DISH_ENTRIES:
        if any(t in qn for t in terms):
            found.extend(terms)
    return found


def _primary_dish_canonical(user_query: str) -> str | None:
    """Forme canonique indexée du premier plat connu nommé dans la requête."""
    qn = _norm_match(user_query)
    for terms, canon, _pin in _DISH_ENTRIES:
        if any(t in qn for t in terms):
            return canon
    return None


def _primary_dish_pin(user_query: str) -> str | None:
    """Requête-pin déterministe du premier plat épinglé nommé (sinon None)."""
    qn = _norm_match(user_query)
    for terms, _canon, pin in _DISH_ENTRIES:
        if pin and any(t in qn for t in terms):
            return pin
    return None


def _card_title_matches_requested_dish(
    card_title: str, requested_terms: list[str]
) -> bool:
    """True si aucun plat précis n'est demandé, ou si le titre concerne ce plat."""
    if not requested_terms:
        return True
    t = _norm_match(card_title)
    return any(term in t for term in requested_terms)


def _drop_speculative_ingredients(slugs: list[str], user_query: str) -> list[str]:
    """Requête nommant un plat connu → le PLAT est la cible : on retire TOUS les
    filtres d'ingrédients. Le LLM extrait souvent des composants (zaatar, fromage),
    voire le nom du plat lui-même ('manaiichs'), comme 'ingrédient' ; en filtre dur
    cela exclut à tort l'article du plat (abstention). Sinon, liste inchangée."""
    if not slugs or not _requested_dish_terms(user_query):
        return slugs
    return []


def _ensure_recipe_card(
    answer: GroundedAnswer,
    reranked: list[RerankedHit],
    user_query: str,
    min_score: float,
) -> GroundedAnswer:
    """Si un article-recette pertinent a été trouvé mais que la génération n'a pas
    émis de carte (elle a seulement décrit le plat), synthétiser une carte minimale
    depuis l'article (titre + chef ; le lien est rendu par l'UI). Respecte les
    abstentions (confiance < 0.5) et la faible pertinence (score < min_score)."""
    if answer.recipe_card is not None or not reranked:
        return answer
    requested = _requested_dish_terms(user_query)
    top = None
    if requested:
        # Plat explicitement nommé : 1er article qui correspond au plat (même si
        # la confiance est moyenne — ce n'est pas une abstention).
        top = next(
            (r for r in reranked
             if _card_title_matches_requested_dish(r.hit.article_title, requested)),
            None,
        )
    elif (answer.confidence or 0.0) >= 0.5 and reranked[0].rerank_score >= min_score:
        # Requête non-plat : carter le top si pas une abstention et pertinence ok.
        top = reranked[0]
    if top is None:
        return answer
    card = RecipeCard(
        title=top.hit.article_title,
        chef=_chef_name_from_metadata(top.hit.metadata),
        source_chunk_ids=[top.hit.chunk_id],
        ingredients=[],
        steps=[],
    )
    return answer.model_copy(update={"recipe_card": card})


def _expand_query_with_aliases(q: str) -> str:
    """Canonicalise les translittérations de plats puis ajoute les alias ortho."""
    base = _canonicalize_dish_aliases((q or "").strip())
    if not base:
        return base
    extras: list[str] = []
    for group in _ALIAS_GROUPS:
        if not any(_contains_term(base, t) for t in group):
            continue
        for t in group:
            if not _contains_term(base, t):
                extras.append(t)
    if not extras:
        return base
    return f"{base} {' '.join(extras)}".strip()


def _base2_normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    t = re.sub(r"[^\w\s]+", " ", t)
    return " ".join(t.split())


def _base2_canonicalize_aliases(text: str) -> str:
    out = _base2_normalize_text(text)
    for group in _BASE2_RECIPE_ALIASES:
        canonical = _base2_normalize_text(group[0])
        for alias in group:
            alias_norm = _base2_normalize_text(alias)
            if alias_norm == canonical:
                continue
            out = re.sub(rf"\b{re.escape(alias_norm)}\b", canonical, out)
    return " ".join(out.split())


def _extract_explicit_recipe_name(user_query: str) -> str | None:
    raw = (user_query or "").strip()
    if not raw:
        return None
    lowered = _base2_canonicalize_aliases(raw)
    candidate = ""
    patterns = (
        r"\b(?:recette|plat)\s+(?:de|du|des|d|l)?\s*(.+)$",
        r"^comment\s+(?:faire|preparer)\s+(?:de|du|des|d|l)?\s*(.+)$",
        r"^ta\s+recette\s+(?:de|du|des|d|l)?\s*(.+)$",
        r"^la\s+recette\s+(?:de|du|des|d|l)?\s*(.+)$",
    )
    for pat in patterns:
        m = re.search(pat, lowered)
        if m:
            candidate = m.group(1)
            break
    if not candidate:
        return None
    tokens = [
        tok
        for tok in candidate.split()
        if tok not in _BASE2_TOKEN_STOPWORDS and len(tok) > 1
    ]
    if not tokens:
        return None
    return " ".join(tokens).strip()


def _load_base2_recipes() -> list[dict[str, Any]]:
    global _BASE2_RECIPES_CACHE
    if _BASE2_RECIPES_CACHE is not None:
        return _BASE2_RECIPES_CACHE
    if not _BASE2_JSON_PATH.is_file():
        _BASE2_RECIPES_CACHE = []
        return _BASE2_RECIPES_CACHE
    try:
        data = json.loads(_BASE2_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("rag.pipeline.base2_load_failed", error=str(exc))
        _BASE2_RECIPES_CACHE = []
        return _BASE2_RECIPES_CACHE
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict):
        for category, items in data.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("nom") or "").strip()
                if not name:
                    continue
                rows.append(
                    {
                        "name": name,
                        "name_norm": _base2_canonicalize_aliases(name),
                        "category": str(category or "").strip(),
                        "serves": item.get("nombre_de_personnes"),
                        "prep": str(item.get("temps_preparation") or "").strip(),
                        "cook": str(item.get("temps_cuisson") or "").strip(),
                        "difficulty": str(item.get("difficulte") or "").strip(),
                        "ingredients": item.get("ingredients") or [],
                        "steps": item.get("etapes") or [],
                    }
                )
    _BASE2_RECIPES_CACHE = rows
    return rows


# Tokens génériques de nom de recette base2 : présents dans plusieurs plats, ils
# ne doivent pas suffire à matcher (sinon "salade" matcherait toutes les salades).
_BASE2_GENERIC_NAME_TOKENS = {
    "salade", "soupe", "plat", "entree", "dessert", "caviar", "puree", "feuilles",
    "vigne", "tranches", "huile", "olive", "froides", "froide", "chaud", "chaude",
    "maison", "libanais", "libanaise", "aubergine", "aubergines", "pois", "chiche",
    "chiches", "tomate", "tomates", "roquette", "lentilles", "coriandre",
}


def _match_base2_recipe(user_query: str) -> dict[str, Any] | None:
    """Meilleure recette base2 par recouvrement de tokens DISTINCTIFS (et non plus
    égalité exacte, qui ratait « fattouche » vs « Salade fattouche »). Robuste aux
    graphies (canonicalisation) et aux noms nus comme aux tournures « recette de X »."""
    recipes = _load_base2_recipes()
    if not recipes:
        return None
    qn = _base2_canonicalize_aliases(user_query)
    qtokens = {
        t for t in qn.split()
        if t not in _BASE2_TOKEN_STOPWORDS and len(t) > 2
    }
    if not qtokens:
        return None
    best: dict[str, Any] | None = None
    best_score = 0
    for recipe in recipes:
        ntokens = {t for t in recipe.get("name_norm", "").split() if len(t) > 2}
        distinctive = {
            t for t in (qtokens & ntokens)
            if t not in _BASE2_GENERIC_NAME_TOKENS and len(t) >= 4
        }
        if not distinctive:
            continue
        score = sum(len(t) for t in distinctive)
        if score > best_score:
            best_score = score
            best = recipe
    return best


def _olj_has_named_dish(user_query: str, reranked: list[RerankedHit]) -> bool:
    """True si un plat connu (aliases) nommé dans la requête a un article correspondant."""
    req = _requested_dish_terms(user_query)
    if not req:
        return False
    return any(
        _card_title_matches_requested_dish(h.hit.article_title, req) for h in reranked
    )


def _olj_has_dish_text(name: str, reranked: list[RerankedHit]) -> bool:
    """True si un titre d'article reranké contient le nom de plat (texte libre)."""
    n = _norm_match(name)
    if len(n) < 4:
        return False
    return any(n in _norm_match(h.hit.article_title) for h in reranked)


# Marqueurs d'une requête PAR INGRÉDIENT (≠ plat nommé) : on ne génère pas.
# « au/aux/à la/à base de/avec/sans X » => X est un ingrédient cuisiné, pas le plat
# (« recette au citron » ≠ un plat « citron »). « recette DE X » reste un plat.
_INGREDIENT_QUERY_MARKERS = ("avec", "sans", "au", "aux")


def _fallback_dish_name(user_query: str, plan: QueryPlan) -> str | None:
    """Nom de plat candidat à la génération de dernier recours, sinon None.

    Distinguer un PLAT nommé (« katayef », « recette de baklava ») d'une requête
    PAR INGRÉDIENT (« recette avec du concombre ») : le LLM met souvent les
    composants — voire le nom du plat lui-même (katayef → ingredient 'katayef') —
    dans `ingredient_slugs`, donc on NE peut PAS s'y fier. On se base sur la
    FORMULATION : marqueurs « avec / sans / à base de » ⇒ requête ingrédient (pas
    de génération) ; sinon un nom de plat explicite (« recette de X ») ou nu
    (1-3 mots) est un plat → on génère."""
    if plan.intent not in ("recipe", "mixed"):
        return None
    qn = _base2_normalize_text(user_query)
    toks_all = qn.split()
    if (
        any(m in toks_all for m in _INGREDIENT_QUERY_MARKERS)
        or "a base" in qn
        or "a la " in qn
        or "a l " in qn
        or "qui contient" in qn
        or "a partir" in qn
    ):
        return None
    name = _extract_explicit_recipe_name(user_query)
    if name:
        return name
    toks = [t for t in toks_all if t not in _BASE2_TOKEN_STOPWORDS and len(t) > 2]
    if 1 <= len(toks) <= 3:
        return " ".join(toks)
    return None


GENERATED_INTRO = (
    "Ce plat n'est pas (encore) dans les carnets de L'Orient-Le Jour ; "
    "voici une version générée à titre indicatif (hors carnets OLJ)."
)


def _build_generated_recipe_answer(
    *,
    user_query: str,
    generated: dict[str, Any],
    reranked: list[RerankedHit],
) -> GroundedAnswer:
    """Réponse pour un plat absent partout : recette générée (clairement étiquetée
    hors-OLJ) dans le texte + une suggestion OLJ proche en carte (si disponible)."""
    top_canonical = next(
        (r for r in reranked if _is_canonical_olj_url(r.hit.article_url)),
        reranked[0] if reranked else None,
    )
    recipe_card: RecipeCard | None = None
    olj_title = ""
    olj_chunk_id: int | None = None
    if top_canonical is not None:
        aid = int(top_canonical.hit.article_external_id)
        picked = _best_chunk_for_article(reranked, aid) or top_canonical
        olj_chunk_id = int(picked.hit.chunk_id)
        olj_title = picked.hit.article_title or top_canonical.hit.article_title or ""
        recipe_card = RecipeCard(
            title=olj_title or "Recette OLJ",
            source_chunk_ids=[olj_chunk_id],
            ingredients=[],
            steps=[],
        )

    name = str(generated.get("name") or "ce plat").strip()
    details_bits: list[str] = []
    serves = str(generated.get("serves") or "").strip()
    prep = str(generated.get("prep") or "").strip()
    cook = str(generated.get("cook") or "").strip()
    difficulty = str(generated.get("difficulty") or "").strip()
    if serves:
        details_bits.append(f"pour {serves}")
    if prep:
        details_bits.append(f"préparation {prep}")
    if cook:
        details_bits.append(f"cuisson {cook}")
    if difficulty:
        details_bits.append(f"difficulté {difficulty}")
    details = ", ".join(details_bits)
    ingredients = [str(x).strip() for x in (generated.get("ingredients") or []) if str(x).strip()]
    steps = [str(x).strip() for x in (generated.get("steps") or []) if str(x).strip()]

    sentences = [GroundedSentence(text=GENERATED_INTRO, source_chunk_ids=[])]
    head = f"Recette de {name}" + (f" ({details})" if details else "") + " :"
    sentences.append(GroundedSentence(text=head, source_chunk_ids=[]))
    if ingredients:
        sentences.append(
            GroundedSentence(
                text="Ingrédients : " + " ; ".join(ingredients[:14]) + ".",
                source_chunk_ids=[],
            )
        )
    if steps:
        numbered = " ".join(f"{i}. {s}" for i, s in enumerate(steps[:8], 1))
        sentences.append(
            GroundedSentence(text="Préparation : " + numbered, source_chunk_ids=[])
        )
    note = str(generated.get("note") or "").strip()
    if note:
        sentences.append(GroundedSentence(text="Astuce : " + note, source_chunk_ids=[]))
    if olj_title and olj_chunk_id is not None:
        sentences.append(
            GroundedSentence(
                text=(
                    f"Sur L'Orient-Le Jour, dans le même esprit, vous pourriez aussi "
                    f"aimer {olj_title}."
                ),
                source_chunk_ids=[olj_chunk_id],
            )
        )
    follow_up = (
        f"Souhaitez-vous que je vous en dise plus sur {olj_title} ?"
        if olj_title
        else "Souhaitez-vous une autre recette ?"
    )
    answer = GroundedAnswer(
        answer_sentences=sentences,
        recipe_card=recipe_card,
        recipe_card_secondary=None,
        chef_card=None,
        follow_up=follow_up,
        confidence=0.45,
    )
    return validate_grounding(answer, reranked, user_query=user_query, preserve_carnets=True)


def _best_chunk_for_article(
    reranked: list[RerankedHit], article_external_id: int
) -> RerankedHit | None:
    best: RerankedHit | None = None
    for h in reranked:
        if int(h.hit.article_external_id) != int(article_external_id):
            continue
        sk = (h.hit.section_kind or "").lower()
        if sk == "recipe_meta":
            return h
        if best is None:
            best = h
            continue
        if sk == "recipe_summary" and (best.hit.section_kind or "").lower() != "recipe_summary":
            best = h
    return best


def _format_base2_ingredients_short(raw_ingredients: Any) -> str:
    if not isinstance(raw_ingredients, list) or not raw_ingredients:
        return ""
    rendered: list[str] = []
    for row in raw_ingredients[:6]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("nom") or "").strip()
        qty = row.get("quantite")
        unit = str(row.get("unite") or "").strip()
        if not name:
            continue
        if qty is None or qty == "":
            rendered.append(name)
            continue
        rendered.append(f"{qty} {unit} {name}".strip())
    return ", ".join(rendered)


def _format_base2_steps_short(raw_steps: Any) -> str:
    if not isinstance(raw_steps, list) or not raw_steps:
        return ""
    steps = [str(s).strip() for s in raw_steps if str(s).strip()]
    if not steps:
        return ""
    return " | ".join(steps[:3])


def _build_base2_last_resort_answer(
    *,
    user_query: str,
    base2_recipe: dict[str, Any],
    reranked: list[RerankedHit],
) -> GroundedAnswer:
    top_canonical = next(
        (r for r in reranked if _is_canonical_olj_url(r.hit.article_url)),
        reranked[0] if reranked else None,
    )
    recipe_card: RecipeCard | None = None
    olj_title = ""
    olj_chunk_id: int | None = None
    if top_canonical is not None:
        aid = int(top_canonical.hit.article_external_id)
        picked = _best_chunk_for_article(reranked, aid) or top_canonical
        olj_chunk_id = int(picked.hit.chunk_id)
        olj_title = picked.hit.article_title or top_canonical.hit.article_title or ""
        recipe_card = RecipeCard(
            title=olj_title or "Recette OLJ",
            chef=None,
            duration_min=None,
            serves=None,
            ingredients=[],
            steps=[],
            source_chunk_ids=[olj_chunk_id],
        )

    serves = base2_recipe.get("serves")
    prep = str(base2_recipe.get("prep") or "").strip()
    cook = str(base2_recipe.get("cook") or "").strip()
    difficulty = str(base2_recipe.get("difficulty") or "").strip()
    details_bits: list[str] = []
    if serves:
        details_bits.append(f"pour {serves} personnes")
    if prep:
        details_bits.append(f"préparation {prep}")
    if cook:
        details_bits.append(f"cuisson {cook}")
    if difficulty:
        details_bits.append(f"difficulté {difficulty}")
    details = ", ".join(details_bits)
    ingredients_short = _format_base2_ingredients_short(base2_recipe.get("ingredients"))
    steps_short = _format_base2_steps_short(base2_recipe.get("steps"))
    base2_name = str(base2_recipe.get("name") or "cette recette").strip()

    sentences = [
        GroundedSentence(text=CARNETS_PHRASE, source_chunk_ids=[]),
    ]
    if details:
        sentences.append(
            GroundedSentence(
                text=(
                    f"Voici tout de même une version courte de {base2_name} "
                    f"({details})."
                ),
                source_chunk_ids=[],
            )
        )
    if ingredients_short:
        sentences.append(
            GroundedSentence(
                text=f"Ingrédients (résumé) : {ingredients_short}.",
                source_chunk_ids=[],
            )
        )
    if steps_short:
        sentences.append(
            GroundedSentence(
                text=f"Étapes (courtes) : {steps_short}.",
                source_chunk_ids=[],
            )
        )
    if olj_title and olj_chunk_id is not None:
        sentences.append(
            GroundedSentence(
                text=(
                    f"Sur L'Orient-Le Jour, je vous recommande aussi {olj_title}, "
                    f"une recette proche publiée dans nos pages « À table »."
                ),
                source_chunk_ids=[olj_chunk_id],
            )
        )
    follow_up = (
        f"Souhaitez-vous que je vous en dise plus sur {olj_title} ?"
        if olj_title
        else "Souhaitez-vous que je cherche une autre variante proche ?"
    )
    answer = GroundedAnswer(
        answer_sentences=sentences,
        recipe_card=recipe_card,
        recipe_card_secondary=None,
        chef_card=None,
        follow_up=follow_up,
        confidence=0.42,
    )
    return validate_grounding(answer, reranked, user_query=user_query, preserve_carnets=True)


def _expand_search_q_with_ingredients(base_q: str, plan: QueryPlan) -> str:
    """Aligne requête BM25/embedding sur les mots d’ingrédient et leurs alias arabes."""
    qn = (base_q or "").strip()
    qlow = qn.lower()
    extra: list[str] = []
    extra_set: set[str] = set()

    def _add(w: str) -> None:
        wl = w.lower()
        if wl and wl not in qlow and wl not in extra_set:
            extra_set.add(wl)
            extra.append(w)

    for s in plan.ingredient_slugs or []:
        w = s.replace("-", " ").strip()
        if not w:
            continue
        _add(w)
        for alias in _INGREDIENT_ARABIC_ALIASES.get(s, ()):
            _add(alias)
    if not extra:
        return _expand_query_with_aliases(qn)
    return _expand_query_with_aliases(f"{qn} {' '.join(extra)}".strip())


_STOPWORDS_FR = {
    "avec",
    "pour",
    "dans",
    "une",
    "des",
    "les",
    "du",
    "de",
    "la",
    "le",
    "un",
    "est",
    "sont",
    "que",
    "qui",
    "recette",
    "recettes",
    "comment",
    "fait",
    "faire",
    "donne",
    "donne-moi",
    "moi",
    "sur",
    "plus",
}


def _retrieval_fallback_queries(user_query: str, plan: QueryPlan) -> list[str]:
    """Requêtes plus courtes / centrées ingrédient si la recherche hybride ne renvoie rien."""
    seen: set[str] = set()
    out: list[str] = []
    uq = (user_query or "").strip()

    def add(s: str) -> None:
        s = " ".join(s.split())
        if len(s) < 2 or s in seen:
            return
        seen.add(s)
        out.append(s)

    for slug in plan.ingredient_slugs:
        if not slug:
            continue
        w = slug.replace("-", " ").strip()
        add(w)
        if w:
            add(f"{w} recette")
            expanded = _expand_query_with_aliases(w)
            if expanded != w:
                add(expanded)
                add(f"{expanded} recette")
    for w in re.findall(r"\b[\wàâäéèêëïîôùûç]+\b", uq.lower()):
        if w not in _STOPWORDS_FR and len(w) > 2:
            add(w)
            if len(w) > 4:
                add(f"{w} recette")
    rw = (plan.rewritten_query or "").strip()
    if rw and rw.casefold() != uq.casefold():
        add(rw)
    add(_expand_query_with_aliases(uq))
    if rw:
        add(_expand_query_with_aliases(rw))
    return out[:10]


@dataclass
class PipelineResult:
    plan: QueryPlan
    hits: list[Hit]
    reranked: list[RerankedHit]
    answer: GroundedAnswer
    timings_ms: dict[str, int]
    is_base2_fallback: bool = False
    cost_breakdown: dict[str, Any] | None = None
    answer_strategy: str = "normal_generate"


# Garde d'entrée (sécurité) : injection de prompt / extraction des instructions /
# jailbreak. Traité EN CODE (pas via le prompt, qui est précisément la cible).
_PROMPT_ATTACK_RE = re.compile(
    r"(?i)(?:"
    r"ignore[a-z]*\s+(?:tou(?:te|s)?s?\s+)?(?:tes|les|ces|vos)\s+(?:instructions|consignes|r[èe]gles|directives)"
    r"|(?:ton|votre|le|the)\s+(?:prompt|system\s+prompt)"
    r"|prompt\s+syst[èe]me|system\s+prompt|your\s+(?:system\s+)?prompt|tes\s+instructions"
    r"|tes\s+consignes|tes\s+r[èe]gles|tes\s+directives|tes\s+r[èe]gles\s+absolues"
    r"|sans\s+(?:aucune\s+)?restrictions?|tu\s+es\s+maintenant|you\s+are\s+now"
    r"|jailbreak|mode\s+d[ée]veloppeur|developer\s+mode|copie[- ]le\s+mot\s+pour\s+mot"
    r"|r[ée]v[èe]le[a-z]*\s+(?:ton|tes|moi)|reveal\s+your"
    r")"
)


def _build_security_redirect_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer_sentences=[
            GroundedSentence(
                text=(
                    "Je suis Sahteïn, l'assistant culinaire de L'Orient-Le Jour. "
                    "Je ne partage pas mes instructions internes, mais je serai "
                    "ravi de vous aider à trouver une recette libanaise."
                ),
                source_chunk_ids=[],
            )
        ],
        recipe_card=None,
        recipe_card_secondary=None,
        chef_card=None,
        follow_up="Quel plat, ingrédient ou chef vous ferait envie ?",
        confidence=0.0,
    )


class RagPipeline:
    def __init__(
        self,
        analyzer: QueryAnalyzer | None = None,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        generator: ResponseGenerator | None = None,
    ) -> None:
        self.analyzer = analyzer or QueryAnalyzer()
        self._session_focus = SessionFocusAnalyzer()
        self.retriever = retriever or HybridRetriever(OpenAIEmbeddings())
        self.reranker = reranker or build_default_reranker()
        self.generator = generator or ResponseGenerator()
        self.settings = get_settings()

    async def _retrieve_hits(
        self,
        session: AsyncSession,
        user_query: str,
        plan: QueryPlan,
        q: str,
        rerank_top_n: int,
        exclude_article_external_ids: list[int],
        *,
        force_corpus_broaden: bool = False,
    ) -> list[Hit]:
        """Recherche hybride + retries (filtres, fallbacks, élargissement corpus)."""
        s = self.settings
        excl = exclude_article_external_ids
        ex_boost = min(80, len(excl) * s.rag_retrieval_extra_limit_per_excluded)
        final_limit = max(30, rerank_top_n * 4) + ex_boost

        has_sql_filters = bool(
            plan.chef_slugs
            or plan.ingredient_slugs
            or plan.category_slugs
            or plan.keyword_slugs
        )
        ingredient_only = bool(plan.ingredient_slugs) and not (
            plan.chef_slugs or plan.category_slugs or plan.keyword_slugs
        )

        hits = await self.retriever.search(
            session,
            q,
            chef_slugs=plan.chef_slugs,
            ingredient_slugs=plan.ingredient_slugs,
            category_slugs=plan.category_slugs,
            keyword_slugs=plan.keyword_slugs,
            final_limit=final_limit,
            exclude_article_external_ids=excl,
        )
        if not hits and has_sql_filters:
            if ingredient_only:
                log.info(
                    "rag.pipeline.ingredient_broad_retry",
                    ingredient_slugs=plan.ingredient_slugs,
                )
                broad = await self.retriever.search(
                    session,
                    q,
                    chef_slugs=[],
                    ingredient_slugs=[],
                    category_slugs=[],
                    keyword_slugs=[],
                    final_limit=final_limit + 24,
                    exclude_article_external_ids=excl,
                )
                if broad and plan.ingredient_slugs:
                    filtered = filter_hits_by_ingredient_slugs(
                        broad, plan.ingredient_slugs
                    )
                    if not filtered:
                        filtered = filter_hits_by_ingredient_slugs(
                            broad, plan.ingredient_slugs, strict=False
                        )
                    # Dernier recours : terme présent dans le texte (toute section),
                    # pour les ingrédients réels mal liés en base (ex. concombre).
                    if not filtered:
                        filtered = filter_hits_by_ingredient_text(
                            broad, plan.ingredient_slugs
                        )
                    hits = filtered
            elif not ingredient_only:
                log.warning(
                    "rag.pipeline.retrieval_empty_with_filters_retrying_broad",
                    ingredient_slugs=plan.ingredient_slugs,
                    chef_slugs=plan.chef_slugs,
                )
                hits = await self.retriever.search(
                    session,
                    q,
                    chef_slugs=[],
                    ingredient_slugs=[],
                    category_slugs=[],
                    keyword_slugs=[],
                    final_limit=final_limit,
                    exclude_article_external_ids=excl,
                )
        if (
            s.rag_retrieval_widen_enabled
            and has_sql_filters
            and hits
            and (
                force_corpus_broaden
                or len(hits) < s.rag_retrieval_widen_min_hits
            )
        ):
            try:
                broad = await self.retriever.search(
                    session,
                    q,
                    chef_slugs=[],
                    ingredient_slugs=[],
                    category_slugs=[],
                    keyword_slugs=[],
                    final_limit=final_limit + 24,
                    exclude_article_external_ids=excl,
                )
                n0 = len(hits)
                hits = _dedupe_merge_hits(hits, broad, cap=120)
                if len(hits) > n0:
                    log.info(
                        "rag.pipeline.retrieval_merged_corpus_broaden",
                        n_before=n0,
                        n_after=len(hits),
                        force=force_corpus_broaden,
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("rag.pipeline.retrieval_widen_merge_failed", error=str(exc))
        if not hits:
            base = q.strip()
            for alt in _retrieval_fallback_queries(user_query, plan):
                if alt.strip() == base:
                    continue
                try:
                    hits = await self.retriever.search(
                        session,
                        alt,
                        chef_slugs=plan.chef_slugs if ingredient_only else [],
                        ingredient_slugs=(
                            plan.ingredient_slugs if ingredient_only else []
                        ),
                        category_slugs=[],
                        keyword_slugs=[],
                        final_limit=final_limit,
                        exclude_article_external_ids=excl,
                    )
                    if not hits and ingredient_only and plan.ingredient_slugs:
                        broad = await self.retriever.search(
                            session,
                            alt,
                            chef_slugs=[],
                            ingredient_slugs=[],
                            category_slugs=[],
                            keyword_slugs=[],
                            final_limit=final_limit + 24,
                            exclude_article_external_ids=excl,
                        )
                        hits = (
                            filter_hits_by_ingredient_slugs(
                                broad, plan.ingredient_slugs
                            )
                            or filter_hits_by_ingredient_slugs(
                                broad, plan.ingredient_slugs, strict=False
                            )
                            or filter_hits_by_ingredient_text(
                                broad, plan.ingredient_slugs
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "rag.pipeline.retrieval_fallback_failed",
                        alt=alt[:80],
                        error=str(exc),
                    )
                    continue
                if hits:
                    log.info(
                        "rag.pipeline.retrieval_fallback_hit",
                        alt_preview=alt[:80],
                        n_hits=len(hits),
                    )
                    break
        if plan.ingredient_slugs:
            # 1) Filtre structuré / par section (précis).
            filtered = filter_hits_by_ingredient_slugs(hits, plan.ingredient_slugs)
            # 2) Sinon, garder les hits courants qui contiennent vraiment le terme.
            if not filtered:
                filtered = filter_hits_by_ingredient_text(hits, plan.ingredient_slugs)
            # 3) Sinon, recherche DÉDIÉE sur le terme ingrédient NU (non dilué par
            #    "recette libanaise…" qui tirait vers des résultats hors-sujet),
            #    puis on ne garde que les articles mentionnant l'ingrédient.
            if not filtered:
                try:
                    terms = " ".join(
                        slug_search_terms(s)[0] for s in plan.ingredient_slugs
                    )
                    broad_ing = await self.retriever.search(
                        session,
                        terms,
                        chef_slugs=[],
                        ingredient_slugs=[],
                        category_slugs=[],
                        keyword_slugs=[],
                        final_limit=final_limit + 24,
                        exclude_article_external_ids=excl,
                    )
                    filtered = filter_hits_by_ingredient_text(
                        broad_ing, plan.ingredient_slugs
                    )
                    if filtered:
                        log.info(
                            "rag.pipeline.ingredient_term_search_recovered",
                            ingredient_slugs=plan.ingredient_slugs,
                            terms=terms,
                            n_hits=len(filtered),
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "rag.pipeline.ingredient_term_search_failed", error=str(exc)
                    )
            if filtered:
                hits = filtered
            else:
                # Aucune recette ne met cet ingrédient en avant : abstention honnête
                # (ne pas renvoyer des cartes hors-sujet type cookies/blette).
                log.warning(
                    "rag.pipeline.ingredient_no_real_match",
                    ingredient_slugs=plan.ingredient_slugs,
                )
                hits = []
        return hits

    async def answer(
        self,
        session: AsyncSession,
        user_query: str,
        *,
        rerank_top_n: int | None = None,
        session_id: str | None = None,
        conversation_history: str | None = None,
        llm_model: str | None = None,
    ) -> PipelineResult:
        with cost_tracker_scope():
            return await self._answer_impl(
                session,
                user_query,
                rerank_top_n=rerank_top_n,
                session_id=session_id,
                conversation_history=conversation_history,
                llm_model=llm_model,
            )

    async def _answer_impl(
        self,
        session: AsyncSession,
        user_query: str,
        *,
        rerank_top_n: int | None = None,
        session_id: str | None = None,
        conversation_history: str | None = None,
        llm_model: str | None = None,
    ) -> PipelineResult:
        timings: dict[str, int] = {}
        rerank_top_n = rerank_top_n or self.settings.rag_rerank_top_k
        model = resolve_llm_model(llm_model)

        # Garde sécurité : injection / extraction du prompt / jailbreak -> refus
        # canné, sans appeler le LLM (qui pourrait paraphraser ses instructions).
        if _PROMPT_ATTACK_RE.search(user_query or ""):
            log.info("rag.pipeline.security_redirect", query=(user_query or "")[:80])
            return PipelineResult(
                plan=QueryPlan(rewritten_query=(user_query or "").strip(), intent="mixed"),
                hits=[],
                reranked=[],
                answer=_build_security_redirect_answer(),
                timings_ms={},
                answer_strategy="security_redirect",
            )

        t0 = time.perf_counter()
        focus: SessionFocus | None = None
        try:
            if conversation_history and conversation_history.strip():
                plan, focus = await asyncio.gather(
                    self.analyzer.analyze(
                        user_query,
                        conversation_history=conversation_history,
                        model=model,
                    ),
                    self._session_focus.infer(
                        user_query,
                        conversation_history,
                        model=model,
                    ),
                )
            else:
                plan = await self.analyzer.analyze(
                    user_query,
                    conversation_history=None,
                    model=model,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("rag.pipeline.query_focus_failed_fallback", error=str(exc))
            try:
                plan = await self.analyzer.analyze(
                    user_query,
                    conversation_history=conversation_history,
                    model=model,
                )
            except Exception as exc2:  # noqa: BLE001
                log.warning("rag.pipeline.query_analyze_failed_fallback", error=str(exc2))
                plan = QueryPlan(
                    rewritten_query=(user_query or "").strip(),
                    intent="mixed",
                    chef_slugs=[],
                    ingredient_slugs=[],
                    category_slugs=[],
                    keyword_slugs=[],
                    focus_section_kinds=[],
                    needs_context_after=False,
                )
            focus = None
        focus = _merge_objection_boost(user_query, conversation_history, focus)
        focus = _merge_follow_up_boost(user_query, focus)
        # Détecter une requête "humeur" sans ingrédient explicite : ne pas hériter
        # des slugs de l'historique (l'utilisateur a changé de sujet).
        _query_has_ingredient = bool(
            plan.ingredient_slugs
            or re.search(
                r"(?i)\b(?:avec du|avec de la|avec de l'|avec des|à base de|"
                r"au|aux|recette de|recette au|recette aux)\s+\w",
                user_query or "",
            )
        )
        _is_mood_query = bool(
            _MOOD_QUERY_RE.search(user_query or "")
            and not _query_has_ingredient
        )
        _conv_to_scan = None if _is_mood_query else conversation_history
        plan = supplement_ingredient_slugs(plan, user_query, _conv_to_scan)
        # Si l'utilisateur exclut explicitement l'ingrédient en cours, vider les slugs.
        if plan.ingredient_slugs and _NEGATIVE_INGREDIENT_RE.search(user_query or ""):
            log.info(
                "rag.pipeline.ingredient_slugs_cleared_negative_constraint",
                ingredient_slugs=plan.ingredient_slugs,
                query=user_query[:80],
            )
            plan = plan.model_copy(update={"ingredient_slugs": []})
        # SOTA : sur une requête nommant un plat connu, retirer les ingrédients
        # spéculatifs (non tapés) inventés par le LLM (ex. manouche→zaatar/pain-pita).
        _kept_ing = _drop_speculative_ingredients(plan.ingredient_slugs, user_query)
        if _kept_ing != plan.ingredient_slugs:
            log.info(
                "rag.pipeline.dish_query_dropped_speculative_ingredients",
                query=user_query[:80],
                before=plan.ingredient_slugs,
                after=_kept_ing,
            )
            plan = plan.model_copy(update={"ingredient_slugs": _kept_ing})
        timings["query_understanding_ms"] = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        base_q = (plan.rewritten_query or user_query or "").strip()
        # Plat nommé connu : la formulation BRUTE de l'utilisateur (canonicalisée)
        # cible mieux que la réécriture du LLM, qui ajoute des termes diluants
        # (ex. "libanais") et perd l'article du plat. Vérifié via diagnose-retrieval :
        # "recette manaiche" -> 1474718 rang 1, mais "recette manaiche libanais" -> faux.
        if _requested_dish_terms(user_query):
            _pin = _primary_dish_pin(user_query)
            _dish_canon = _primary_dish_canonical(user_query)
            if _pin:
                # Plat épinglé (translittération instable) : requête déterministe
                # plat+chef qui pointe l'article connu de façon fiable.
                base_q = _pin
            elif _dish_canon and not plan.chef_slugs:
                # Plat nommé sans chef précisé : requête canonique PROPRE. On
                # strippe le bruit conversationnel ("je voudrais un X classique",
                # "je veux du X") qui diluait le retrieval -> abstention.
                base_q = f"recette {_dish_canon}"
            else:
                # Plat + chef (ou cas particulier) : garder la formulation (le
                # chef désambiguïse, ex. taboulé de Kamal Mouzawak).
                _canon_raw = _canonicalize_dish_aliases((user_query or "").strip())
                if _canon_raw:
                    if len(_canon_raw.split()) <= 2 and "recette" not in _canon_raw.lower():
                        _canon_raw = f"recette {_canon_raw}"
                    base_q = _canon_raw
        q = _expand_search_q_with_ingredients(base_q, plan)
        if focus and (focus.search_boost_phrase or "").strip():
            q = f"{q} {(focus.search_boost_phrase or '').strip()}".strip()
        force_broaden = bool(focus and focus.suggest_broaden_corpus_search)
        wants_different = bool(
            focus and focus.user_wants_different_article
        ) or wants_another_recipe(user_query)
        excluded: list[int] = []
        if session_id and wants_different:
            try:
                last_id = await sessions_store.last_offered_article_external_id(
                    session_id
                )
                if last_id is not None:
                    excluded = [last_id]
            except Exception as exc:  # noqa: BLE001
                log.warning("rag.pipeline.session_exclusions_failed", error=str(exc))
        try:
            hits = await self._retrieve_hits(
                session,
                user_query,
                plan,
                q,
                rerank_top_n,
                excluded,
                force_corpus_broaden=force_broaden,
            )
            if not hits and excluded and not wants_different:
                log.info(
                    "rag.pipeline.retrieval_retry_no_session_article_exclusions",
                    n_excluded=len(excluded),
                )
                hits = await self._retrieve_hits(
                    session,
                    user_query,
                    plan,
                    q,
                    rerank_top_n,
                    [],
                    force_corpus_broaden=force_broaden,
                )
            elif not hits and excluded and wants_different:
                log.info(
                    "rag.pipeline.no_retry_user_wants_different",
                    n_excluded=len(excluded),
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("rag.pipeline.retrieval_failed")
            hits = []
        timings["retrieval_ms"] = int((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        rq = q
        hits_for_rerank = hits
        if self.settings.rag_prererank_interleave and len(hits) > 1:
            hits_for_rerank = _interleave_hits_by_article(hits)
            n_art = len({h.article_external_id for h in hits})
            if n_art > 1 and len(hits_for_rerank) == len(hits):
                log.info(
                    "rag.pipeline.prererank_interleave",
                    n_hits=len(hits),
                    n_articles=n_art,
                )
        try:
            reranked = await self.reranker.rerank(
                rq, hits_for_rerank, top_n=rerank_top_n
            )
        except Exception as exc:  # noqa: BLE001
            # Cohere indispo, timeout, quota, etc. — on continue sans rerank.
            log.warning("rag.pipeline.rerank_failed_fallback", error=str(exc))
            reranked = [
                RerankedHit(hit=h, rerank_score=float(h.score_rrf or 0.0))
                for h in hits_for_rerank[:rerank_top_n]
            ]
        _reranked_above_thresh = [
            r for r in reranked if r.rerank_score >= self.settings.rag_min_rerank_score
        ]
        if not _reranked_above_thresh:
            _best_score = max((r.rerank_score for r in reranked), default=0.0)
            if plan.ingredient_slugs and _best_score < 0.10:
                # Tous les résultats sont très mauvais pour une requête ingrédient :
                # préférer un message "non trouvé" propre plutôt que des chunks hors-sujet.
                reranked = []
            else:
                reranked = reranked[: max(1, rerank_top_n // 2)]
        else:
            reranked = _reranked_above_thresh
        if self.settings.rag_article_rerank_enabled and len(reranked) > 1:
            try:
                article_docs = _article_rerank_documents(
                    reranked,
                    max_articles=self.settings.rag_article_rerank_max_articles,
                    max_chunks_per_article=self.settings.rag_article_rerank_max_chunks_per_article,
                )
                article_ranked = await self.reranker.rerank(
                    rq,
                    article_docs,
                    top_n=min(len(article_docs), self.settings.rag_article_rerank_max_articles),
                    cost_step="article_rerank",
                )
                reranked = _apply_article_rerank(
                    reranked,
                    article_ranked,
                    keep_top_articles=self.settings.rag_article_rerank_keep_top_articles,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("rag.pipeline.article_rerank_failed_fallback", error=str(exc))
        reranked = _apply_source_priority(reranked, user_query)
        if plan.ingredient_slugs:
            ing_reranked = filter_reranked_by_ingredient_slugs(
                reranked,
                plan.ingredient_slugs,
                retrieval_hits=hits,
            )
            if ing_reranked:
                reranked = ing_reranked
            elif reranked and hits:
                valid_aids = {int(h.article_external_id) for h in hits}
                reranked = [
                    r
                    for r in reranked
                    if int(r.hit.article_external_id) in valid_aids
                ]
                log.info(
                    "rag.pipeline.ingredient_rerank_kept_sql_prefilter",
                    ingredient_slugs=plan.ingredient_slugs,
                    n_reranked=len(reranked),
                )
            elif reranked:
                log.warning(
                    "rag.pipeline.ingredient_rerank_filter_empty",
                    ingredient_slugs=plan.ingredient_slugs,
                )
                reranked = []
            # Filtre anti-"sauce collection" : un article dont le titre est une liste
            # de sauces/condiments (ex. "Sauces : tarator, toum, yaourt, concombre")
            # ne constitue pas une "vraie recette" au sens de l'utilisateur.
            # On l'élimine si des articles plus pertinents existent.
            if reranked and re.match(
                r"(?i)^sauces?\s*[:\–-]",
                reranked[0].hit.article_title or "",
            ):
                non_sauce = [
                    r for r in reranked
                    if not re.match(
                        r"(?i)^sauces?\s*[:\–-]",
                        r.hit.article_title or "",
                    )
                ]
                if non_sauce:
                    log.info(
                        "rag.pipeline.sauce_collection_filtered",
                        filtered_title=(reranked[0].hit.article_title or "")[:80],
                        n_remaining=len(non_sauce),
                    )
                    reranked = non_sauce
            # Ranking par CENTRALITÉ de l'ingrédient : titre > ingredients_list >
            # simple mention (corrige « recette tomate » -> blette au lieu de kebbé).
            before_top = reranked[0].hit.article_title if reranked else None
            reranked = rerank_by_ingredient_centrality(
                reranked, plan.ingredient_slugs, retrieval_hits=hits
            )
            if reranked and reranked[0].hit.article_title != before_top:
                log.info(
                    "rag.pipeline.ingredient_centrality_reordered",
                    ingredient_slugs=plan.ingredient_slugs,
                    new_top=(reranked[0].hit.article_title or "")[:80],
                )
        timings["rerank_ms"] = int((time.perf_counter() - t2) * 1000)

        t3 = time.perf_counter()
        base2_fallback_used = False
        base2_recipe = _match_base2_recipe(user_query)
        # Plat nommé mais absent de l'OLJ ? (aliases OU texte libre). Le fallback
        # base2/génération ne se déclenche que dans ce cas (ou corpus vide) — pas
        # quand l'OLJ a réellement le plat (sinon on l'écraserait).
        cand_dish = _fallback_dish_name(user_query, plan)
        req_terms = _requested_dish_terms(user_query)
        if req_terms:
            # Plat connu (aliases) : la détection par alias fait AUTORITÉ — elle gère
            # les translittérations (ex. « manouche » ↔ titre « mana'ichs »), là où
            # une comparaison texte brute échouerait et déclencherait à tort le fallback.
            dish_absent = not _olj_has_named_dish(user_query, reranked)
        elif cand_dish is not None:
            dish_absent = not _olj_has_dish_text(cand_dish, reranked)
        else:
            dish_absent = False
        trigger_fallback = (not reranked) or dish_absent

        async def _normal_generate() -> GroundedAnswer:
            thread_summary = (focus.thread_summary or "").strip() if focus else ""
            return await self.generator.generate(
                user_query,
                reranked,
                conversation_history=conversation_history,
                session_thread_summary=thread_summary or None,
                model=model,
                required_ingredient_slugs=plan.ingredient_slugs or None,
            )

        async def _empty_answer() -> GroundedAnswer:
            corpus_ok = await _probe_corpus_searchable(self.retriever, session)
            return self.generator.build_empty_retrieval_answer(
                user_query,
                required_ingredient_slugs=plan.ingredient_slugs or None,
                corpus_searchable=corpus_ok,
            )

        # Stratégie de réponse explicite (observabilité : une seule source de vérité
        # du chemin pris, tracée + exposée). Voir log "rag.pipeline.decision".
        answer_strategy = "normal_generate"
        if (
            not reranked
            and plan.ingredient_slugs
            and wants_different
            and base2_recipe is None
        ):
            answer = _build_no_alternate_ingredient_answer(plan.ingredient_slugs)
            answer_strategy = "no_alternate_ingredient"
        elif trigger_fallback and base2_recipe is not None:
            answer = _build_base2_last_resort_answer(
                user_query=user_query,
                base2_recipe=base2_recipe,
                reranked=reranked,
            )
            base2_fallback_used = True
            answer_strategy = "base2_last_resort"
            log.info(
                "rag.pipeline.base2_last_resort_used",
                query=user_query[:80],
                base2_recipe=base2_recipe.get("name"),
                has_olj_suggestion=bool(answer.recipe_card),
            )
        elif trigger_fallback and cand_dish is not None:
            generated = await get_or_generate_recipe(session, cand_dish)
            if generated is not None:
                answer = _build_generated_recipe_answer(
                    user_query=user_query,
                    generated=generated,
                    reranked=reranked,
                )
                answer_strategy = (
                    "generated_cached" if generated.get("_cached") else "generated_new"
                )
                log.info(
                    "rag.pipeline.generated_recipe_used",
                    query=user_query[:80],
                    dish=generated.get("name"),
                    cached=generated.get("_cached"),
                )
            elif not reranked:
                answer = await _empty_answer()
                answer_strategy = "empty_no_generation"
            else:
                answer = await _normal_generate()
                answer_strategy = "normal_generate_dish_absent"
        elif not reranked:
            answer = await _empty_answer()
            answer_strategy = "empty_retrieval"
        else:
            answer = await _normal_generate()
        # Gating de pertinence de la carte : si l'utilisateur nomme un plat précis
        # et que la carte proposée n'y correspond pas (titre), on la retire plutôt
        # que d'afficher une fiche hors-sujet (bug "manouche -> Lahm bi aajine").
        if answer.recipe_card is not None:
            _requested = _requested_dish_terms(user_query)
            if _requested and not _card_title_matches_requested_dish(
                answer.recipe_card.title, _requested
            ):
                log.info(
                    "rag.pipeline.recipe_card_dropped_dish_mismatch",
                    query=user_query[:80],
                    card_title=(answer.recipe_card.title or "")[:80],
                )
                answer = answer.model_copy(
                    update={"recipe_card": None, "recipe_card_secondary": None}
                )

        # Si un article-recette pertinent a été trouvé mais sans carte émise
        # (le LLM a décrit au lieu de carter), synthétiser la carte depuis l'article.
        answer = _ensure_recipe_card(
            answer, reranked, user_query, self.settings.rag_min_rerank_score
        )

        timings["generation_ms"] = int((time.perf_counter() - t3) * 1000)
        timings["total_ms"] = sum(timings.values())

        # ── Décision de routage : UNE ligne structurée par requête (rien louper).
        # Capture tous les signaux qui ont déterminé la stratégie -> debuggable
        # a posteriori sans rejouer la requête.
        log.info(
            "rag.pipeline.decision",
            query=user_query[:120],
            strategy=answer_strategy,
            intent=plan.intent,
            n_hits=len(hits),
            n_reranked=len(reranked),
            dish_absent=bool(dish_absent),
            trigger_fallback=bool(trigger_fallback),
            requested_dish=bool(req_terms),
            cand_dish=cand_dish,
            base2_matched=(base2_recipe.get("name") if base2_recipe else None),
            ingredient_slugs=plan.ingredient_slugs,
            chef_slugs=plan.chef_slugs,
            has_card=bool(answer.recipe_card),
            confidence=round(float(answer.confidence or 0.0), 3),
        )
        log.info(
            "rag.pipeline.completed",
            query=user_query[:80],
            intent=plan.intent,
            strategy=answer_strategy,
            n_hits=len(hits),
            n_reranked=len(reranked),
            timings_ms=timings,
        )
        cost_acc = get_request_cost()
        cost_breakdown = cost_acc.to_dict() if cost_acc is not None else None
        if cost_breakdown:
            log.info(
                "rag.pipeline.cost",
                estimated_usd=cost_breakdown.get("estimated_usd"),
            )
        return PipelineResult(
            plan=plan, hits=hits, reranked=reranked,
            answer=answer, timings_ms=timings, is_base2_fallback=base2_fallback_used,
            cost_breakdown=cost_breakdown, answer_strategy=answer_strategy,
        )

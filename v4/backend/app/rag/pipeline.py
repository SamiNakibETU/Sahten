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

from .. import sessions as sessions_store
from ..llm.response_generator import (
    GroundedAnswer,
    GroundedSentence,
    RecipeCard,
    ResponseGenerator,
    validate_grounding,
)
from ..llm.query_understanding import QueryAnalyzer, QueryPlan
from ..llm.session_focus import SessionFocus, SessionFocusAnalyzer
from ..settings import get_settings
from .embeddings import OpenAIEmbeddings
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
            suggest_broaden_corpus_search=True,
        )
    boost = f"{(focus.search_boost_phrase or '').strip()} {extra}".strip()
    return focus.model_copy(
        update={
            "search_boost_phrase": boost,
            "suggest_broaden_corpus_search": True,
            "user_wants_different_article": True,
            "thread_summary": (
                (focus.thread_summary or "").strip()
                + " [Objection] Maintenir l'ingrédient du fil ; prioriser une fiche où il apparaît au texte."
            )[:400],
        }
    )


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


def _expand_query_with_aliases(q: str) -> str:
    """Ajoute des alias orthographiques à la requête documentaire."""
    base = (q or "").strip()
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


def _match_base2_recipe(user_query: str) -> dict[str, Any] | None:
    requested = _extract_explicit_recipe_name(user_query)
    if not requested:
        return None
    recipes = _load_base2_recipes()
    if not recipes:
        return None
    wanted = _base2_canonicalize_aliases(requested)
    for recipe in recipes:
        if wanted == recipe.get("name_norm", ""):
            return recipe
    return None


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
        GroundedSentence(
            text="Désolé, je n'ai pas cette recette dans mes carnets",
            source_chunk_ids=[],
        )
    ]
    if details:
        sentences.append(
            GroundedSentence(
                text=f"En dernier recours, voici une version courte de {base2_name} ({details}).",
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
                text=f'Pour la version OLJ indexée, la fiche « {olj_title} » est la plus proche disponible.',
                source_chunk_ids=[olj_chunk_id],
            )
        )
    follow_up = (
        f'Si vous voulez, je peux vous guider pas à pas sur la fiche OLJ « {olj_title} ».'
        if olj_title
        else "Si vous voulez, je peux chercher une autre variante OLJ indexée."
    )
    answer = GroundedAnswer(
        answer_sentences=sentences,
        recipe_card=recipe_card,
        recipe_card_secondary=None,
        chef_card=None,
        follow_up=follow_up,
        confidence=0.42,
    )
    return validate_grounding(answer, reranked, user_query=user_query)


def _expand_search_q_with_ingredients(base_q: str, plan: QueryPlan) -> str:
    """Aligne requête BM25/embedding sur les mots d’ingrédient s’ils ne sont pas déjà dans q."""
    qn = (base_q or "").strip()
    qlow = qn.lower()
    extra: list[str] = []
    for s in plan.ingredient_slugs or []:
        w = s.replace("-", " ").strip()
        if not w:
            continue
        if w.lower() in qlow:
            continue
        extra.append(w)
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
                        chef_slugs=[],
                        ingredient_slugs=[],
                        category_slugs=[],
                        keyword_slugs=[],
                        final_limit=final_limit,
                        exclude_article_external_ids=excl,
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
        return hits

    async def answer(
        self,
        session: AsyncSession,
        user_query: str,
        *,
        rerank_top_n: int | None = None,
        session_id: str | None = None,
        conversation_history: str | None = None,
    ) -> PipelineResult:
        timings: dict[str, int] = {}
        rerank_top_n = rerank_top_n or self.settings.rag_rerank_top_k

        t0 = time.perf_counter()
        focus: SessionFocus | None = None
        try:
            if conversation_history and conversation_history.strip():
                plan, focus = await asyncio.gather(
                    self.analyzer.analyze(
                        user_query, conversation_history=conversation_history
                    ),
                    self._session_focus.infer(
                        user_query, conversation_history
                    ),
                )
            else:
                plan = await self.analyzer.analyze(
                    user_query, conversation_history=None
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("rag.pipeline.query_focus_failed_fallback", error=str(exc))
            try:
                plan = await self.analyzer.analyze(
                    user_query, conversation_history=conversation_history
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
        timings["query_understanding_ms"] = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        base_q = (plan.rewritten_query or user_query or "").strip()
        q = _expand_search_q_with_ingredients(base_q, plan)
        if focus and (focus.search_boost_phrase or "").strip():
            q = f"{q} {(focus.search_boost_phrase or '').strip()}".strip()
        force_broaden = bool(focus and focus.suggest_broaden_corpus_search)
        excluded: list[int] = []
        if session_id and bool(focus and focus.user_wants_different_article):
            try:
                excluded = await sessions_store.recent_article_external_ids(session_id)
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
            if not hits and excluded:
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
        reranked = [
            r for r in reranked if r.rerank_score >= self.settings.rag_min_rerank_score
        ] or reranked[: max(1, rerank_top_n // 2)]
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
                )
                reranked = _apply_article_rerank(
                    reranked,
                    article_ranked,
                    keep_top_articles=self.settings.rag_article_rerank_keep_top_articles,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("rag.pipeline.article_rerank_failed_fallback", error=str(exc))
        reranked = _apply_source_priority(reranked, user_query)
        timings["rerank_ms"] = int((time.perf_counter() - t2) * 1000)

        t3 = time.perf_counter()
        base2_fallback_used = False
        base2_recipe = _match_base2_recipe(user_query)
        if base2_recipe is not None:
            answer = _build_base2_last_resort_answer(
                user_query=user_query,
                base2_recipe=base2_recipe,
                reranked=reranked,
            )
            base2_fallback_used = True
            log.info(
                "rag.pipeline.base2_last_resort_used",
                query=user_query[:80],
                base2_recipe=base2_recipe.get("name"),
                has_olj_suggestion=bool(answer.recipe_card),
            )
        else:
            thread_summary = (
                (focus.thread_summary or "").strip()
                if focus
                else ""
            )
            answer = await self.generator.generate(
                user_query,
                reranked,
                conversation_history=conversation_history,
                session_thread_summary=thread_summary or None,
            )
        timings["generation_ms"] = int((time.perf_counter() - t3) * 1000)
        timings["total_ms"] = sum(timings.values())

        log.info(
            "rag.pipeline.completed",
            query=user_query[:80],
            intent=plan.intent,
            n_hits=len(hits),
            n_reranked=len(reranked),
            timings_ms=timings,
        )
        return PipelineResult(
            plan=plan, hits=hits, reranked=reranked,
            answer=answer, timings_ms=timings, is_base2_fallback=base2_fallback_used,
        )

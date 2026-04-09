"""
Hybrid Retriever (MVP)
======================

Implements a durable retrieval pipeline:
  1) Retrieve (high recall): lexical TF-IDF + optional embeddings
  2) Fuse rankings with RRF (no magic weights)
  3) Rerank (high precision): LLM reranker on topK
  4) Select: dedup + diversity rules (menu) + hard filters

Embeddings are OFF by default (ENABLE_EMBEDDINGS=false).
Enable when: 1000+ recipes OR frequent semantic queries.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.ingredient_normalizer import ingredient_normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity

from ..core.config import get_settings
from ..data.ingredient_normalizer import ingredient_normalizer
from ..data.normalizers import normalize_text
from .citation_quality import sanitize_cited_passage, title_suggests_non_recipe_article
from .editorial_snippets import extract_editorial_snippets
from .retrieval_constants import (
    GENERIC_QUERY_TOKENS,
    MATCH_REASON_KEYWORDS,
    doc_category_matches_inferred,
    ingredient_overlap_is_meaningful,
)
from ..llm.reranker import LLMReranker, RerankCandidate, RerankItem
from ..schemas.canonical import CanonicalCategory, CanonicalRecipeDoc
from ..schemas.query_analysis import QueryAnalysis
from ..schemas.responses import AlternativeMatch, OLJRecommendation, RecipeCard, SharedIngredientProof

logger = logging.getLogger(__name__)


def _extract_targeted_passage(text: str, term: str, max_chars: int) -> str:
    """
    Extract a passage containing the term when possible.
    Splits on sentence boundaries (. ! ?) and prefers the sentence containing term.
    Falls back to text start.
    """
    text_lower = text.lower()
    term_ascii = unidecode(term)
    term_lower = term.lower()
    # Find first occurrence
    pos = text_lower.find(term_lower)
    if pos < 0:
        pos = unidecode(text_lower).find(term_ascii)
    if pos >= 0:
        start = max(0, text.rfind(".", 0, pos), text.rfind("!", 0, pos), text.rfind("?", 0, pos)) + 1
        end = len(text)
        for sep in ".!?":
            i = text.find(sep, pos + 1)
            if i >= 0:
                end = min(end, i + 1)
        excerpt = text[start:end].strip()
        if excerpt and len(excerpt) <= max_chars:
            return excerpt
        if len(excerpt) > max_chars:
            return excerpt[:max_chars].rsplit(" ", 1)[0] + "…"
    return text[:max_chars].strip() + ("…" if len(text) > max_chars else "")


@dataclass
class RetrievalDebug:
    lexical_top: List[Tuple[str, float]]
    semantic_top: List[Tuple[str, float]]
    rrf_top: List[Tuple[str, float]]
    reranked: List[Tuple[str, float]]
    selected: List[str]


class HybridRetriever:
    """
    Retriever backed by a canonical dataset generated offline.

    Features:
      - TF-IDF lexical search (always ON)
      - Embeddings semantic search (optional, OFF by default)
      - LLM reranking (high precision)
      - Hot reload for CMS webhook integration
    """

    def __init__(
        self,
        *,
        olj_canonical_path: Optional[str] = None,
        base2_path: Optional[str] = None,
        enable_llm_rerank: bool = True,
    ):
        settings = get_settings()

        base_dir = Path(__file__).parent.parent.parent.parent
        self.olj_canonical_path = olj_canonical_path or str(
            base_dir / settings.olj_canonical_path
        )
        self.olj_raw_path = str(base_dir / settings.olj_data_path)
        self.base2_path = base2_path or str(base_dir / settings.base2_data_path)

        self.enable_llm_rerank = enable_llm_rerank
        self.reranker = LLMReranker()

        self.olj_docs: List[CanonicalRecipeDoc] = []
        self.base2_recipes: Dict[str, List[Dict[str, Any]]] = {}

        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        
        # Embeddings storage (populated if enable_embeddings=True)
        self._embeddings_matrix: Optional[np.ndarray] = None

        self._load_data()
        self._build_lexical_index()
        
        # Build embeddings index if enabled
        if settings.enable_embeddings:
            self._build_semantic_index()

    def reload(self) -> None:
        """
        Hot reload data and indices.
        
        Called after CMS webhook adds new recipes.
        Thread-safe: builds new indices before swapping.
        """
        logger.info("Reloading retriever data and indices...")
        
        # Reload data
        self._load_data()
        
        # Rebuild indices
        self._build_lexical_index()
        
        settings = get_settings()
        if settings.enable_embeddings:
            self._build_semantic_index()
        
        logger.info("Retriever reloaded: %d docs", len(self.olj_docs))

    def _load_data(self) -> None:
        """Load canonical OLJ docs and Base2 recipes."""
        # Load canonical OLJ docs
        try:
            raw = json.loads(Path(self.olj_canonical_path).read_text(encoding="utf-8"))
            self.olj_docs = [CanonicalRecipeDoc(**d) for d in raw]
            logger.info("Loaded %s canonical OLJ docs from %s", len(self.olj_docs), self.olj_canonical_path)
        except FileNotFoundError:
            logger.warning(
                "Canonical OLJ dataset not found: %s. Falling back to raw dataset.",
                self.olj_canonical_path,
            )
            self.olj_docs = self._build_minimal_canonical_from_raw()
        except Exception as e:
            logger.error("Failed to load canonical OLJ: %s", e)
            self.olj_docs = []

        # Load Base2
        try:
            self.base2_recipes = json.loads(Path(self.base2_path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("Base2 data file not found: %s", self.base2_path)
            self.base2_recipes = {}
        except Exception as e:
            logger.error("Failed to load Base2: %s", e)
            self.base2_recipes = {}

    def _build_minimal_canonical_from_raw(self) -> List[CanonicalRecipeDoc]:
        """Build minimal CanonicalRecipeDoc list from raw OLJ data (fallback)."""
        try:
            raw = json.loads(Path(self.olj_raw_path).read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return []
        except Exception as e:
            logger.error("Failed to load raw OLJ dataset for fallback: %s", e)
            return []

        docs: List[CanonicalRecipeDoc] = []
        for item in raw:
            url = item.get("url")
            if not url:
                continue

            title = (item.get("title") or item.get("recipe_section_title") or "").strip()
            if not title:
                slug = str(url).rstrip("/").split("/")[-1].replace("-", " ").strip()
                title = slug[:1].upper() + slug[1:] if slug else "Sans titre"

            enrichment = item.get("enrichment") or {}

            is_recipe = bool(enrichment.get("is_recipe", True))
            is_lebanese = bool(enrichment.get("is_lebanese", True))
            cuisine_type = enrichment.get("cuisine_type") or None

            cat_raw = (enrichment.get("category") or item.get("category") or "").strip().lower()
            diff_raw = (enrichment.get("difficulty") or item.get("difficulty") or "").strip().lower()

            category_map = {
                "mezze": "mezze_froid", "mezze froid": "mezze_froid",
                "mezze_chaud": "mezze_chaud", "mezze chaud": "mezze_chaud",
                "plat principal": "plat_principal", "plat_principal": "plat_principal",
                "dessert": "dessert", "entrée": "entree", "entree": "entree",
                "salade": "salade", "soupe": "soupe", "sauce": "sauces", "sauces": "sauces",
                "cocktail": "boisson", "apéro": "boisson", "apero": "boisson", "boisson": "boisson",
            }
            difficulty_map = {"facile": "facile", "moyenne": "moyenne", "difficile": "difficile"}

            category_canonical = category_map.get(cat_raw, "autre")
            difficulty_canonical = difficulty_map.get(diff_raw, "non_specifie")

            chef_name = (item.get("chef_name") or "").strip() or None
            tags = item.get("tags") or []
            main_ingredients = enrichment.get("main_ingredients") or []
            aliases = enrichment.get("aliases") or []

            desc = (item.get("recipe_description") or "").strip()
            ingredients = " ".join((item.get("ingredients") or [])[:80])
            instructions = " ".join((item.get("instructions") or [])[:120])
            tags_str = " ".join(tags) if isinstance(tags, list) else str(tags)
            alias_str = " ".join(aliases) if isinstance(aliases, list) else str(aliases)
            mains_str = " ".join(main_ingredients) if isinstance(main_ingredients, list) else str(main_ingredients)

            search_text = " ".join(
                x for x in [title, chef_name or "", desc, ingredients, instructions, tags_str, alias_str, mains_str] if x
            ).strip()
            if not search_text:
                search_text = title

            try:
                docs.append(
                    CanonicalRecipeDoc(
                        url=url,
                        title=title,
                        chef_name=chef_name,
                        cuisine_type=cuisine_type,
                        is_lebanese=is_lebanese,
                        is_recipe=is_recipe,
                        category_canonical=category_canonical,
                        difficulty_canonical=difficulty_canonical,
                        tags=[t for t in tags if t] if isinstance(tags, list) else [],
                        main_ingredients=[i for i in main_ingredients if i] if isinstance(main_ingredients, list) else [],
                        aliases=[a for a in aliases if a] if isinstance(aliases, list) else [],
                        search_text=search_text,
                        raw_category=cat_raw or None,
                        raw_difficulty=diff_raw or None,
                        raw_enrichment_present=bool(enrichment),
                    )
                )
            except Exception:
                continue

        # Dedup by URL
        by_url: Dict[str, CanonicalRecipeDoc] = {}
        for d in docs:
            by_url[str(d.url)] = by_url.get(str(d.url), d)
        return list(by_url.values())

    def _build_lexical_index(self) -> None:
        """Build TF-IDF index for lexical search."""
        texts = [d.search_text for d in self.olj_docs]
        if not texts:
            self._tfidf = None
            self._tfidf_matrix = None
            return

        # Word n-grams for phrase matching
        self._tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            analyzer="word",
            min_df=1,
        )
        word_matrix = self._tfidf.fit_transform(texts)

        # Char n-grams for typo tolerance
        char_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
        )
        char_matrix = char_vectorizer.fit_transform(texts)

        self._tfidf_matrix = (word_matrix, char_matrix, char_vectorizer)

    def _build_semantic_index(self) -> None:
        """
        Build embeddings index for semantic search.

        Only called if settings.enable_embeddings=True.
        OpenAI, mock, etc. via embedding_client.get_embeddings.
        """
        settings = get_settings()

        try:
            from ..services.embedding_client import get_embeddings
            from ..services.embedding_cache import (
                save_embedding_matrix,
                try_load_embedding_matrix,
            )

            texts = [d.search_text for d in self.olj_docs]
            urls = [str(d.url) for d in self.olj_docs]
            if not texts:
                self._embeddings_matrix = None
                return

            cache_key_model = f"{settings.embedding_provider}:{settings.embedding_model}"
            cache_dir = settings.data_dir / settings.embedding_cache_dir
            if settings.embedding_cache_enabled and settings.embedding_provider == "openai":
                loaded = try_load_embedding_matrix(
                    cache_dir=cache_dir,
                    urls=urls,
                    embedding_model=cache_key_model,
                    expected_shape_n=len(texts),
                )
                if loaded is not None:
                    self._embeddings_matrix = loaded
                    return

            logger.info("Building embeddings for %d documents...", len(texts))
            embeddings = get_embeddings(texts)
            self._embeddings_matrix = np.array(embeddings)
            logger.info("Embeddings built: shape %s", self._embeddings_matrix.shape)
            if settings.embedding_cache_enabled and settings.embedding_provider == "openai":
                save_embedding_matrix(
                    cache_dir=cache_dir,
                    urls=urls,
                    embedding_model=cache_key_model,
                    matrix=self._embeddings_matrix,
                )

        except Exception as e:
            logger.error("Failed to build embeddings: %s", e)
            self._embeddings_matrix = None

    def _lexical_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Retrieve documents using TF-IDF similarity."""
        if not self._tfidf or not self._tfidf_matrix:
            return []
        
        word_matrix, char_matrix, char_vectorizer = self._tfidf_matrix
        q_word = self._tfidf.transform([query])
        q_char = char_vectorizer.transform([query])

        s_word = cosine_similarity(q_word, word_matrix).ravel()
        s_char = cosine_similarity(q_char, char_matrix).ravel()
        scores = (s_word + s_char) / 2.0

        MIN_LEXICAL_SCORE = 0.05
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > MIN_LEXICAL_SCORE]

    def _semantic_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Retrieve documents using embedding similarity.
        
        Only active if settings.enable_embeddings=True.
        Returns empty list otherwise.
        """
        settings = get_settings()
        
        if not settings.enable_embeddings:
            return []
        
        if self._embeddings_matrix is None:
            return []
        
        try:
            from ..services.embedding_client import get_embeddings
            
            # Get query embedding
            query_emb = get_embeddings([query])[0]
            query_vec = np.array(query_emb)
            
            # Compute cosine similarity
            sims = self._embeddings_matrix @ query_vec
            sims = sims / (np.linalg.norm(self._embeddings_matrix, axis=1) * np.linalg.norm(query_vec) + 1e-9)
            
            idx = np.argsort(-sims)[:top_k]
            return [(int(i), float(sims[i])) for i in idx if sims[i] > 0]
            
        except Exception as e:
            logger.error("Semantic retrieval failed: %s", e)
            return []

    @staticmethod
    def _rrf_fuse(
        ranked_lists: List[List[Tuple[int, float]]], *, k: int = 60
    ) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion (RRF) - combines rankings without weight tuning."""
        scores: Dict[int, float] = {}
        for lst in ranked_lists:
            for rank, (idx, _) in enumerate(lst, start=1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        fused = sorted(scores.items(), key=lambda x: -x[1])
        return [(idx, float(score)) for idx, score in fused]

    async def _rerank(
        self,
        raw_query: str,
        fused: List[Tuple[int, float]],
        top_k: int,
        *,
        category_allowlist: Optional[set[str]] = None,
        must_contain_any: Optional[set[str]] = None,
        rerank_user_query: Optional[str] = None,
        session_rerank_prefix: str = "",
    ) -> List[RerankItem]:
        """Rerank : cross-encoder optionnel puis LLM (ou CE seul si configuré)."""
        settings = get_settings()
        scan_cap = top_k * 3 if settings.enable_cross_encoder_rerank else top_k
        fused_slice = fused if category_allowlist else fused[:scan_cap]

        candidates: List[RerankCandidate] = []
        for idx, _ in fused_slice:
            d = self.olj_docs[idx]
            if category_allowlist and d.category_canonical not in category_allowlist:
                continue
            if must_contain_any:
                text = (d.search_text or "").lower()
                if not any(term in text for term in must_contain_any):
                    continue
            candidates.append(
                RerankCandidate(
                    url=str(d.url),
                    title=d.title,
                    search_text_excerpt=d.search_text[:600],
                    category=d.category_canonical,
                    cuisine_type=d.cuisine_type,
                    is_lebanese=d.is_lebanese,
                )
            )
            if len(candidates) >= scan_cap:
                break

        q = (rerank_user_query or raw_query).strip()

        if settings.enable_cross_encoder_rerank and candidates:
            from ..llm.cross_encoder_rerank import score_pairs

            texts = [f"{c.title}\n{c.search_text_excerpt}" for c in candidates]
            ce_scores = score_pairs(q, texts, model_name=settings.cross_encoder_model)
            if ce_scores and len(ce_scores) == len(candidates):
                order = sorted(range(len(ce_scores)), key=lambda i: -ce_scores[i])
                candidates = [candidates[i] for i in order]
                ce_scores = [ce_scores[i] for i in order]
                if not settings.rerank_llm_after_cross_encoder:
                    mx = max(ce_scores) if ce_scores else 1e-9
                    items: List[RerankItem] = []
                    for c, s in zip(candidates, ce_scores):
                        doc = next((d for d in self.olj_docs if str(d.url) == c.url), None)
                        term = (c.title or "recette").split()[0] if c.title else "recette"
                        excerpt = (
                            _extract_targeted_passage(doc.search_text or "", term, max_chars=180)
                            if doc
                            else None
                        )
                        items.append(
                            RerankItem(
                                url=c.url,
                                score=float(min(1.0, max(0.15, s / mx))),
                                cited_passage=excerpt,
                            )
                        )
                    return items[:top_k]

                candidates = candidates[: max(8, min(top_k, 10))]

        return await self.reranker.rerank(
            q,
            candidates,
            max_items=min(10, len(candidates)),
            session_prefix=session_rerank_prefix or "",
        )

    def _select_cards(
        self,
        analysis: QueryAnalysis,
        reranked: List[RerankItem],
        *,
        max_results: int,
        category_allowlist: Optional[set[str]] = None,
        exclude_urls: Optional[set[str]] = None,
    ) -> List[RecipeCard]:
        """Select final recipe cards with deduplication and grounding."""
        used_urls = set()
        if exclude_urls:
            used_urls |= set(exclude_urls)
        cards: List[RecipeCard] = []
        
        for it in reranked:
            if it.url in used_urls:
                continue
            used_urls.add(it.url)

            doc = next((d for d in self.olj_docs if str(d.url) == it.url), None)
            if not doc:
                continue

            if not doc.is_recipe:
                continue
            if title_suggests_non_recipe_article(doc.title):
                logger.debug("Skip non-recipe-looking title for card: %s", doc.title[:80])
                continue
            if category_allowlist and doc.category_canonical not in category_allowlist:
                continue

            safe_passage = sanitize_cited_passage(
                it.cited_passage,
                title=doc.title,
            )
            passage_final = safe_passage or it.cited_passage
            lead, story = extract_editorial_snippets(
                doc.search_text or "",
                passage_final,
            )

            cards.append(
                RecipeCard(
                    source="olj",
                    title=doc.title,
                    url=str(doc.url),
                    chef=doc.chef_name,
                    category=doc.category_canonical,
                    image_url=doc.image_url,  # Image de la recette
                    cited_passage=passage_final,
                    recipe_lead=lead or None,
                    story_snippet=story or None,
                )
            )
            if len(cards) >= max_results:
                break
        return cards

    _DESSERT_HINTS = (
        "dessert",
        "sucre",
        "sucré",
        "sucree",
        "gateau",
        "gâteau",
        "cookie",
        "biscuit",
        "patisserie",
        "pâtisserie",
        "douceur",
        "chocolat",
    )

    @staticmethod
    def _plan_rerank_prefix(analysis: QueryAnalysis) -> str:
        p = analysis.plan
        if p is None:
            return ""
        rf = (p.retrieval_focus or "").strip()
        rf_bit = f" retrieval_focus={rf[:140]}" if rf else ""
        return f"[QueryPlan] task={p.task} cuisine_scope={p.cuisine_scope} course={p.course}{rf_bit}"

    @staticmethod
    def _allowlist_for_intent(analysis: QueryAnalysis, raw_query: str) -> Optional[set[str]]:
        """Contraintes de catégorie : priorité au QueryPlan, sinon heuristiques legacy."""
        q = unidecode(raw_query or "").lower()
        p = analysis.plan

        if p is not None:
            if p.course == "dessert":
                return {"dessert"}
            if p.course == "plat":
                return {"plat_principal"}
            if p.course == "entree":
                return {"entree"}
            if p.course == "mezze":
                return {"mezze_froid", "mezze_chaud"}
            if p.task == "browse_corpus" and p.cuisine_scope == "lebanese_olj":
                if p.course == "dessert":
                    return {"dessert"}
                if p.course in ("any", "plat") and not any(x in q for x in HybridRetriever._DESSERT_HINTS):
                    return {"plat_principal", "mezze_froid", "mezze_chaud", "entree"}
            return None

        if analysis.intent == "recipe_by_category" and analysis.category:
            return {analysis.category}

        if analysis.intent == "multi_recipe":
            if analysis.category:
                if analysis.category in {"mezze_froid", "mezze_chaud"}:
                    return {"mezze_froid", "mezze_chaud"}
                return {analysis.category}
            if "mezze" in q or "meze" in q:
                return {"mezze_froid", "mezze_chaud"}
            if "dessert" in q or "sucre" in q:
                return {"dessert"}
        if analysis.intent == "recipe_by_mood":
            tags = {unidecode(t).lower() for t in (analysis.mood_tags or [])}
            if "liban" in tags and not any(x in q for x in HybridRetriever._DESSERT_HINTS):
                return {"plat_principal", "mezze_froid", "mezze_chaud", "entree"}
        return None

    def search(
        self, analysis: QueryAnalysis, *, raw_query: str, debug: bool = False
    ) -> Tuple[List[RecipeCard], bool, Optional[dict]]:
        """Synchronous search (without LLM reranking)."""
        settings = get_settings()

        if analysis.intent == "menu_composition":
            recipes, is_base2, dbg = self._search_menu(analysis, raw_query, debug=debug)
            return recipes, is_base2, dbg

        olj_cards, dbg = self._search_olj(analysis, raw_query, debug=debug)

        if olj_cards:
            return olj_cards[: analysis.recipe_count], False, dbg

        base2_cards = self._search_base2(analysis)
        if base2_cards:
            return base2_cards[: analysis.recipe_count], True, dbg

        return [], False, dbg

    def _search_olj(
        self,
        analysis: QueryAnalysis,
        raw_query: str,
        *,
        debug: bool,
        conversation_context: Optional[str] = None,
    ) -> Tuple[List[RecipeCard], Optional[dict]]:
        """Search OLJ docs using hybrid retrieval."""
        settings = get_settings()
        top_k = settings.retrieve_top_k

        # Build retrieval query
        parts: List[str] = [raw_query]
        if analysis.dish_name:
            parts.append(analysis.dish_name)
        parts.extend(analysis.dish_name_variants or [])
        parts.extend(analysis.ingredients or [])
        parts.extend(analysis.mood_tags or [])
        if analysis.chef_name:
            parts.append(analysis.chef_name)
        if analysis.category:
            parts.append(analysis.category)
        retrieval_query = " ".join(p for p in parts if p)
        if conversation_context:
            retrieval_query = (
                f"{retrieval_query}\n\nContexte conversation récent:\n{conversation_context}"
            )

        # Hybrid retrieval
        lexical = self._lexical_retrieve(retrieval_query, top_k=top_k)
        semantic = self._semantic_retrieve(retrieval_query, top_k=top_k)
        fused = self._rrf_fuse([lexical, semantic]) if semantic else lexical

        # Use fused scores directly (reranking happens in async variant)
        reranked = [
            RerankItem(url=str(self.olj_docs[idx].url), score=float(score))
            for idx, score in (fused[:settings.rerank_top_k] if isinstance(fused, list) else fused)
        ]

        allowlist = self._allowlist_for_intent(analysis, raw_query)
        cards = self._select_cards(
            analysis,
            reranked,
            max_results=max(analysis.recipe_count, settings.max_results),
            category_allowlist=allowlist,
        )

        dbg = None
        if debug:
            dbg = {
                "retrieval_query": retrieval_query,
                "lexical_top": [(str(self.olj_docs[i].url), s) for i, s in lexical[:10]],
                "semantic_top": [(str(self.olj_docs[i].url), s) for i, s in semantic[:10]] if semantic else [],
                "rrf_top": [(str(self.olj_docs[i].url), s) for i, s in fused[:10]] if isinstance(fused, list) else [],
                "reranked": [(it.url, it.score) for it in reranked],
                "selected": [c.url for c in cards if c.url],
            }
        return cards, dbg

    async def search_with_rerank(
        self,
        analysis: QueryAnalysis,
        *,
        raw_query: str,
        debug: bool = False,
        exclude_urls: Optional[List[str]] = None,
        conversation_context: Optional[str] = None,
        retrieval_query_boost: Optional[str] = None,
        session_context_dict: Optional[dict] = None,
    ) -> Tuple[List[RecipeCard], bool, Optional[dict]]:
        """
        Async search with LLM reranking.
        
        Args:
            analysis: Query analysis from LLM
            raw_query: Original user query
            debug: Include debug info
            exclude_urls: URLs to exclude (e.g., already proposed in session)
            conversation_context: Résumé des derniers tours (même onglet) pour retrieve + rerank.
            retrieval_query_boost: Phrase additionnelle (ex. sortie query rewriter LLM).
        """
        settings = get_settings()
        exclude_set = set(exclude_urls or [])
        rerank_top_k = settings.retrieve_top_k
        rerank_user_query: Optional[str] = None
        if conversation_context:
            rerank_user_query = (
                "Contexte (échanges précédents dans la même conversation):\n"
                f"{conversation_context}\n\nQuestion actuelle:\n{raw_query}"
            )
        session_rerank_prefix = ""
        if session_context_dict:
            tlist = session_context_dict.get("recent_recipe_titles") or []
            if tlist:
                session_rerank_prefix = (
                    "[Session] Fiches déjà proposées dans ce fil : "
                    + " ; ".join(str(t) for t in tlist[:5] if t)
                )
        if analysis.intent == "recipe_specific":
            rerank_top_k = min(rerank_top_k, 8)
        elif analysis.intent in {"recipe_by_ingredient", "recipe_by_category"}:
            rerank_top_k = min(rerank_top_k, 10)
        elif analysis.intent == "menu_composition":
            rerank_top_k = min(rerank_top_k, 12)

        # Menu composition: category-constrained searches
        if analysis.intent == "menu_composition":
            picked: List[RecipeCard] = []
            used: set[str] = set(exclude_set)  # Include session exclusions

            async def pick_one(cat_allow: set[str], query_hint: str) -> Optional[RecipeCard]:
                local_query = f"{raw_query} {query_hint}".strip()
                if conversation_context:
                    local_query = f"{local_query}\n\nContexte conversation récent:\n{conversation_context}"

                parts: List[str] = [local_query]
                if analysis.dish_name:
                    parts.append(analysis.dish_name)
                parts.extend(analysis.dish_name_variants or [])
                parts.extend(analysis.ingredients or [])
                parts.extend(analysis.mood_tags or [])
                if analysis.chef_name:
                    parts.append(analysis.chef_name)
                retrieval_query = " ".join(p for p in parts if p)

                lexical = self._lexical_retrieve(retrieval_query, top_k=rerank_top_k)
                semantic = self._semantic_retrieve(retrieval_query, top_k=rerank_top_k)
                fused = self._rrf_fuse([lexical, semantic]) if semantic else lexical

                reranked = await self._rerank(
                    raw_query,
                    fused,
                    top_k=rerank_top_k,
                    category_allowlist=cat_allow,
                    rerank_user_query=rerank_user_query,
                    session_rerank_prefix=session_rerank_prefix,
                )
                cards = self._select_cards(
                    analysis,
                    reranked,
                    max_results=3,
                    category_allowlist=cat_allow,
                    exclude_urls=used,
                )
                return cards[0] if cards else None

            mezze = await pick_one({"mezze_froid", "mezze_chaud", "entree"}, "mezze entrée")
            if mezze and mezze.url:
                picked.append(mezze)
                used.add(mezze.url)

            plat = await pick_one({"plat_principal"}, "plat principal")
            if plat and plat.url:
                picked.append(plat)
                used.add(plat.url)

            dessert = await pick_one({"dessert"}, "dessert")
            if dessert and dessert.url:
                picked.append(dessert)
                used.add(dessert.url)

            if len(picked) < 3:
                base2_cards = self._search_base2(QueryAnalysis(**{**analysis.model_dump(), "recipe_count": 3}))
                for c in base2_cards:
                    if len(picked) >= 3:
                        break
                    picked.append(c)
                return picked[:3], bool(base2_cards), {"selected": [c.url for c in picked if c.url]} if debug else None

            return picked[:3], False, {"selected": [c.url for c in picked if c.url]} if debug else None

        # Regular search with rerank
        # Mood tag → culinary keyword expansion for better TF-IDF matching
        _MOOD_KEYWORD_EXPANSION: Dict[str, str] = {
            "leger":       "léger salade frais entrée mezze cru citron menthe tomate fattoush",
            "rapide":      "rapide facile simple sauté poêlé express préparation minute",
            "reconfortant":"réconfortant soupe velouté mijoté chaud hiver lentilles pois chiche",
            "frais":       "frais salade cru été estival citron concombre herbes",
            "hiver":       "hiver chaud soupe velouté lentilles mijot bourgol blé",
            "ete":         "été frais salade cru estival tomate concombre menthe",
            "festif":      "festif fête partage mezze généreux riz viande agneau",
            "traditionnel":"traditionnel libanais authentique classique mémé grandmère",
            "moderne":     "moderne revisité contemporain fusion",
            "convivial":   "partage mezze amis famille généreux plateau",
            "facile":      "facile simple rapide préparation basique ingrédients",
            "copieux":     "copieux généreux plat principal riz viande agneau",
            "chaud":       "chaud soupe ragoût mijoté four",
            "froid":       "froid frais salade mezze cru",
            "liban":       "libanais liban cuisine levantine oriental méditerranéen",
        }

        parts_rw: List[str] = []
        rf = analysis.effective_retrieval_focus()
        if rf:
            parts_rw.append(rf)
        parts_rw.append(raw_query)
        if analysis.dish_name:
            parts_rw.append(analysis.dish_name)
        parts_rw.extend(analysis.dish_name_variants or [])
        parts_rw.extend(analysis.ingredients or [])
        parts_rw.extend(analysis.mood_tags or [])
        # For mood-based queries: expand mood tags to culinary keywords
        if analysis.intent == "recipe_by_mood" and analysis.mood_tags:
            for tag in analysis.mood_tags:
                expanded = _MOOD_KEYWORD_EXPANSION.get(tag)
                if expanded:
                    parts_rw.append(expanded)
        if analysis.chef_name:
            parts_rw.append(analysis.chef_name)
        if analysis.category:
            parts_rw.append(analysis.category)
        retrieval_query = " ".join(p for p in parts_rw if p)
        if retrieval_query_boost:
            retrieval_query = f"{retrieval_query}\n{retrieval_query_boost}".strip()
        if conversation_context:
            retrieval_query = (
                f"{retrieval_query}\n\nContexte conversation récent:\n{conversation_context}"
            )

        lexical = self._lexical_retrieve(retrieval_query, top_k=rerank_top_k)
        semantic = self._semantic_retrieve(retrieval_query, top_k=rerank_top_k)
        fused = self._rrf_fuse([lexical, semantic]) if semantic else lexical

        allowlist = self._allowlist_for_intent(analysis, raw_query)
        must_any = None
        if analysis.intent == "recipe_by_ingredient" and analysis.ingredients:
            must_any = {i.lower() for i in analysis.ingredients if i}

        rerank_shortcircuit = False
        RERANK_SHORTCIRCUIT_LEX = 0.35
        if (
            analysis.intent in {"recipe_specific", "recipe_by_category"}
            and lexical
            and not conversation_context
        ):
            top_idx, lex_score = lexical[0]
            if 0 <= top_idx < len(self.olj_docs):
                top_doc = self.olj_docs[top_idx]
                title_l = (top_doc.title or "").lower()
                dish = (analysis.dish_name or "").lower()
                variants = [dish] + [v.strip().lower() for v in (analysis.dish_name_variants or []) if v]
                query_words = {w for w in raw_query.lower().split() if len(w) >= 4} - GENERIC_QUERY_TOKENS

                dish_in_title = dish and any(v in title_l for v in variants if len(v) >= 3)
                query_word_in_title = query_words and any(w in title_l for w in query_words)

                if dish_in_title or (query_word_in_title and lex_score > RERANK_SHORTCIRCUIT_LEX):
                    rerank_shortcircuit = True
                    take_n = max(3, min(rerank_top_k, len(fused) if isinstance(fused, list) else 3))
                    reranked = [
                        RerankItem(url=str(self.olj_docs[i].url), score=float(s))
                        for i, s in (fused[:take_n] if isinstance(fused, list) else [])
                    ]
                    logger.info("Rerank short-circuit: dish=%s found in title=%s (lex_score=%.3f)",
                                dish, title_l[:60], lex_score)
        if not rerank_shortcircuit:
            plan_prefix = self._plan_rerank_prefix(analysis)
            rr_uq = rerank_user_query
            if plan_prefix:
                base_q = rr_uq if rr_uq else raw_query
                rr_uq = f"{plan_prefix}\n\n{base_q}"
            reranked = await self._rerank(
                raw_query,
                fused,
                top_k=rerank_top_k,
                category_allowlist=allowlist,
                must_contain_any=must_any,
                rerank_user_query=rr_uq,
                session_rerank_prefix=session_rerank_prefix,
            )
        # Filter low scores — threshold is lower for mood/vague queries
        if analysis.intent in {"recipe_by_mood", "recipe_by_diet", "multi_recipe"}:
            MIN_RERANK_SCORE = 0.05
        else:
            MIN_RERANK_SCORE = 0.15
        if not rerank_shortcircuit:
            reranked = [it for it in reranked if it.score >= MIN_RERANK_SCORE]

        cards = self._select_cards(
            analysis,
            reranked,
            max_results=max(analysis.recipe_count, settings.max_results),
            category_allowlist=allowlist,
            exclude_urls=exclude_set,
        )

        dbg = None
        if debug:
            dbg = {
                "retrieval_query": retrieval_query,
                "retrieval_query_boost": retrieval_query_boost,
                "lexical_top": [(str(self.olj_docs[i].url), s) for i, s in lexical[:10]],
                "semantic_top": [(str(self.olj_docs[i].url), s) for i, s in semantic[:10]] if semantic else [],
                "rrf_top": [(str(self.olj_docs[i].url), s) for i, s in fused[:10]] if isinstance(fused, list) else [],
                "reranked": [(it.url, it.score) for it in reranked],
                "selected": [c.url for c in cards if c.url],
                "excluded_from_session": list(exclude_set),
                "rerank_top_k": rerank_top_k,
                "rerank_shortcircuit": rerank_shortcircuit,
            }
        elif rerank_shortcircuit:
            dbg = {"rerank_shortcircuit": True}

        if cards:
            return cards[: analysis.recipe_count], False, dbg

        base2_cards = self._search_base2(analysis)
        if base2_cards:
            return base2_cards[: analysis.recipe_count], True, dbg

        return [], False, dbg

    def _search_menu(
        self, analysis: QueryAnalysis, raw_query: str, *, debug: bool
    ) -> Tuple[List[RecipeCard], bool, Optional[dict]]:
        """Search for menu composition (entrée + plat + dessert)."""
        cards, dbg = self._search_olj(
            QueryAnalysis(**{**analysis.model_dump(), "recipe_count": max(10, analysis.recipe_count)}),
            raw_query,
            debug=debug,
        )
        if not cards:
            return [], False, dbg

        def cat_of(c: RecipeCard) -> str:
            return (c.category or "").lower()

        picked: List[RecipeCard] = []
        used = set()

        for c in cards:
            if c.url in used:
                continue
            if cat_of(c) in ("mezze_froid", "mezze_chaud", "entree"):
                picked.append(c)
                used.add(c.url)
                break

        for c in cards:
            if c.url in used:
                continue
            if cat_of(c) == "plat_principal":
                picked.append(c)
                used.add(c.url)
                break

        for c in cards:
            if c.url in used:
                continue
            if cat_of(c) == "dessert":
                picked.append(c)
                used.add(c.url)
                break

        for c in cards:
            if len(picked) >= 3:
                break
            if c.url and c.url not in used:
                picked.append(c)
                used.add(c.url)

        return picked[:3], False, dbg

    # Stoplist pour filtrer le bruit dans main_ingredients (olj_canonical)
    _MAIN_INGREDIENTS_STOPLIST = frozenset({
        "verts", "fermes", "doux", "2", "poudre", "fleur", "jus",
        "brunoise", "dœuf",
    })

    @staticmethod
    def _filter_main_ingredients(mains: List[str]) -> List[str]:
        """Filtre le bruit dans main_ingredients (artefacts d'extraction)."""
        out: List[str] = []
        for ing in (mains or []):
            if not ing or len(ing) < 3:
                continue
            if ing.isdigit():
                continue
            low = ing.lower().strip()
            if low in HybridRetriever._MAIN_INGREDIENTS_STOPLIST:
                continue
            # Corriger "dolive" -> "olive" (fragment de "huile d'olive")
            if low == "dolive":
                out.append("olive")
            else:
                out.append(ing)
        return out

    @staticmethod
    def _extract_ingredients_from_dish_name(dish_name: Optional[str]) -> List[str]:
        """Heuristique : extrait des ingrédients du nom du plat."""
        if not dish_name or not dish_name.strip():
            return []
        name = dish_name.strip().lower()
        ingredients: List[str] = []
        # "tarte au citron", "tarte aux pommes", "crème à la vanille"
        for m in re.finditer(r"(?:au|à la|aux|avec)\s+(\w+)", name):
            ingredients.append(m.group(1))
        # Si plusieurs mots sans préposition : "poulet curry" -> ["poulet", "curry"]
        if not ingredients and " " in name:
            stop = {"de", "du", "la", "le", "les", "un", "une", "et", "ou"}
            for w in name.split():
                if len(w) >= 3 and w not in stop:
                    ingredients.append(w)
        return ingredients

    def _doc_to_recipe_card(self, doc: CanonicalRecipeDoc) -> RecipeCard:
        """Convertit un CanonicalRecipeDoc en RecipeCard (sans cited_passage)."""
        image_url = getattr(doc, "image_url", None) or getattr(doc, "_image_url", None)
        return RecipeCard(
            source="olj",
            title=doc.title,
            url=str(doc.url),
            chef=doc.chef_name,
            category=doc.category_canonical,
            image_url=image_url,
            cited_passage=None,
        )

    def get_olj_recommendation_by_ingredient(
        self,
        analysis: QueryAnalysis,
        raw_query: str,
        *,
        exclude_urls: Optional[set[str]] = None,
    ) -> Optional[Tuple[RecipeCard, Optional[str]]]:
        """
        Recommandation OLJ avec au moins un ingrédient en commun.
        Retourne (RecipeCard, matched_ingredient) ou None.
        matched_ingredient: str pour personnaliser la narrative, None si fallback.
        """
        exclude = set(exclude_urls or [])

        # 1) Collecte des ingrédients
        query_ingredients: List[str] = []
        query_ingredients.extend(analysis.ingredients or [])
        query_ingredients.extend(getattr(analysis, "inferred_ingredients", None) or [])
        from_dish = self._extract_ingredients_from_dish_name(analysis.dish_name)
        for ing in from_dish:
            if ing and ing not in query_ingredients:
                query_ingredients.append(ing)

        # 2) Match par ingrédient sur OLJ
        for doc in self.olj_docs:
            if str(doc.url) in exclude:
                continue
            if not doc.is_recipe:
                continue
            filtered = self._filter_main_ingredients(doc.main_ingredients)
            if not query_ingredients:
                break
            match_count, _ = ingredient_normalizer.match_ingredients(
                query_ingredients, filtered
            )
            if match_count >= 1:
                # Trouver le premier ingrédient qui matche
                matched_ingredient: Optional[str] = None
                for q_ing in query_ingredients:
                    mc, _ = ingredient_normalizer.match_ingredients([q_ing], filtered)
                    if mc >= 1:
                        matched_ingredient = q_ing
                        break
                card = self._doc_to_recipe_card(doc)
                return (card, matched_ingredient)

        # 3) Fallback : catégorie
        target_cat = analysis.category
        for doc in self.olj_docs:
            if str(doc.url) in exclude:
                continue
            if not doc.is_recipe:
                continue
            if target_cat and doc.category_canonical == target_cat:
                return (self._doc_to_recipe_card(doc), None)

        # 4) Fallback : première recette OLJ
        for doc in self.olj_docs:
            if str(doc.url) in exclude:
                continue
            if doc.is_recipe:
                return (self._doc_to_recipe_card(doc), None)

        # 5) Fallback Base2

        base2_analysis = QueryAnalysis(
            **{**analysis.model_dump(), "recipe_count": 1, "ingredients": query_ingredients}
        )
        base2_cards = self._search_base2(base2_analysis)
        if base2_cards:
            return (base2_cards[0], None)

        return None

    def _search_base2(self, analysis: QueryAnalysis) -> List[RecipeCard]:
        """Fallback search in Base2 dataset."""
        from ..data.normalizers import normalize_text

        parts = [analysis.dish_name or "", *(analysis.dish_name_variants or []), *(analysis.ingredients or [])]
        norm_terms: list[str] = []
        for p in parts:
            if not p:
                continue
            np = normalize_text(p)
            if np and len(np) >= 3:
                norm_terms.append(np)
            for w in str(p).split():
                nw = normalize_text(w)
                if nw and len(nw) >= 3:
                    norm_terms.append(nw)
        norm_terms = list(dict.fromkeys(norm_terms))
        results: List[Tuple[Dict[str, Any], float, str]] = []

        for cat_name, recipes in self.base2_recipes.items():
            for r in recipes:
                name = (r.get("nom") or "").lower()
                text = name
                for ing in r.get("ingredients", []) or []:
                    if isinstance(ing, dict):
                        text += " " + (ing.get("nom") or "")
                    else:
                        text += " " + str(ing)
                text_norm = normalize_text(text)
                score = 0.0
                for nt in norm_terms:
                    if nt in text_norm or text_norm in nt:
                        score += 2.0
                    else:
                        for w in text_norm.split():
                            if len(nt) >= 4 and len(w) >= 4:
                                if difflib.SequenceMatcher(None, nt, w).ratio() >= 0.72:
                                    score += 1.8
                                    break
                # Compat : tokens simples (ex. sous-chaînes courtes)
                if score == 0:
                    qlow = " ".join(parts).lower()
                    for t in qlow.split():
                        if t and len(t) >= 3 and t in text.lower():
                            score += 1.0
                if score > 0:
                    results.append((r, score, cat_name))

        results.sort(key=lambda x: -x[1])
        cards: List[RecipeCard] = []
        for r, _, cat_name in results[: analysis.recipe_count]:
            ingredients_list: List[str] = []
            for ing in r.get("ingredients", []) or []:
                if isinstance(ing, dict):
                    parts = []
                    if ing.get("quantite"):
                        parts.append(str(ing["quantite"]))
                    if ing.get("unite"):
                        parts.append(ing["unite"])
                    if ing.get("nom"):
                        parts.append(ing["nom"])
                    ingredients_list.append(" ".join(parts).strip())
                else:
                    ingredients_list.append(str(ing))

            cards.append(
                RecipeCard(
                    source="base2",
                    title=r.get("nom", "Sans titre"),
                    category=cat_name,
                    ingredients=ingredients_list or None,
                    steps=r.get("etapes", []) or None,
                    prep_time=r.get("temps_preparation"),
                    cook_time=r.get("temps_cuisson"),
                    servings=r.get("nombre_de_personnes"),
                )
            )
        return cards

    def get_grounding_for_term(self, term: str, max_chars: int = 400) -> Optional[Tuple[str, str, str]]:
        """
        Retrieve targeted OLJ passage for a term (clarification/recipe_info).
        Returns (excerpt, title, url) or None. Grounds answers in articles only.
        Prefers sentences containing the term for better relevance.
        """
        if not term or not term.strip():
            return None
        q = term.strip().lower()
        lexical = self._lexical_retrieve(term.strip(), top_k=5)
        for idx, score in lexical:
            if idx >= len(self.olj_docs):
                continue
            d = self.olj_docs[idx]
            if not d.search_text or not d.is_lebanese:
                continue
            text = (d.search_text or "").strip()
            if not text:
                continue
            excerpt = _extract_targeted_passage(text, q, max_chars)
            if excerpt:
                return (excerpt, d.title, str(d.url))
        return None

    def resolve_exact_dish(
        self,
        dish_name: str,
        dish_name_variants: Optional[List[str]] = None,
        exclude_urls: Optional[set] = None,
    ) -> Optional[RecipeCard]:
        """
        Deterministic exact match using canonical index (title_normalized, aliases_normalized, dish_name_normalized).
        Used before full retrieval to guarantee OLJ recipe when dish is in base.
        """
        if not dish_name or not dish_name.strip():
            return None
        exclude = set(exclude_urls or [])
        variants = {dish_name.strip()} | {v.strip() for v in (dish_name_variants or []) if v and v.strip()}
        normalized_variants = {normalize_text(v) for v in variants if v and len(normalize_text(v)) >= 3}

        for d in self.olj_docs:
            if str(d.url) in exclude or not d.is_recipe or not d.is_lebanese:
                continue
            title_norm = d.title_normalized or normalize_text(d.title)
            aliases_norm = d.aliases_normalized or [normalize_text(a) for a in (d.aliases or []) if a]
            dish_norm = d.dish_name_normalized or title_norm
            for v in normalized_variants:
                if v in title_norm or title_norm in v:
                    logger.info("ExactDishResolver: canonical match '%s' -> %s", dish_name[:30], d.title)
                    return RecipeCard(
                        source="olj",
                        title=d.title,
                        url=str(d.url),
                        chef=d.chef_name,
                        category=d.category_canonical,
                        image_url=d.image_url,
                        cited_passage=None,
                    )
                if v in dish_norm or dish_norm in v:
                    logger.info("ExactDishResolver: dish_name match '%s' -> %s", dish_name[:30], d.title)
                    return RecipeCard(
                        source="olj",
                        title=d.title,
                        url=str(d.url),
                        chef=d.chef_name,
                        category=d.category_canonical,
                        image_url=d.image_url,
                        cited_passage=None,
                    )
                for a in aliases_norm:
                    if v in a or a in v:
                        logger.info("ExactDishResolver: alias match '%s' -> %s", dish_name[:30], d.title)
                        return RecipeCard(
                            source="olj",
                            title=d.title,
                            url=str(d.url),
                            chef=d.chef_name,
                            category=d.category_canonical,
                            image_url=d.image_url,
                            cited_passage=None,
                        )
        return None

    def get_mood_fallback_recipe(
        self,
        analysis: "QueryAnalysis",
        *,
        exclude_urls: Optional[set] = None,
    ) -> Optional[RecipeCard]:
        """
        Fallback for mood/vague queries: pick the best OLJ recipe matching the mood.

        Mood → preferred category mapping, then pick any recipe not already shown.
        """
        import random

        _MOOD_TO_CATEGORIES: Dict[str, List[str]] = {
            "leger":        ["salade", "entree", "mezze_froid"],
            "rapide":       ["entree", "mezze_froid", "mezze_chaud", "salade"],
            "frais":        ["salade", "entree", "mezze_froid"],
            "hiver":        ["soupe", "plat_principal", "mezze_chaud"],
            "reconfortant": ["soupe", "plat_principal", "mezze_chaud"],
            "ete":          ["salade", "entree", "mezze_froid"],
            "festif":       ["plat_principal", "mezze_chaud", "mezze_froid"],
            "copieux":      ["plat_principal", "mezze_chaud"],
            "convivial":    ["mezze_froid", "mezze_chaud", "plat_principal"],
            "facile":       ["entree", "mezze_froid", "salade"],
        }

        exclude = set(exclude_urls or [])
        mood_tags = getattr(analysis, "mood_tags", []) or []

        # Collect preferred categories in order of priority
        preferred: List[str] = []
        for tag in mood_tags:
            for cat in _MOOD_TO_CATEGORIES.get(tag, []):
                if cat not in preferred:
                    preferred.append(cat)

        recipe_docs = [d for d in self.olj_docs if d.is_recipe and str(d.url) not in exclude]
        if not recipe_docs:
            return None

        # Try preferred categories first
        for cat in preferred:
            matching = [d for d in recipe_docs if (d.category_canonical or "") == cat]
            if matching:
                chosen = random.choice(matching)
                logger.info("Mood fallback: tag=%s cat=%s -> %s", mood_tags, cat, chosen.title)
                return self._doc_to_recipe_card(chosen)

        # Any recipe as last resort
        chosen = random.choice(recipe_docs)
        logger.info("Mood fallback (no cat match): tag=%s -> %s", mood_tags, chosen.title)
        return self._doc_to_recipe_card(chosen)

    def get_olj_recommendation(
        self, analysis: QueryAnalysis, exclude_titles: Optional[List[str]] = None
    ) -> Optional[OLJRecommendation]:
        """Get a recommendation from OLJ docs."""
        exclude = {t.lower() for t in (exclude_titles or [])}
        target_cat = analysis.category

        for d in self.olj_docs:
            if d.title.lower() in exclude:
                continue
            if target_cat and d.category_canonical == target_cat:
                return OLJRecommendation(
                    title=d.title,
                    url=str(d.url),
                    reason="Une recette à découvrir sur L'Orient-Le Jour",
                )

        if self.olj_docs:
            d = self.olj_docs[0]
            return OLJRecommendation(
                title=d.title,
                url=str(d.url),
                reason="Une recette à découvrir sur L'Orient-Le Jour",
            )
        return None

    def _infer_category(self, analysis: QueryAnalysis, raw_query: str) -> CanonicalCategory:
        """Infer recipe category from analysis and raw query for fallback when no ingredient match."""
        q = (raw_query or "").lower() + " " + (analysis.dish_name or "").lower()
        dessert_words = {"dessert", "gateau", "gâteau", "crepe", "crêpe", "tarte", "sucré", "doux", "chocolat"}
        plat_words = {
            "plat",
            "viande",
            "poisson",
            "poulet",
            "boeuf",
            "agneau",
            "couscous",
            "paella",
            "risotto",
            "burger",
            "pizza",
            "lasagne",
            "pates",
            "pâte",
            "pâtes",
            "pasta",
            "carbonara",
            "spaghetti",
            "curry",
            "ramen",
        }
        mezze_words = {"mezze", "entree", "entrée", "houmous", "taboulé", "falafel"}
        if any(w in q for w in dessert_words):
            return "dessert"
        if any(w in q for w in plat_words):
            return "plat_principal"
        if any(w in q for w in mezze_words):
            return "mezze_froid"
        if analysis.category and analysis.category in ("dessert", "mezze_froid", "mezze_chaud", "plat_principal", "entree", "soupe", "salade"):
            return analysis.category  # type: ignore[return-value]
        return "plat_principal"

    def get_alternative_by_shared_ingredient(
        self,
        raw_query: str,
        analysis: QueryAnalysis,
        exclude_urls: Optional[set] = None,
        inferred_ingredients: Optional[List[str]] = None,
    ) -> Optional[AlternativeMatch]:
        """
        Find a Lebanese recipe that shares at least one main ingredient
        with the user's request. Returns AlternativeMatch only when proof exists.
        inferred_ingredients: from LLM (inferred_main_ingredients) — couvre tout plat
        non listé sans heuristique statique.
        """
        exclude = set(exclude_urls or [])

        # Normalisations issues du LLM (pour assouplir le filtre texte 2-termes)
        llm_inferred_norms: set[str] = set()
        if inferred_ingredients:
            for ing in inferred_ingredients:
                if not ing:
                    continue
                norm_list = ingredient_normalizer.normalize_ingredient_list([ing])
                for n in norm_list:
                    llm_inferred_norms.add(normalize_text(n.normalized))
                    for e in ingredient_normalizer.get_equivalents(n.normalized):
                        if e:
                            llm_inferred_norms.add(normalize_text(e))

        # Build searchable terms from query + analysis (including Lebanese alternatives)
        query_terms: set[str] = set()
        raw_ingredients: list[str] = list(inferred_ingredients or []) + list(
            analysis.ingredients or []
        )

        # Extract from dish_name and raw_query (words 3+ chars)
        for text in [analysis.dish_name, raw_query]:
            if not text:
                continue
            words = text.lower().replace("-", " ").replace("'", " ").split()
            for w in words:
                if len(w) >= 3 and w.isalpha() and normalize_text(w) not in GENERIC_QUERY_TOKENS:
                    raw_ingredients.append(w)

        # Use normalizer to get equivalents AND Lebanese alternatives (e.g. pâtes -> moghrabieh)
        for ing in raw_ingredients:
            if not ing:
                continue
            norm_list = ingredient_normalizer.normalize_ingredient_list([ing])
            for n in norm_list:
                if n.normalized.lower() in GENERIC_QUERY_TOKENS:
                    continue
                query_terms.add(n.normalized.lower())
                equivs = ingredient_normalizer.get_equivalents(n.normalized)
                query_terms.update(
                    e.lower()
                    for e in equivs
                    if e and e.lower() not in GENERIC_QUERY_TOKENS
                )

        # Score each OLJ doc by shared ingredient overlap (PROVEN match only)
        # First pass: canonical main_ingredients_normalized (strict)
        # Second pass: fallback on normalized text fields (plus strict : filtré ci-dessous)
        query_norm = {normalize_text(t) for t in query_terms}
        candidates: List[Tuple[CanonicalRecipeDoc, int, List[str], bool]] = []
        for d in self.olj_docs:
            if str(d.url) in exclude or not d.is_recipe or not d.is_lebanese:
                continue
            if not query_norm:
                continue

            canonical_ings = {
                normalize_text(item)
                for item in (d.main_ingredients_normalized or [])
                if item
            }
            shared_canonical = [t for t in query_norm if t and t in canonical_ings]
            if shared_canonical:
                candidates.append((d, len(shared_canonical) + 10, shared_canonical, True))
                continue

            doc_text = normalize_text((d.search_text or "") + " " + " ".join(d.main_ingredients or []))
            shared: List[str] = [t for t in query_norm if t and t in doc_text]
            if shared:
                candidates.append((d, len(shared), shared, False))

        inferred_cat = self._infer_category(analysis, raw_query)
        filtered: List[Tuple[CanonicalRecipeDoc, int, List[str], bool]] = []
        for d, score, shared, from_canonical in candidates:
            if not ingredient_overlap_is_meaningful(shared):
                continue
            if not doc_category_matches_inferred(d.category_canonical, inferred_cat):
                continue
            # Match texte plein : au moins 2 termes sauf si l'un vient des ingrédients inférés LLM
            if not from_canonical:
                shared_norms_set = {normalize_text(s) for s in shared}
                has_llm_support = bool(shared_norms_set & llm_inferred_norms)
                if len(shared_norms_set) < 2 and not has_llm_support:
                    continue
            filtered.append((d, score, shared, from_canonical))

        if not filtered:
            return None

        def _sort_key(item: Tuple[CanonicalRecipeDoc, int, List[str], bool]) -> Tuple[int, int, int]:
            d, score, shared, from_canonical = item
            cat_bonus = 0
            if d.category_canonical == inferred_cat:
                cat_bonus = 1
            return (1 if from_canonical else 0, cat_bonus, score)

        filtered.sort(key=_sort_key, reverse=True)
        best_doc, score, shared, _from_canonical = filtered[0]
        meaningful_inputs = [
            normalize_text(item)
            for item in raw_ingredients
            if item and normalize_text(item) not in GENERIC_QUERY_TOKENS
        ]
        query_ing = meaningful_inputs[0] if meaningful_inputs else raw_query[:30]
        proof = SharedIngredientProof(
            query_ingredient=query_ing,
            normalized_ingredient=list(query_norm)[0] if query_norm else "",
            shared_ingredients=shared[:10],
            recipe_title=best_doc.title,
            recipe_url=str(best_doc.url),
            proof_score=score,
        )
        match_reason = self._infer_match_reason(shared)
        logger.info(
            "get_alternative: SharedIngredientProof for '%s' -> %s (shared=%s)",
            raw_query[:50], best_doc.title, shared[:5],
        )
        passage_term = (
            shared[0]
            if shared
            else (sorted(query_norm)[0] if query_norm else "")
        )
        cited = (
            _extract_targeted_passage(
                best_doc.search_text or "", passage_term, max_chars=200
            )
            if passage_term
            else None
        )
        card = RecipeCard(
            source="olj",
            title=best_doc.title,
            url=str(best_doc.url),
            chef=best_doc.chef_name,
            category=best_doc.category_canonical,
            image_url=best_doc.image_url,
            cited_passage=cited,
        )
        return AlternativeMatch(recipe_card=card, proof=proof, match_reason=match_reason)

    def get_category_fallback_match(
        self,
        analysis: QueryAnalysis,
        raw_query: str,
        exclude_urls: Optional[set] = None,
    ) -> Optional[AlternativeMatch]:
        """
        Dernier recours : une recette OLJ libanaise dans la catégorie inférée,
        en privilégiant le chevauchement avec inferred_main_ingredients.
        """
        exclude = set(exclude_urls or [])
        inferred_cat = self._infer_category(analysis, raw_query)
        inferred = list(analysis.inferred_main_ingredients or [])

        def score_doc(d: CanonicalRecipeDoc) -> Tuple[int, int]:
            overlap = 0
            doc_mains = {normalize_text(x) for x in (d.main_ingredients_normalized or []) if x}
            doc_text = normalize_text(
                (d.search_text or "") + " " + " ".join(d.main_ingredients or [])
            )
            for ing in inferred:
                ni = normalize_text(ing)
                if ni and (ni in doc_mains or ni in doc_text):
                    overlap += 2
            cat_bonus = 3 if d.category_canonical == inferred_cat else 0
            return (cat_bonus + overlap, overlap)

        candidates = [
            d
            for d in self.olj_docs
            if str(d.url) not in exclude and d.is_recipe and d.is_lebanese
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda d: score_doc(d), reverse=True)
        best = candidates[0]

        doc_mains_norm = list(best.main_ingredients_normalized or [])
        shared: List[str] = []
        inferred_norms = {normalize_text(x) for x in inferred if x}
        for im in inferred_norms:
            if im and im in doc_mains_norm:
                shared.append(im)
        if not shared and doc_mains_norm:
            shared = [doc_mains_norm[0]]
        elif not shared and best.main_ingredients:
            shared = [normalize_text(best.main_ingredients[0])]

        proof = SharedIngredientProof(
            query_ingredient=(analysis.dish_name or raw_query[:50]).strip(),
            normalized_ingredient=next(iter(inferred_norms), "") if inferred_norms else "",
            shared_ingredients=shared[:10],
            recipe_title=best.title,
            recipe_url=str(best.url),
            proof_score=score_doc(best)[0],
        )
        passage_term = shared[0] if shared else (
            normalize_text(best.main_ingredients[0])
            if best.main_ingredients
            else ""
        )
        cited = (
            _extract_targeted_passage(
                best.search_text or "", passage_term, max_chars=200
            )
            if passage_term
            else None
        )
        card = RecipeCard(
            source="olj",
            title=best.title,
            url=str(best.url),
            chef=best.chef_name,
            category=best.category_canonical,
            image_url=best.image_url,
            cited_passage=cited,
        )
        return AlternativeMatch(recipe_card=card, proof=proof, match_reason="category_fallback")

    def _infer_match_reason(self, shared_terms: List[str]) -> str:
        shared_set = {normalize_text(term) for term in (shared_terms or []) if term}
        for reason, keys in MATCH_REASON_KEYWORDS.items():
            if shared_set & keys:
                return reason
        return "shared_ingredient_match"

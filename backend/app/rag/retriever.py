"""
Hybrid Retriever (Durable)
==========================

Implements a durable retrieval pipeline:
  1) Retrieve (high recall): lexical TF-IDF + optional embeddings
  2) Fuse rankings with RRF (no magic weights)
  3) Rerank (high precision): LLM reranker on topK
  4) Select: dedup + diversity rules (menu) + hard filters

This avoids fragile ad-hoc boosts at runtime and makes decisions explainable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
import hashlib
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core.config import get_settings
from ..llm.reranker import LLMReranker, RerankCandidate, RerankItem
from ..schemas.canonical import CanonicalRecipeDoc
from ..schemas.query_analysis import QueryAnalysis
from ..schemas.responses import OLJRecommendation, RecipeCard

logger = logging.getLogger(__name__)


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

    Returns:
      (recipes, is_base2_fallback, debug_info)
    """

    def __init__(
        self,
        *,
        olj_canonical_path: Optional[str] = None,
        base2_path: Optional[str] = None,
        enable_llm_rerank: bool = True,
    ):
        settings = get_settings()

        base_dir = Path(__file__).parent.parent.parent.parent  # v2/
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

        self._load_data()
        self._build_lexical_index()

    def _load_data(self) -> None:
        # Load canonical OLJ docs
        try:
            raw = json.loads(Path(self.olj_canonical_path).read_text(encoding="utf-8"))
            self.olj_docs = [CanonicalRecipeDoc(**d) for d in raw]
            logger.info("Loaded %s canonical OLJ docs from %s", len(self.olj_docs), self.olj_canonical_path)
        except FileNotFoundError:
            # Durable fallback: build a minimal canonical dataset in-memory from the raw OLJ JSON.
            # This keeps system operational even before the offline backfill step is run.
            logger.warning(
                "Canonical OLJ dataset not found: %s. Falling back to raw dataset: %s",
                self.olj_canonical_path,
                self.olj_raw_path,
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
        """
        Build a minimal CanonicalRecipeDoc list from `data_base_OLJ_enriched.json`.

        This is a *runtime safety net* only:
        - We do not attempt full backfill quality here
        - We do ensure critical fields are non-empty to keep retrieval stable
        """
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
                # Last resort: derive from URL slug
                slug = str(url).rstrip("/").split("/")[-1].replace("-", " ").strip()
                title = slug[:1].upper() + slug[1:] if slug else "Sans titre"

            enrichment = item.get("enrichment") or {}

            is_recipe = bool(enrichment.get("is_recipe", True))
            is_lebanese = bool(enrichment.get("is_lebanese", True))
            cuisine_type = enrichment.get("cuisine_type") or None

            # Canonical category/difficulty: prefer enrichment, fallback to raw fields
            cat_raw = (enrichment.get("category") or item.get("category") or "").strip().lower()
            diff_raw = (enrichment.get("difficulty") or item.get("difficulty") or "").strip().lower()

            category_map = {
                "mezze": "mezze_froid",
                "mezze froid": "mezze_froid",
                "mezze_chaud": "mezze_chaud",
                "mezze chaud": "mezze_chaud",
                "plat principal": "plat_principal",
                "plat_principal": "plat_principal",
                "dessert": "dessert",
                "entrée": "entree",
                "entree": "entree",
                "salade": "salade",
                "soupe": "soupe",
                "sauce": "sauces",
                "sauces": "sauces",
                "cocktail": "boisson",
                "apéro": "boisson",
                "apero": "boisson",
                "boisson": "boisson",
            }
            difficulty_map = {
                "facile": "facile",
                "moyenne": "moyenne",
                "difficile": "difficile",
            }

            category_canonical = category_map.get(cat_raw, "autre")
            difficulty_canonical = difficulty_map.get(diff_raw, "non_specifie")

            chef_name = (item.get("chef_name") or "").strip() or None
            tags = item.get("tags") or []
            main_ingredients = enrichment.get("main_ingredients") or []
            aliases = enrichment.get("aliases") or []

            # Single search_text (critical): use the same recipe fields everywhere
            desc = (item.get("recipe_description") or "").strip()
            ingredients = " ".join((item.get("ingredients") or [])[:80])
            instructions = " ".join((item.get("instructions") or [])[:120])
            tags_str = " ".join(tags) if isinstance(tags, list) else str(tags)
            alias_str = " ".join(aliases) if isinstance(aliases, list) else str(aliases)
            mains_str = " ".join(main_ingredients) if isinstance(main_ingredients, list) else str(main_ingredients)

            search_text = " ".join(
                x
                for x in [title, chef_name or "", desc, ingredients, instructions, tags_str, alias_str, mains_str]
                if x
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
                        category_canonical=category_canonical,  # type: ignore[arg-type]
                        difficulty_canonical=difficulty_canonical,  # type: ignore[arg-type]
                        tags=[t for t in tags if t] if isinstance(tags, list) else [],
                        main_ingredients=[i for i in main_ingredients if i]
                        if isinstance(main_ingredients, list)
                        else [],
                        aliases=[a for a in aliases if a] if isinstance(aliases, list) else [],
                        search_text=search_text,
                        raw_category=cat_raw or None,
                        raw_difficulty=diff_raw or None,
                        raw_enrichment_present=bool(enrichment),
                    )
                )
            except Exception:
                # Skip malformed items rather than failing startup
                continue

        # Dedup by URL (keep first)
        by_url: Dict[str, CanonicalRecipeDoc] = {}
        for d in docs:
            by_url[str(d.url)] = by_url.get(str(d.url), d)
        return list(by_url.values())

    def _build_lexical_index(self) -> None:
        texts = [d.search_text for d in self.olj_docs]
        if not texts:
            self._tfidf = None
            self._tfidf_matrix = None
            return

        # Robust for typos/transliteration: use word + char n-grams
        self._tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            analyzer="word",
            min_df=1,
        )
        word_matrix = self._tfidf.fit_transform(texts)

        # Additional char-level signal
        char_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
        )
        char_matrix = char_vectorizer.fit_transform(texts)

        # Store both matrices and vectorizers
        self._tfidf_matrix = (word_matrix, char_matrix, char_vectorizer)

    def search(
        self, analysis: QueryAnalysis, *, raw_query: str, debug: bool = False
    ) -> Tuple[List[RecipeCard], bool, Optional[dict]]:
        settings = get_settings()

        if analysis.intent == "menu_composition":
            recipes, is_base2, dbg = self._search_menu(analysis, raw_query, debug=debug)
            return recipes, is_base2, dbg

        # 1) OLJ retrieve
        olj_cards, dbg = self._search_olj(analysis, raw_query, debug=debug)

        if olj_cards:
            return olj_cards[: analysis.recipe_count], False, dbg

        # 2) Base2 fallback
        base2_cards = self._search_base2(analysis)
        if base2_cards:
            return base2_cards[: analysis.recipe_count], True, dbg

        return [], False, dbg

    def _lexical_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if not self._tfidf or not self._tfidf_matrix:
            return []
        word_matrix, char_matrix, char_vectorizer = self._tfidf_matrix
        q_word = self._tfidf.transform([query])
        q_char = char_vectorizer.transform([query])

        # Combine cosine similarities (simple average)
        s_word = cosine_similarity(q_word, word_matrix).ravel()
        s_char = cosine_similarity(q_char, char_matrix).ravel()
        scores = (s_word + s_char) / 2.0

        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    def _semantic_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        settings = get_settings()
        if not settings.enable_embeddings:
            return []

        # Deterministic, dependency-free embeddings (runtime stable).
        def embed(text: str, dim: int = 128) -> np.ndarray:
            seed = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
            rng = random.Random(int(seed, 16))
            raw = np.array([rng.uniform(-1.0, 1.0) for _ in range(dim)], dtype=np.float32)
            norm = float(np.linalg.norm(raw)) or 1.0
            return raw / norm

        docs = self.olj_docs
        if not docs:
            return []

        q = embed(query)
        D = np.stack([embed(d.search_text) for d in docs], axis=0)
        sims = (D @ q).ravel()
        idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idx if sims[i] > 0]

    @staticmethod
    def _rrf_fuse(
        ranked_lists: List[List[Tuple[int, float]]], *, k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF).
        Uses ranks only -> reduces brittle weighting.
        """
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
    ) -> List[RerankItem]:
        candidates: List[RerankCandidate] = []
        # Important: when applying a category allowlist, scan deeper than top_k to
        # ensure we collect enough candidates in the target category.
        for idx, _ in fused if category_allowlist else fused[:top_k]:
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
            if len(candidates) >= top_k:
                break
        return await self.reranker.rerank(raw_query, candidates, max_items=min(10, len(candidates)))

    def _select_cards(
        self,
        analysis: QueryAnalysis,
        reranked: List[RerankItem],
        *,
        max_results: int,
        category_allowlist: Optional[set[str]] = None,
        exclude_urls: Optional[set[str]] = None,
    ) -> List[RecipeCard]:
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
            if category_allowlist and doc.category_canonical not in category_allowlist:
                continue

            cards.append(
                RecipeCard(
                    source="olj",
                    title=doc.title,
                    url=str(doc.url),
                    chef=doc.chef_name,
                    category=doc.category_canonical,
                )
            )
            if len(cards) >= max_results:
                break
        return cards

    @staticmethod
    def _allowlist_for_intent(analysis: QueryAnalysis, raw_query: str) -> Optional[set[str]]:
        """
        Apply durable category constraints for intents that require them.
        """
        q = (raw_query or "").lower()
        if analysis.intent == "recipe_by_category" and analysis.category:
            return {analysis.category}

        if analysis.intent == "multi_recipe":
            if analysis.category:
                if analysis.category in {"mezze_froid", "mezze_chaud"}:
                    return {"mezze_froid", "mezze_chaud", "entree"}
                return {analysis.category}
            if "mezze" in q or "meze" in q:
                return {"mezze_froid", "mezze_chaud", "entree"}
            if "dessert" in q or "sucre" in q:
                return {"dessert"}
        return None

    def _search_olj(
        self, analysis: QueryAnalysis, raw_query: str, *, debug: bool
    ) -> Tuple[List[RecipeCard], Optional[dict]]:
        settings = get_settings()

        top_k = settings.retrieve_top_k

        # Build a retrieval query from analysis
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

        lexical = self._lexical_retrieve(retrieval_query, top_k=top_k)
        semantic = self._semantic_retrieve(retrieval_query, top_k=top_k)
        fused = self._rrf_fuse([lexical, semantic])

        reranked: List[RerankItem] = []
        if self.enable_llm_rerank and fused:
            reranked = []

        if not reranked:
            reranked = [
                RerankItem(url=str(self.olj_docs[idx].url), score=float(score))
                for idx, score in fused[: settings.rerank_top_k]
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
                "semantic_top": [(str(self.olj_docs[i].url), s) for i, s in semantic[:10]],
                "rrf_top": [(str(self.olj_docs[i].url), s) for i, s in fused[:10]],
                "reranked": [(it.url, it.score) for it in reranked],
                "selected": [c.url for c in cards if c.url],
            }
        return cards, dbg

    async def search_with_rerank(
        self, analysis: QueryAnalysis, *, raw_query: str, debug: bool = False
    ) -> Tuple[List[RecipeCard], bool, Optional[dict]]:
        """
        Async variant that actually runs LLM rerank before selecting.
        Used by main bot.
        """
        settings = get_settings()

        # Menu composition: do 3 category-constrained searches
        if analysis.intent == "menu_composition":
            picked: List[RecipeCard] = []
            used: set[str] = set()

            async def pick_one(cat_allow: set[str], query_hint: str) -> Optional[RecipeCard]:
                local_query = f"{raw_query} {query_hint}".strip()

                parts: List[str] = [local_query]
                if analysis.dish_name:
                    parts.append(analysis.dish_name)
                parts.extend(analysis.dish_name_variants or [])
                parts.extend(analysis.ingredients or [])
                parts.extend(analysis.mood_tags or [])
                if analysis.chef_name:
                    parts.append(analysis.chef_name)
                retrieval_query = " ".join(p for p in parts if p)

                lexical = self._lexical_retrieve(retrieval_query, top_k=settings.retrieve_top_k)
                semantic = self._semantic_retrieve(retrieval_query, top_k=settings.retrieve_top_k)
                fused = self._rrf_fuse([lexical, semantic])

                reranked = await self._rerank(
                    raw_query,
                    fused,
                    top_k=settings.retrieve_top_k,
                    category_allowlist=cat_allow,
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

            # If we couldn't fill all 3 from OLJ, fallback to Base2
            if len(picked) < 3:
                base2_cards = self._search_base2(QueryAnalysis(**{**analysis.model_dump(), "recipe_count": 3}))
                for c in base2_cards:
                    if len(picked) >= 3:
                        break
                    picked.append(c)
                return picked[:3], bool(base2_cards), {"selected": [c.url for c in picked if c.url]} if debug else None

            return picked[:3], False, {"selected": [c.url for c in picked if c.url]} if debug else None

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

        lexical = self._lexical_retrieve(retrieval_query, top_k=settings.retrieve_top_k)
        semantic = self._semantic_retrieve(retrieval_query, top_k=settings.retrieve_top_k)
        fused = self._rrf_fuse([lexical, semantic])

        allowlist = self._allowlist_for_intent(analysis, raw_query)
        must_any = None
        if analysis.intent == "recipe_by_ingredient" and analysis.ingredients:
            must_any = {i.lower() for i in analysis.ingredients if i}
        reranked = await self._rerank(
            raw_query,
            fused,
            top_k=settings.retrieve_top_k,
            category_allowlist=allowlist,
            must_contain_any=must_any,
        )
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
                "semantic_top": [(str(self.olj_docs[i].url), s) for i, s in semantic[:10]],
                "rrf_top": [(str(self.olj_docs[i].url), s) for i, s in fused[:10]],
                "reranked": [(it.url, it.score) for it in reranked],
                "selected": [c.url for c in cards if c.url],
            }

        if cards:
            return cards[: analysis.recipe_count], False, dbg

        base2_cards = self._search_base2(analysis)
        if base2_cards:
            return base2_cards[: analysis.recipe_count], True, dbg

        return [], False, dbg

    def _search_menu(
        self, analysis: QueryAnalysis, raw_query: str, *, debug: bool
    ) -> Tuple[List[RecipeCard], bool, Optional[dict]]:
        # Reuse OLJ search and then pick diverse categories
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

    def _search_base2(self, analysis: QueryAnalysis) -> List[RecipeCard]:
        # Reuse simple Base2 scoring
        query_terms = " ".join([analysis.dish_name or "", *analysis.ingredients]).lower()
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
                score = 0.0
                for t in query_terms.split():
                    if t and t in text.lower():
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

    def get_olj_recommendation(
        self, analysis: QueryAnalysis, exclude_titles: Optional[List[str]] = None
    ) -> Optional[OLJRecommendation]:
        # Simple recommendation
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

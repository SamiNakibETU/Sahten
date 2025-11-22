"""
Reranker
Refines retrieval results using heuristic scoring and optional LLM re-ranking
"""

import logging
from typing import Literal

from app.models.schemas import RetrievalCandidate, QueryPlan
from app.data.normalizers import normalize_text
from app.models.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranks retrieval candidates using multiple signals

    Scoring factors:
    - Base retrieval score (from TF-IDF)
    - Lebanese/Mediterranean relevance
    - Ingredient match (for ingredient queries)
    - Freshness/popularity (for OLJ articles)
    - Constraint satisfaction
    """

    def __init__(self):
        # Lebanese/Mediterranean cuisine indicators
        self.lebanese_indicators = [
            "liban", "libanais", "beyrouth", "méditerranéen",
            "mezze", "tahini", "zaatar", "sumac", "grenade",
            "arak", "laban", "kishk",
        ]

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query_plan: QueryPlan,
        top_k: int | None = None,
    ) -> list[RetrievalCandidate]:
        """
        Rerank candidates and return top-k

        Applies heuristic scoring based on query type and constraints
        """
        if not candidates:
            return []

        if top_k is None:
            top_k = settings.rerank_top_k

        # Calculate final scores
        reranked = []
        for candidate in candidates:
            final_score = self._calculate_final_score(candidate, query_plan)
            candidate.score = final_score
            reranked.append(candidate)

        # Sort by final score
        reranked.sort(key=lambda c: c.score, reverse=True)

        return reranked[:top_k]

    def _calculate_final_score(self, candidate: RetrievalCandidate, query_plan: QueryPlan) -> float:
        """Calculate final score for a candidate"""
        # Start with base retrieval score
        score = candidate.score

        # Factor 1: Lebanese/Mediterranean relevance (10% boost)
        if self._is_lebanese_relevant(candidate):
            score *= 1.1

        # Factor 2: Ingredient match (20% boost for ingredient queries)
        if query_plan.need_type == "recipe_by_ingredients" and query_plan.ingredients:
            ingredient_match_score = self._calculate_ingredient_match(
                candidate, query_plan.ingredients
            )
            score *= (1.0 + ingredient_match_score * 0.2)

        # Factor 3: Primary dish match (30% boost)
        if query_plan.primary_dish:
            if self._matches_primary_dish(candidate, query_plan.primary_dish):
                score *= 1.3

        # Factor 4: Constraint satisfaction (15% boost per constraint)
        if query_plan.constraints:
            constraint_boost = self._calculate_constraint_satisfaction(
                candidate, query_plan.constraints
            )
            score *= (1.0 + constraint_boost * 0.15)

        # Factor 5: Source preference based on need type
        if query_plan.need_type == "recipe_by_ingredients":
            # Prefer Base 2 for ingredient queries
            if candidate.source == "base2":
                score *= 1.15
        elif query_plan.need_type == "recipe_by_name":
            # Slight preference for OLJ for recipe name queries
            if candidate.source == "olj":
                score *= 1.05

        return score

    def _is_lebanese_relevant(self, candidate: RetrievalCandidate) -> bool:
        """Check if candidate is Lebanese/Mediterranean cuisine"""
        content_lower = candidate.content.lower()
        metadata_str = str(candidate.metadata).lower()

        combined = content_lower + " " + metadata_str

        return any(indicator in combined for indicator in self.lebanese_indicators)

    def _calculate_ingredient_match(self, candidate: RetrievalCandidate, ingredients: list[str]) -> float:
        """
        Calculate ingredient match score (0.0 to 1.0)

        Returns fraction of query ingredients that appear in candidate
        """
        if not ingredients:
            return 0.0

        content_lower = candidate.content.lower()

        # For Base 2 candidates, also check metadata ingredients
        if candidate.source == "base2" and "ingredients" in candidate.metadata:
            meta_ingredients = candidate.metadata.get("ingredients", [])
            meta_text = " ".join(str(ing).lower() for ing in meta_ingredients)
            content_lower += " " + meta_text

        normalized_content = normalize_text(content_lower)

        matches = 0
        for ingredient in ingredients:
            normalized_ing = normalize_text(ingredient)
            if normalized_ing in normalized_content:
                matches += 1

        return matches / len(ingredients)

    def _matches_primary_dish(self, candidate: RetrievalCandidate, primary_dish: str) -> bool:
        """Check if candidate matches the primary dish"""
        primary_normalized = normalize_text(primary_dish)

        # Check in content
        if primary_normalized in normalize_text(candidate.content):
            return True

        # Check in metadata
        if candidate.source == "olj":
            title = candidate.metadata.get("title", "")
            if primary_normalized in normalize_text(title):
                return True

        if candidate.source == "base2":
            name = candidate.metadata.get("name", "")
            if primary_normalized in normalize_text(name):
                return True

        return False

    def _calculate_constraint_satisfaction(
        self,
        candidate: RetrievalCandidate,
        constraints: list[str],
    ) -> float:
        """
        Calculate constraint satisfaction score (0.0 to 1.0)

        Returns fraction of constraints satisfied by candidate
        """
        if not constraints:
            return 0.0

        content_lower = candidate.content.lower()
        metadata_str = str(candidate.metadata).lower()
        combined = content_lower + " " + metadata_str

        satisfied = 0
        for constraint in constraints:
            constraint_lower = normalize_text(constraint)
            if constraint_lower in normalize_text(combined):
                satisfied += 1

        return satisfied / len(constraints)

    def deduplicate(
        self,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """
        Remove duplicate candidates

        Duplicates can occur when the same recipe appears in both OLJ and Base 2
        """
        seen_recipes = set()
        deduped = []

        for candidate in candidates:
            # Create a unique key based on content
            if candidate.source == "olj":
                key = f"olj_{candidate.article_id}"
            else:
                key = f"base2_{candidate.recipe_id}"

            if key not in seen_recipes:
                seen_recipes.add(key)
                deduped.append(candidate)

        return deduped

    def diversify(
        self,
        candidates: list[RetrievalCandidate],
        max_per_source: int = 5,
    ) -> list[RetrievalCandidate]:
        """
        Ensure diversity in results by limiting max candidates per source

        Useful to avoid showing only Base 2 or only OLJ results
        """
        olj_count = 0
        base2_count = 0
        diversified = []

        for candidate in candidates:
            if candidate.source == "olj" and olj_count < max_per_source:
                diversified.append(candidate)
                olj_count += 1
            elif candidate.source == "base2" and base2_count < max_per_source:
                diversified.append(candidate)
                base2_count += 1

            # Stop if we have enough from both sources
            if olj_count >= max_per_source and base2_count >= max_per_source:
                break

        return diversified

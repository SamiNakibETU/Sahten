"""
RAG Retriever
Combines content index search with query planning for intelligent retrieval
"""

import logging
from typing import Literal

from app.models.schemas import QueryPlan, RetrievalCandidate
from app.data.content_index import ContentIndex
from app.models.config import settings

logger = logging.getLogger(__name__)


class Retriever:
    """
    RAG retriever that uses QueryPlan to search content index

    Supports:
    - Recipe by name (search both OLJ and Base 2)
    - Recipe by ingredients (prioritize Base 2, but also search OLJ)
    - General suggestions (both sources)
    """

    def __init__(self, content_index: ContentIndex):
        self.content_index = content_index

    def retrieve(self, query_plan: QueryPlan, top_k: int | None = None) -> list[RetrievalCandidate]:
        """
        Retrieve relevant documents based on query plan

        Returns list of RetrievalCandidate objects with source, content, score, and metadata
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        need_type = query_plan.need_type

        # No retrieval needed for these types
        if need_type in ["greeting", "about_bot", "off_topic"]:
            return []

        # Route to appropriate retrieval strategy
        if need_type == "recipe_by_ingredients":
            return self._retrieve_by_ingredients(query_plan, top_k)
        elif need_type == "recipe_by_name":
            return self._retrieve_by_name(query_plan, top_k)
        elif need_type == "suggestions":
            return self._retrieve_suggestions(query_plan, top_k)
        else:
            # Fallback: general search
            return self._retrieve_general(query_plan, top_k)

    def _retrieve_by_ingredients(self, query_plan: QueryPlan, top_k: int) -> list[RetrievalCandidate]:
        """
        Retrieve recipes matching ingredients

        Prioritizes Base 2 structured recipes, but also includes OLJ
        """
        candidates: list[RetrievalCandidate] = []

        # First, search Base 2 using specialized ingredient search
        if query_plan.ingredients:
            base2_results = self.content_index.search_by_ingredients(
                ingredients=query_plan.ingredients,
                top_k=top_k,
            )

            for doc, score in base2_results:
                candidates.append(
                    RetrievalCandidate(
                        source=doc.source,
                        content=doc.content,
                        score=score * 1.2,  # Boost Base 2 for ingredient queries
                        metadata=doc.metadata,
                        article_id=doc.metadata.get("article_id"),
                        recipe_id=doc.metadata.get("recipe_id"),
                    )
                )

        # Also search OLJ for context/storytelling
        olj_results = self.content_index.search(
            query=query_plan.retrieval_query,
            top_k=max(3, top_k // 2),  # Fewer OLJ results
            source_filter="olj",
        )

        for doc, score in olj_results:
            candidates.append(
                RetrievalCandidate(
                    source=doc.source,
                    content=doc.content,
                    score=score * 0.8,  # Lower score for OLJ in ingredient search
                    metadata=doc.metadata,
                    article_id=doc.metadata.get("article_id"),
                    recipe_id=doc.metadata.get("recipe_id"),
                )
            )

        # Sort by score and return top-k
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def _retrieve_by_name(self, query_plan: QueryPlan, top_k: int) -> list[RetrievalCandidate]:
        """
        Retrieve specific recipe by name

        Searches both OLJ and Base 2, prioritizing exact matches
        """
        candidates: list[RetrievalCandidate] = []

        # Search both sources
        all_results = self.content_index.search(
            query=query_plan.retrieval_query,
            top_k=top_k * 2,  # Get more candidates for filtering
            source_filter="all",
        )

        for doc, score in all_results:
            # Boost if primary dish matches document
            boost = 1.0
            if query_plan.primary_dish:
                primary_lower = query_plan.primary_dish.lower()
                if primary_lower in doc.content.lower():
                    boost = 1.3

            candidates.append(
                RetrievalCandidate(
                    source=doc.source,
                    content=doc.content,
                    score=score * boost,
                    metadata=doc.metadata,
                    article_id=doc.metadata.get("article_id"),
                    recipe_id=doc.metadata.get("recipe_id"),
                )
            )

        # Sort by score and return top-k
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def _retrieve_suggestions(self, query_plan: QueryPlan, top_k: int) -> list[RetrievalCandidate]:
        """
        Retrieve general suggestions

        Returns a mix of popular/recent recipes from both sources
        """
        candidates: list[RetrievalCandidate] = []

        # Search with broad query
        results = self.content_index.search(
            query=query_plan.retrieval_query,
            top_k=top_k,
            source_filter="all",
        )

        for doc, score in results:
            candidates.append(
                RetrievalCandidate(
                    source=doc.source,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata,
                    article_id=doc.metadata.get("article_id"),
                    recipe_id=doc.metadata.get("recipe_id"),
                )
            )

        return candidates

    def _retrieve_general(self, query_plan: QueryPlan, top_k: int) -> list[RetrievalCandidate]:
        """Fallback general retrieval"""
        results = self.content_index.search(
            query=query_plan.retrieval_query,
            top_k=top_k,
            source_filter="all",
        )

        candidates = []
        for doc, score in results:
            candidates.append(
                RetrievalCandidate(
                    source=doc.source,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata,
                    article_id=doc.metadata.get("article_id"),
                    recipe_id=doc.metadata.get("recipe_id"),
                )
            )

        return candidates

    def filter_by_constraints(
        self,
        candidates: list[RetrievalCandidate],
        constraints: list[str],
    ) -> list[RetrievalCandidate]:
        """
        Filter candidates by constraints (e.g., vegetarian, quick, etc.)

        This is optional and can be called after retrieve() if needed
        """
        if not constraints:
            return candidates

        filtered = []
        for candidate in candidates:
            content_lower = candidate.content.lower()
            metadata_str = str(candidate.metadata).lower()

            # Check if any constraint is satisfied
            satisfies_constraint = any(
                constraint.lower() in content_lower or constraint.lower() in metadata_str
                for constraint in constraints
            )

            if satisfies_constraint:
                # Boost score for matching constraints
                candidate.score *= 1.1
                filtered.append(candidate)
            else:
                # Still include but with lower priority
                candidate.score *= 0.9
                filtered.append(candidate)

        filtered.sort(key=lambda c: c.score, reverse=True)
        return filtered

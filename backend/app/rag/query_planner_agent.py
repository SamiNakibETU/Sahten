"""
Query Planner Agent
Converts classification results into structured retrieval plans
"""

import logging
from typing import Literal

from app.models.schemas import ClassificationResult, QueryPlan
from app.data.culinary_graph import culinary_graph
from app.data.normalizers import normalize_text

logger = logging.getLogger(__name__)


class QueryPlannerAgent:
    """
    Query planner that creates structured retrieval plans

    Takes ClassificationResult and produces QueryPlan with:
    - need_type: what kind of response is needed
    - retrieval_query: optimized query for content search
    - link_query: query for article resolution
    """

    def plan(self, classification: ClassificationResult, original_query: str) -> QueryPlan:
        """Create a query plan from classification result"""

        intent = classification.intent
        language = classification.language
        slots = classification.slots

        # Extract slot lists
        dishes = slots.get("dishes", [])
        ingredients = slots.get("ingredients", [])
        methods = slots.get("methods", [])
        occasions = slots.get("occasions", [])

        # Determine need_type based on intent and slots
        need_type = self._determine_need_type(intent, dishes, ingredients)

        # Extract primary dish if any
        primary_dish = dishes[0] if dishes else None

        # Build constraints
        constraints = []
        constraints.extend(methods)
        constraints.extend(occasions)

        # Build retrieval query
        retrieval_query = self._build_retrieval_query(
            original_query,
            dishes,
            ingredients,
            methods,
            occasions,
        )

        # Build link query
        link_query = self._build_link_query(
            need_type,
            primary_dish,
            dishes,
            ingredients,
        )

        return QueryPlan(
            need_type=need_type,
            primary_dish=primary_dish,
            ingredients=ingredients,
            constraints=constraints,
            language=language,
            retrieval_query=retrieval_query,
            link_query=link_query,
        )

    def _determine_need_type(
        self,
        intent: str,
        dishes: list[str],
        ingredients: list[str],
    ) -> Literal["recipe_by_ingredients", "recipe_by_name", "suggestions", "off_topic", "greeting", "about_bot"]:
        """Determine what type of response is needed"""

        # Non-food intents
        if intent == "greeting":
            return "greeting"
        elif intent == "farewell":
            return "greeting"  # Treat farewell same as greeting for now
        elif intent == "about_bot":
            return "about_bot"
        elif intent == "off_topic":
            return "off_topic"
        elif intent == "anti_injection":
            return "off_topic"  # Treat injection attempts as off-topic

        # Food requests
        if dishes:
            # User mentioned specific dish(es)
            return "recipe_by_name"
        elif ingredients:
            # User has ingredients and wants suggestions
            return "recipe_by_ingredients"
        else:
            # General suggestions
            return "suggestions"

    def _build_retrieval_query(
        self,
        original_query: str,
        dishes: list[str],
        ingredients: list[str],
        methods: list[str],
        occasions: list[str],
    ) -> str:
        """
        Build optimized retrieval query

        Combines dishes, ingredients, and constraints into a search-optimized query
        """
        query_parts = []

        # Add dishes (highest priority)
        if dishes:
            query_parts.extend(dishes)

        # Add ingredients
        if ingredients:
            query_parts.extend(ingredients)

        # Add methods and occasions
        query_parts.extend(methods)
        query_parts.extend(occasions)

        # If we have extracted terms, use them
        if query_parts:
            return " ".join(query_parts)

        # Otherwise, use normalized original query
        return normalize_text(original_query)

    def _build_link_query(
        self,
        need_type: str,
        primary_dish: str | None,
        dishes: list[str],
        ingredients: list[str],
    ) -> str | None:
        """
        Build query for link resolution

        Returns None for non-food intents
        """
        # No link needed for non-food intents
        if need_type in ["greeting", "about_bot", "off_topic"]:
            return None

        # For recipe by name, use the dish name
        if need_type == "recipe_by_name" and primary_dish:
            return primary_dish

        # For recipe by ingredients, try to find a matching dish from culinary graph
        if need_type == "recipe_by_ingredients" and ingredients:
            # Check if ingredients match a known dish
            for ingredient in ingredients:
                matching_dishes = culinary_graph.get_dishes_by_ingredient(ingredient)
                if matching_dishes:
                    return matching_dishes[0]  # Return first match

            # Otherwise use ingredients as query
            return " ".join(ingredients)

        # For suggestions, return a general Lebanese cuisine query
        if need_type == "suggestions":
            return "recettes libanaises"

        # Fallback
        return "recettes"

    def refine_with_context(
        self,
        plan: QueryPlan,
        conversation_history: list[dict] | None = None,
    ) -> QueryPlan:
        """
        Refine query plan with conversation context

        This can be used for multi-turn conversations (future enhancement)
        For now, returns the plan as-is
        """
        # TODO: Implement conversation context handling
        # - Track what recipes were already suggested
        # - Handle follow-up questions ("Et pour dessert?")
        # - Maintain user preferences

        return plan

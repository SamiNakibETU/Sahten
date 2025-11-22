"""
Scenario Alignment
Maps query understanding to editorial scenarios
"""

import logging
from typing import Literal

from app.models.schemas import (
    ClassificationResult,
    QueryPlan,
    LinkResolutionResult,
    RetrievalCandidate,
    ScenarioContext,
)

logger = logging.getLogger(__name__)


# Scenario definitions
SCENARIOS = {
    1: {
        "name": "olj_recipe_available",
        "description": "OLJ recipe exists, storytelling only with link",
        "use_base": "olj",
        "show_full_recipe": False,
        "include_link": True,
    },
    2: {
        "name": "base2_recipe_with_olj_suggestion",
        "description": "Base 2 recipe with OLJ article suggestion",
        "use_base": "base2",
        "show_full_recipe": True,
        "include_link": True,
    },
    3: {
        "name": "no_match_with_fallback",
        "description": "No match found, suggest fallback OLJ article",
        "use_base": "none",
        "show_full_recipe": False,
        "include_link": True,
    },
    4: {
        "name": "greeting",
        "description": "User greeting with OLJ suggestion",
        "use_base": "none",
        "show_full_recipe": False,
        "include_link": True,
    },
    5: {
        "name": "about_bot",
        "description": "Bot self-description with OLJ example",
        "use_base": "none",
        "show_full_recipe": False,
        "include_link": True,
    },
    6: {
        "name": "off_topic_redirect",
        "description": "Off-topic query, redirect to cuisine + OLJ link",
        "use_base": "none",
        "show_full_recipe": False,
        "include_link": True,
    },
    7: {
        "name": "non_french_polite_decline",
        "description": "Non-French query, polite decline in French",
        "use_base": "none",
        "show_full_recipe": False,
        "include_link": False,
    },
    8: {
        "name": "ingredient_suggestions",
        "description": "Multiple recipe suggestions based on ingredients",
        "use_base": "mixed",
        "show_full_recipe": False,
        "include_link": True,
    },
}


class ScenarioAligner:
    """
    Determines which scenario to use based on query analysis

    Scenarios:
    1. OLJ recipe available → storytelling only, link required
    2. Base 2 recipe → full recipe, OLJ suggestion
    3. No match → sorry message, fallback OLJ link
    4. Greeting → warm welcome, OLJ suggestion
    5. About bot → self-description, OLJ example
    6. Off-topic → redirect to cooking, OLJ link
    7. Non-French → polite decline in French
    8. Ingredient query → multiple suggestions
    """

    def align(
        self,
        classification: ClassificationResult,
        query_plan: QueryPlan,
        link_result: LinkResolutionResult,
        retrieval_candidates: list[RetrievalCandidate] | None = None,
    ) -> ScenarioContext:
        """
        Determine the appropriate scenario

        Decision tree:
        1. Check language (non-French → scenario 7)
        2. Check intent (greeting/about_bot/off_topic → scenarios 4/5/6)
        3. Check link availability (OLJ match → scenario 1)
        4. Check retrieval candidates (Base 2 → scenario 2)
        5. Fallback (scenario 3)
        """

        # Language check
        if classification.language == "non_fr":
            return self._create_context(7)

        # Intent-based scenarios
        if classification.intent == "greeting" or classification.intent == "farewell":
            return self._create_context(4)

        if classification.intent == "about_bot":
            return self._create_context(5)

        if classification.intent == "off_topic" or classification.intent == "anti_injection":
            return self._create_context(6)

        # Food request scenarios
        if classification.intent == "food_request":
            # Check if we have a good OLJ match
            if link_result.primary_article and link_result.confidence > 0.6:
                # High confidence OLJ match → scenario 1
                return self._create_context(1)

            # Check if we have Base 2 recipes
            if retrieval_candidates:
                base2_candidates = [c for c in retrieval_candidates if c.source == "base2"]

                if base2_candidates and base2_candidates[0].score > 0.4:
                    # Good Base 2 match → scenario 2
                    return self._create_context(2)

            # Ingredient query with multiple options
            if query_plan.need_type == "recipe_by_ingredients" and len(query_plan.ingredients) > 1:
                return self._create_context(8)

            # No good match → fallback
            return self._create_context(3)

        # Default fallback
        return self._create_context(3)

    def _create_context(self, scenario_id: int) -> ScenarioContext:
        """Create ScenarioContext from scenario ID"""
        scenario_def = SCENARIOS.get(scenario_id)

        if not scenario_def:
            logger.error(f"Unknown scenario ID: {scenario_id}")
            scenario_def = SCENARIOS[3]  # Fallback to scenario 3

        return ScenarioContext(
            scenario_id=scenario_id,
            scenario_name=scenario_def["name"],
            use_base=scenario_def["use_base"],
            show_full_recipe=scenario_def["show_full_recipe"],
            include_link=scenario_def["include_link"],
        )

    def get_scenario_description(self, scenario_id: int) -> str:
        """Get human-readable scenario description"""
        scenario = SCENARIOS.get(scenario_id)
        return scenario["description"] if scenario else "Unknown scenario"

"""
Ingredient Normalizer
Handles ingredient synonyms and equivalences for better search matching
Supports French/English, alternative spellings, and common variations
"""

import logging
from dataclasses import dataclass

from app.data.normalizers import normalize_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedIngredient:
    """Structured normalized ingredient with alternative flag."""

    normalized: str
    original: str
    is_alternative: bool = False

    def __contains__(self, item: str) -> bool:
        """Allow substring checks in tests like `'poulet' in normalized_ingredient`."""
        if item is None:
            return False
        lowered = (item or "").lower()
        return lowered in (self.normalized or "") or lowered in (self.original or "")

TYPO_CORRECTIONS = {
    "yaour": "yaourt",
    "yourt": "yaourt",
    "yogurt": "yaourt",
    "yogourt": "yaourt",
    "youghourt": "yaourt",
    "yaourth": "yaourt",
    "poullet": "poulet",
    "poulett": "poulet",
    "tomat": "tomate",
    "oigon": "oignon",
    "oinon": "oignon",
    "citro": "citron",
    "aubergin": "aubergine",
    "courgett": "courgette",
}


# Ingredient equivalence groups (all variations that mean the same thing)
INGREDIENT_EQUIVALENCES = [
    # Chickpeas
    ["pois chiches", "chickpeas", "garbanzo", "pois chiche"],

    # Tahini
    ["tahini", "tahine", "tahin", "crème de sésame", "sesame paste"],

    # Lemon
    ["citron", "lemon", "jus de citron", "lemon juice"],

    # Garlic
    ["ail", "garlic", "gousse d'ail", "garlic clove"],

    # Eggplant/Aubergine
    ["aubergine", "eggplant"],

    # Yogurt
    ["yaourt", "yogurt", "yoghurt", "laban"],

    # Parsley
    ["persil", "parsley"],

    # Bulgur
    ["boulgour", "bulgur", "bulghur"],

    # Tomato
    ["tomate", "tomato", "tomates", "tomatoes"],

    # Onion
    ["oignon", "onion", "oignons", "onions"],

    # Olive oil
    ["huile d'olive", "olive oil", "huile olive"],

    # Meat
    ["viande", "meat", "viande hachée", "ground meat", "minced meat"],

    # Chicken
    ["poulet", "chicken"],

    # Lamb
    ["agneau", "lamb"],

    # Rice
    ["riz", "rice"],

    # Fava beans
    ["fèves", "fava beans", "broad beans", "feves"],

    # Green beans
    ["haricots verts", "green beans"],

    # White beans
    ["haricots blancs", "white beans"],

    # Sumac
    ["sumac", "sumaq"],

    # Pomegranate
    ["grenade", "pomegranate", "mélasse de grenade", "pomegranate molasses"],

    # Mint
    ["menthe", "mint"],

    # Cucumber
    ["concombre", "cucumber"],

    # Zucchini/Courgette
    ["courgette", "zucchini"],

    # Potato
    ["pomme de terre", "potato", "potatoes", "patate", "patates"],

    # Spinach
    ["épinards", "spinach"],

    # Cheese
    ["fromage", "cheese"],

    # Bread
    ["pain", "bread"],

    # Nuts (general)
    ["noix", "nuts", "walnuts"],

    # Pine nuts
    ["pignons", "pine nuts", "pignons de pin"],

    # Pistachios
    ["pistache", "pistachio", "pistaches", "pistachios"],

    # Dates
    ["dattes", "dates", "datte", "date"],

    # Semolina
    ["semoule", "semolina"],

    # Flour
    ["farine", "flour"],

    # Sugar
    ["sucre", "sugar"],

    # Milk
    ["lait", "milk"],

    # Cream
    ["crème", "cream"],

    # Cardamom
    ["cardamome", "cardamom"],

    # Cinnamon
    ["cannelle", "cinnamon"],

    # Red pepper
    ["poivron rouge", "red pepper", "red bell pepper"],

    # Hot pepper/chili
    ["piment", "chili", "hot pepper"],

    # Okra
    ["gombo", "okra", "bamia"],

    # Vine leaves
    ["feuilles de vigne", "vine leaves", "grape leaves"],

    # Cabbage
    ["chou", "cabbage"],

    # Arugula/Rocket
    ["roquette", "arugula", "rocket"],

    # Dandelion greens
    ["pissenlit", "dandelion greens"],

    # Cauliflower
    ["chou-fleur", "cauliflower"],

    # Fish
    ["poisson", "fish"],

    # Liver
    ["foie", "liver"],

    # Freekeh
    ["freekeh", "frikeh", "farik"],

    # Rose water
    ["eau de rose", "rose water"],

    # Orange blossom water
    ["eau de fleur d'oranger", "orange blossom water"],

    # Sesame
    ["sésame", "sesame"],
]

ALTERNATIVE_EQUIVALENCES = {
    "mozzarella": "halloumi",
    "parmesan": "akkawi",
    "creme fraiche": "labneh",
    "pates": "moghrabieh",
}


class IngredientNormalizer:
    """
    Normalizes ingredients and finds equivalences
    """

    def __init__(self):
        # Build reverse mapping: normalized ingredient -> equivalence group
        self.equivalence_map: dict[str, set[str]] = {}
        self.alternative_map: dict[str, str] = {}

        for group in INGREDIENT_EQUIVALENCES:
            # Normalize all terms in group
            normalized_group = {normalize_text(ing) for ing in group}

            # Map each normalized term to the full group
            for norm_ing in normalized_group:
                self.equivalence_map[norm_ing] = normalized_group

        for source, target in ALTERNATIVE_EQUIVALENCES.items():
            normalized_source = normalize_text(source)
            normalized_target = normalize_text(target)
            if normalized_source and normalized_target:
                self.alternative_map[normalized_source] = normalized_target

        logger.info(f"Built ingredient normalizer with {len(INGREDIENT_EQUIVALENCES)} equivalence groups")

    def get_equivalents(self, ingredient: str) -> set[str]:
        """
        Get all equivalent forms of an ingredient

        Args:
            ingredient: The ingredient to find equivalents for

        Returns:
            Set of normalized equivalent ingredient names (including the original)
        """
        corrected = self._apply_typo_corrections(ingredient)
        normalized = normalize_text(corrected)
        normalized = TYPO_CORRECTIONS.get(normalized, normalized)

        # Check direct match
        if normalized in self.equivalence_map:
            return self.equivalence_map[normalized]

        # Check partial match (ingredient might be part of a phrase)
        for key, equivalents in self.equivalence_map.items():
            if normalized in key or key in normalized:
                return equivalents

        # No equivalents found, return just the normalized form
        return {normalized}

    def normalize_ingredient_list(self, ingredients: list[str]) -> list[NormalizedIngredient]:
        """
        Normalize a list of ingredients to their canonical forms

        Args:
            ingredients: List of ingredient names

        Returns:
            List of structured normalized ingredients with alternative flags
        """
        normalized_results: list[NormalizedIngredient] = []
        seen: set[tuple[str, bool]] = set()

        for ingredient in ingredients or []:
            if not ingredient:
                continue
            # Get all equivalents
            equivalents = self.get_equivalents(ingredient)

            # Add all equivalents to enable broader matching
            for equivalent in equivalents:
                key = (equivalent, False)
                if key in seen:
                    continue
                normalized_results.append(
                    NormalizedIngredient(
                        normalized=equivalent,
                        original=ingredient,
                        is_alternative=False,
                    )
                )
                seen.add(key)

            normalized = normalize_text(self._apply_typo_corrections(ingredient))
            normalized = TYPO_CORRECTIONS.get(normalized, normalized)
            alternative = self.alternative_map.get(normalized)
            if alternative:
                key = (alternative, True)
                if key not in seen:
                    normalized_results.append(
                        NormalizedIngredient(
                            normalized=alternative,
                            original=ingredient,
                            is_alternative=True,
                        )
                    )
                    seen.add(key)

        return normalized_results

    def match_ingredients(
        self,
        query_ingredients: list[str],
        doc_ingredients: list[str],
    ) -> tuple[int, float]:
        """
        Match ingredients with equivalence support

        Args:
            query_ingredients: Ingredients from user query
            doc_ingredients: Ingredients from document

        Returns:
            Tuple of (match_count, match_ratio)
        """
        # Normalize both lists with equivalences
        query_norm = self._unique_terms(self.normalize_ingredient_list(query_ingredients))
        doc_norm = self._unique_terms(self.normalize_ingredient_list(doc_ingredients))

        # Count matches (with equivalence)
        matches = 0
        for q_ing in query_norm:
            for d_ing in doc_norm:
                if q_ing in d_ing or d_ing in q_ing:
                    matches += 1
                    break  # Count each query ingredient only once

        # Calculate ratio based on original query length
        ratio = matches / len(query_ingredients) if query_ingredients else 0.0

        return matches, ratio

    def _unique_terms(self, normalized: list[NormalizedIngredient]) -> list[str]:
        """Extract unique normalized strings from structured results."""
        seen: set[str] = set()
        ordered: list[str] = []
        for item in normalized:
            if item.normalized not in seen:
                ordered.append(item.normalized)
                seen.add(item.normalized)
        return ordered

    def _apply_typo_corrections(self, ingredient: str | None) -> str:
        if not ingredient:
            return ""
        lowered = ingredient.strip().lower()
        for typo, correction in TYPO_CORRECTIONS.items():
            if lowered == typo:
                return correction
            if lowered.startswith(f"{typo} "):
                remainder = lowered[len(typo):]
                return f"{correction}{remainder}".strip()
        return lowered


# Global instance
ingredient_normalizer = IngredientNormalizer()

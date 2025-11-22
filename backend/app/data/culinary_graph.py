"""
Culinary Knowledge Graph
Manages relationships between dishes, ingredients, and cuisines
Helps with query understanding and retrieval
"""

import logging
from typing import Literal
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DishInfo:
    """Information about a Lebanese/Mediterranean dish"""

    name: str
    normalized: str
    category: Literal[
        "mezze_cold",
        "mezze_hot",
        "main_course",
        "salad",
        "soup",
        "dessert",
        "drink",
        "bread",
    ]
    key_ingredients: list[str] = field(default_factory=list)
    variations: list[str] = field(default_factory=list)
    occasions: list[str] = field(default_factory=list)
    is_lebanese: bool = True


class CulinaryGraph:
    """
    Knowledge graph of Lebanese and Mediterranean cuisine
    Helps with query understanding and retrieval
    """

    def __init__(self):
        self.dishes: dict[str, DishInfo] = {}
        self._build_knowledge_base()

    def _build_knowledge_base(self):
        """Build the culinary knowledge graph"""

        # Mezzes froids (Cold mezzes)
        self._add_dish("hummus", "mezze_cold", ["pois chiches", "tahini", "citron"], ["houmous", "hommos"])
        self._add_dish("moutabbal", "mezze_cold", ["aubergine", "tahini", "citron"], ["mutabbal", "baba ganoush"])
        self._add_dish("labneh", "mezze_cold", ["yaourt", "ail"], ["labne", "labné"])
        self._add_dish("tabbouleh", "mezze_cold", ["persil", "boulgour", "tomate"], ["taboulé", "taboule"])
        self._add_dish("fattoush", "mezze_cold", ["salade", "pain", "sumac"], occasions=["déjeuner"])
        self._add_dish("warak enab", "mezze_cold", ["feuilles de vigne", "riz"], ["feuilles de vigne farcies"])

        # Mezzes chauds (Hot mezzes)
        self._add_dish("kebbeh", "mezze_hot", ["viande", "boulgour"], ["kibbeh", "kibbe"])
        self._add_dish("sambousek", "mezze_hot", ["pâte", "viande", "fromage"], ["samosa libanais"])
        self._add_dish("falafel", "mezze_hot", ["pois chiches", "fèves", "épices"])
        self._add_dish("fatayer", "mezze_hot", ["pâte", "épinards", "viande"])
        self._add_dish("makanek", "mezze_hot", ["saucisse", "citron", "grenade"])

        # Plats principaux (Main courses)
        self._add_dish("kafta", "main_course", ["viande hachée", "persil", "oignon"], ["kofta", "kefta"])
        self._add_dish("shawarma", "main_course", ["viande", "épices"], ["chawarma"])
        self._add_dish("moghrabieh", "main_course", ["perles", "poulet", "pois chiches"], ["maftoul"])
        self._add_dish("sayadieh", "main_course", ["poisson", "riz", "oignon caramélisé"])
        self._add_dish("tajine", "main_course", ["viande", "légumes"], is_lebanese=False)

        # Desserts
        self._add_dish("baklava", "dessert", ["pâte filo", "noix", "sirop"])
        self._add_dish("kunefe", "dessert", ["kadaif", "fromage", "sirop"], ["knafeh", "kenefeh"])
        self._add_dish("halva", "dessert", ["tahini", "sucre"], ["halawa"])
        self._add_dish("maamoul", "dessert", ["semoule", "dattes", "noix"])

        # Soupes
        self._add_dish("lentil soup", "soup", ["lentilles", "citron"], ["chorba adas"])

        logger.info(f"Built culinary graph with {len(self.dishes)} dishes")

    def _add_dish(
        self,
        name: str,
        category: str,
        key_ingredients: list[str] = None,
        variations: list[str] = None,
        occasions: list[str] = None,
        is_lebanese: bool = True,
    ):
        """Add a dish to the knowledge graph"""
        from app.data.normalizers import normalize_recipe_name

        normalized = normalize_recipe_name(name)

        self.dishes[normalized] = DishInfo(
            name=name,
            normalized=normalized,
            category=category,
            key_ingredients=key_ingredients or [],
            variations=variations or [],
            occasions=occasions or [],
            is_lebanese=is_lebanese,
        )

        # Also add variations
        for variation in variations or []:
            normalized_var = normalize_recipe_name(variation)
            if normalized_var not in self.dishes:
                self.dishes[normalized_var] = self.dishes[normalized]

    def find_dish(self, query: str) -> DishInfo | None:
        """Find a dish by name or variation"""
        from app.data.normalizers import normalize_recipe_name

        normalized = normalize_recipe_name(query)

        # Direct lookup
        if normalized in self.dishes:
            return self.dishes[normalized]

        # Partial match
        for dish_norm, dish_info in self.dishes.items():
            if normalized in dish_norm or dish_norm in normalized:
                return dish_info

        return None

    def is_lebanese_dish(self, dish_name: str) -> bool:
        """Check if a dish is Lebanese/Mediterranean"""
        dish = self.find_dish(dish_name)
        return dish.is_lebanese if dish else False

    def get_dish_category(self, dish_name: str) -> str | None:
        """Get the category of a dish"""
        dish = self.find_dish(dish_name)
        return dish.category if dish else None

    def get_key_ingredients(self, dish_name: str) -> list[str]:
        """Get key ingredients for a dish"""
        dish = self.find_dish(dish_name)
        return dish.key_ingredients if dish else []

    def get_dishes_by_category(self, category: str) -> list[str]:
        """Get all dishes in a category"""
        return [
            dish.name
            for dish in self.dishes.values()
            if dish.category == category
        ]

    def get_dishes_by_ingredient(self, ingredient: str) -> list[str]:
        """Get dishes that use a specific ingredient"""
        from app.data.normalizers import normalize_text

        normalized_ing = normalize_text(ingredient)

        matching_dishes = []
        for dish in self.dishes.values():
            for key_ing in dish.key_ingredients:
                if normalized_ing in normalize_text(key_ing):
                    matching_dishes.append(dish.name)
                    break

        return matching_dishes


# Global culinary graph instance
culinary_graph = CulinaryGraph()

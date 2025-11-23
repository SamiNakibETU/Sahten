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
        self._add_dish("shanklish", "mezze_cold", ["fromage", "thym", "tomate"], ["chanklish"])
        self._add_dish("muhammara", "mezze_cold", ["poivron rouge", "noix", "mélasse de grenade"])
        self._add_dish("raheb", "mezze_cold", ["aubergine", "tomate", "grenade"], ["salade d'aubergine"])
        self._add_dish("balila", "mezze_cold", ["pois chiches", "citron", "ail"])
        self._add_dish("mtabbal batata", "mezze_cold", ["pomme de terre", "tahini"], ["purée de pomme de terre"])
        self._add_dish("hindbeh", "mezze_cold", ["pissenlit", "oignon", "citron"], ["salade de pissenlit"])
        self._add_dish("arnabeet", "mezze_cold", ["chou-fleur", "tahini"], ["chou-fleur frit"])

        # Mezzes chauds (Hot mezzes)
        self._add_dish("kebbeh", "mezze_hot", ["viande", "boulgour"], ["kibbeh", "kibbe"])
        self._add_dish("sambousek", "mezze_hot", ["pâte", "viande", "fromage"], ["samosa libanais"])
        self._add_dish("falafel", "mezze_hot", ["pois chiches", "fèves", "épices"])
        self._add_dish("fatayer", "mezze_hot", ["pâte", "épinards", "viande"])
        self._add_dish("makanek", "mezze_hot", ["saucisse", "citron", "grenade"])
        self._add_dish("soujouk", "mezze_hot", ["saucisse", "épices"], ["sujuk"])
        self._add_dish("kibbeh nayeh", "mezze_hot", ["viande crue", "boulgour"], ["kebbeh cru"])
        self._add_dish("batata harra", "mezze_hot", ["pomme de terre", "ail", "piment"], ["pommes de terre épicées"])
        self._add_dish("jawaneh", "mezze_hot", ["ailes de poulet", "ail", "citron"])
        self._add_dish("ras asfour", "mezze_hot", ["foie", "ail", "citron"])
        self._add_dish("awarma", "mezze_hot", ["viande confite", "graisse"])
        self._add_dish("cheese rolls", "mezze_hot", ["fromage", "pâte"], ["rouleaux au fromage"])
        self._add_dish("fatayer sabanekh", "mezze_hot", ["pâte", "épinards"], ["chaussons aux épinards"])

        # Plats principaux (Main courses)
        self._add_dish("kafta", "main_course", ["viande hachée", "persil", "oignon"], ["kofta", "kefta"])
        self._add_dish("shawarma", "main_course", ["viande", "épices"], ["chawarma"])
        self._add_dish("moghrabieh", "main_course", ["perles", "poulet", "pois chiches"], ["maftoul"])
        self._add_dish("sayadieh", "main_course", ["poisson", "riz", "oignon caramélisé"])
        self._add_dish("tajine", "main_course", ["viande", "légumes"], is_lebanese=False)
        self._add_dish("kibbeh bil sanieh", "main_course", ["viande", "boulgour"], ["kebbeh au four"])
        self._add_dish("loubieh bi zeit", "main_course", ["haricots verts", "tomate", "huile d'olive"])
        self._add_dish("bamia", "main_course", ["gombo", "viande", "tomate"], ["okra"])
        self._add_dish("fasoulia", "main_course", ["haricots blancs", "viande", "tomate"])
        self._add_dish("mousakaa", "main_course", ["aubergine", "pois chiches", "tomate"], ["moussaka"])
        self._add_dish("sheikh el mahshi", "main_course", ["aubergine farcie", "viande", "tomate"])
        self._add_dish("malfouf", "main_course", ["chou farci", "riz", "viande"])
        self._add_dish("kousa mahshi", "main_course", ["courgette farcie", "riz", "viande"])
        self._add_dish("mehshi warak enab", "main_course", ["feuilles de vigne", "viande", "riz"])
        self._add_dish("dawood basha", "main_course", ["boulettes", "tomate", "pignons"])
        self._add_dish("chicken freekeh", "main_course", ["poulet", "freekeh"], ["poulet au freekeh"])
        self._add_dish("fatteh", "main_course", ["yaourt", "pois chiches", "pain"], ["fatteh hummus"])
        self._add_dish("fatteh djaj", "main_course", ["poulet", "yaourt", "pain"], ["fatteh au poulet"])
        self._add_dish("makloubeh", "main_course", ["aubergine", "riz", "viande"], ["maqluba"])
        self._add_dish("ouzi", "main_course", ["agneau", "riz", "pâte"], ["ouzi libanais"])

        # Salades
        self._add_dish("salata lebnanaise", "salad", ["tomate", "concombre", "citron"], ["salade libanaise"])
        self._add_dish("rocca salad", "salad", ["roquette", "tomate", "citron"], ["salade de roquette"])
        self._add_dish("cabbage salad", "salad", ["chou", "citron"], ["salade de chou"])
        self._add_dish("cucumber yogurt", "salad", ["concombre", "yaourt", "menthe"], ["salade de concombre"])

        # Soupes
        self._add_dish("lentil soup", "soup", ["lentilles", "citron"], ["chorba adas"])
        self._add_dish("freekeh soup", "soup", ["freekeh", "poulet", "légumes"])
        self._add_dish("chicken soup", "soup", ["poulet", "légumes", "vermicelles"])
        self._add_dish("vegetable soup", "soup", ["légumes", "tomate"], ["soupe aux légumes"])

        # Desserts
        self._add_dish("baklava", "dessert", ["pâte filo", "noix", "sirop"])
        self._add_dish("kunefe", "dessert", ["kadaif", "fromage", "sirop"], ["knafeh", "kenefeh"])
        self._add_dish("halva", "dessert", ["tahini", "sucre"], ["halawa"])
        self._add_dish("maamoul", "dessert", ["semoule", "dattes", "noix"])
        self._add_dish("atayef", "dessert", ["crêpe", "fromage", "noix", "sirop"], ["qatayef"])
        self._add_dish("halawet el jibn", "dessert", ["fromage", "semoule", "sirop"])
        self._add_dish("riz bi halib", "dessert", ["riz", "lait", "eau de rose"], ["riz au lait"])
        self._add_dish("mouhalabieh", "dessert", ["lait", "fécule", "eau de rose"], ["crème de lait"])
        self._add_dish("namoura", "dessert", ["semoule", "yaourt", "sirop"], ["basbousa"])
        self._add_dish("sfouf", "dessert", ["semoule", "curcuma", "anis"])
        self._add_dish("mafroukeh", "dessert", ["semoule", "pistache", "sirop"])
        self._add_dish("aish el saraya", "dessert", ["pain", "crème", "pistache"])
        self._add_dish("layali lubnan", "dessert", ["semoule", "crème", "pistache"], ["nuits libanaises"])

        # Pains
        self._add_dish("manakish", "bread", ["pâte", "zaatar", "huile"], ["manaeesh"])
        self._add_dish("manakish zaatar", "bread", ["pâte", "zaatar"], ["zaatar manakish"])
        self._add_dish("manakish jibneh", "bread", ["pâte", "fromage"], ["cheese manakish"])
        self._add_dish("kaak", "bread", ["sésame", "pâte"], ["pain au sésame"])
        self._add_dish("pita", "bread", ["pâte"], ["pain pita"])
        self._add_dish("saj bread", "bread", ["pâte"], ["pain saj"])
        self._add_dish("markouk", "bread", ["pâte"], ["pain markouk"])

        # Boissons
        self._add_dish("jallab", "drink", ["mélasse de datte", "eau de rose", "pignons"])
        self._add_dish("ayran", "drink", ["yaourt", "eau", "sel"], ["laban"])
        self._add_dish("white coffee", "drink", ["eau de fleur d'oranger"], ["café blanc"])
        self._add_dish("lemonade mint", "drink", ["citron", "menthe"], ["limonade à la menthe"])
        self._add_dish("lebanese coffee", "drink", ["café", "cardamome"], ["café libanais"])

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

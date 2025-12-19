"""
Dish normalizer mapping aliases to canonical dishes and source hints.
"""

from __future__ import annotations

import difflib
import logging
from typing import Iterable

from app.data.normalizers import normalize_text

logger = logging.getLogger(__name__)

# alias -> (canonical, [ingredients], base2_only)
DISH_ALIASES: dict[str, tuple[str, list[str], bool]] = {
    # Format: "alias": ("canonical", ["ingredients"], is_base2_only)

    # === TABOULE ===
    "taboule": ("tabbouleh", ["persil", "boulgour", "tomate", "oignon", "citron"], False),
    "taboul\u00E9": ("tabbouleh", ["persil", "boulgour", "tomate", "oignon", "citron"], False),
    "tabouleh": ("tabbouleh", ["persil", "boulgour", "tomate", "oignon", "citron"], False),
    "tabbouleh": ("tabbouleh", ["persil", "boulgour", "tomate", "oignon", "citron"], False),
    "tabouli": ("tabbouleh", ["persil", "boulgour", "tomate", "oignon", "citron"], False),
    "tabbouli": ("tabbouleh", ["persil", "boulgour", "tomate", "oignon", "citron"], False),
    "tabboul\u00E9": ("tabbouleh", ["persil", "boulgour", "tomate", "oignon", "citron"], False),

    # === KEBBEH VARIANTS ===
    "kebbe": ("kibbeh", ["viande", "boulgour", "oignon", "pignons"], False),
    "kebb\u00E9": ("kibbeh", ["viande", "boulgour", "oignon", "pignons"], False),
    "kebbeh": ("kibbeh", ["viande", "boulgour", "oignon", "pignons"], False),
    "kibbe": ("kibbeh", ["viande", "boulgour", "oignon", "pignons"], False),
    "kibbeh": ("kibbeh", ["viande", "boulgour", "oignon", "pignons"], False),
    "kubbeh": ("kibbeh", ["viande", "boulgour", "oignon", "pignons"], False),
    "kibbi": ("kibbeh", ["viande", "boulgour", "oignon", "pignons"], False),
    "kebbe nayye": ("kibbeh_nayyeh", ["viande crue", "boulgour", "oignon"], False),
    "kebb\u00E9 nayy\u00E9": ("kibbeh_nayyeh", ["viande crue", "boulgour", "oignon"], False),
    "kibbeh nayyeh": ("kibbeh_nayyeh", ["viande crue", "boulgour", "oignon"], False),
    "kebbe cru": ("kibbeh_nayyeh", ["viande crue", "boulgour", "oignon"], False),

    # === FEUILLES DE VIGNE / WARAK ENAB ===
    "feuilles de vigne": ("warak_enab", ["feuilles de vigne", "riz", "viande"], False),
    "feuilles de vignes": ("warak_enab", ["feuilles de vigne", "riz", "viande"], False),
    "feuille de vigne": ("warak_enab", ["feuilles de vigne", "riz", "viande"], False),
    "warak enab": ("warak_enab", ["feuilles de vigne", "riz", "viande"], False),
    "warak inab": ("warak_enab", ["feuilles de vigne", "riz", "viande"], False),
    "waraq enab": ("warak_enab", ["feuilles de vigne", "riz", "viande"], False),
    "dawali": ("warak_enab", ["feuilles de vigne", "riz", "viande"], False),
    "dolma": ("warak_enab", ["feuilles de vigne", "riz", "viande"], False),
    "yaprak": ("warak_enab", ["feuilles de vigne", "riz", "viande"], False),

    # === CHICH TAOUK (BASE2 ONLY) ===
    "chich taouk": ("chich_taouk", ["poulet", "yaourt", "ail", "citron"], True),
    "shish taouk": ("chich_taouk", ["poulet", "yaourt", "ail", "citron"], True),
    "shish tawook": ("chich_taouk", ["poulet", "yaourt", "ail", "citron"], True),
    "chichtaouk": ("chich_taouk", ["poulet", "yaourt", "ail", "citron"], True),
    "taouk": ("chich_taouk", ["poulet", "yaourt", "ail", "citron"], True),
    "tawook": ("chich_taouk", ["poulet", "yaourt", "ail", "citron"], True),

    # === CHAWARMA ===
    "chawarma": ("shawarma", ["poulet", "epices", "pain"], False),
    "shawarma": ("shawarma", ["poulet", "epices", "pain"], False),
    "shawurma": ("shawarma", ["poulet", "epices", "pain"], False),
    "shawerma": ("shawarma", ["poulet", "epices", "pain"], False),

    # === HUMMUS (BASE2 ONLY) ===
    "hummus": ("hummus", ["pois chiches", "tahini", "citron", "ail"], True),
    "houmous": ("hummus", ["pois chiches", "tahini", "citron", "ail"], True),
    "hommos": ("hummus", ["pois chiches", "tahini", "citron", "ail"], True),
    "houmos": ("hummus", ["pois chiches", "tahini", "citron", "ail"], True),
    "humus": ("hummus", ["pois chiches", "tahini", "citron", "ail"], True),

    # === FALAFEL (BASE2 ONLY) ===
    "falafel": ("falafel", ["pois chiches", "feves", "persil", "coriandre"], True),
    "falafels": ("falafel", ["pois chiches", "feves", "persil", "coriandre"], True),

    # === FATTOUSH (BASE2 ONLY) ===
    "fattoush": ("fattoush", ["pain pita", "legumes", "sumac", "pourpier"], True),
    "fattouche": ("fattoush", ["pain pita", "legumes", "sumac", "pourpier"], True),
    "fatoush": ("fattoush", ["pain pita", "legumes", "sumac", "pourpier"], True),
    "fattouch": ("fattoush", ["pain pita", "legumes", "sumac", "pourpier"], True),

    # === FATTEH (OLJ) ===
    "fatteh": ("fatteh", ["pain", "pois chiches", "yaourt", "pignons"], False),
    "fatte": ("fatteh", ["pain", "pois chiches", "yaourt", "pignons"], False),
    "fetteh": ("fatteh", ["pain", "pois chiches", "yaourt", "pignons"], False),
    "fatt\u00E9": ("fatteh", ["pain", "pois chiches", "yaourt", "pignons"], False),

    # === MOUTABBAL / BABA GHANOUSH ===
    "moutabbal": ("moutabbal", ["aubergine", "tahini", "ail", "citron"], False),
    "mutabbal": ("moutabbal", ["aubergine", "tahini", "ail", "citron"], False),
    "mtabbal": ("moutabbal", ["aubergine", "tahini", "ail", "citron"], False),
    "baba ghanoush": ("moutabbal", ["aubergine", "tahini", "ail", "citron"], False),
    "baba ganoush": ("moutabbal", ["aubergine", "tahini", "ail", "citron"], False),
    "baba ganouj": ("moutabbal", ["aubergine", "tahini", "ail", "citron"], False),

    # === LABNE (BASE2) ===
    "labne": ("labne", ["yaourt", "sel"], True),
    "labn\u00E9": ("labne", ["yaourt", "sel"], True),
    "labneh": ("labne", ["yaourt", "sel"], True),
    "labna": ("labne", ["yaourt", "sel"], True),

    # === KAFTA (OLJ) ===
    "kafta": ("kafta", ["viande hachee", "oignon", "persil"], False),
    "kofta": ("kafta", ["viande hachee", "oignon", "persil"], False),
    "kefta": ("kafta", ["viande hachee", "oignon", "persil"], False),
    "kefte": ("kafta", ["viande hachee", "oignon", "persil"], False),

    # === SAMBOUSSEK (OLJ) ===
    "samboussek": ("samboussek", ["pate", "viande", "oignon", "pignons"], False),
    "sambousek": ("samboussek", ["pate", "viande", "oignon", "pignons"], False),
    "sambousik": ("samboussek", ["pate", "viande", "oignon", "pignons"], False),
    "samoussa": ("samboussek", ["pate", "viande", "oignon", "pignons"], False),

    # === MANAKISH (OLJ) ===
    "manakish": ("manakish", ["pate", "zaatar", "huile d'olive"], False),
    "mana'ich": ("manakish", ["pate", "zaatar", "huile d'olive"], False),
    "manouche": ("manakish", ["pate", "zaatar", "huile d'olive"], False),
    "manouch\u00E9": ("manakish", ["pate", "zaatar", "huile d'olive"], False),
    "man'ouch\u00E9": ("manakish", ["pate", "zaatar", "huile d'olive"], False),
    "manouseh": ("manakish", ["pate", "zaatar", "huile d'olive"], False),
    "manoucheh": ("manakish", ["pate", "zaatar", "huile d'olive"], False),

    # === MJADDARA (OLJ) ===
    "mjaddara": ("mjaddara", ["lentilles", "riz", "oignon frit"], False),
    "mujaddara": ("mjaddara", ["lentilles", "riz", "oignon frit"], False),
    "moujaddara": ("mjaddara", ["lentilles", "riz", "oignon frit"], False),
    "mejadra": ("mjaddara", ["lentilles", "riz", "oignon frit"], False),

    # === KOUSSA MAHCHI (OLJ) ===
    "koussa mahchi": ("koussa_mahchi", ["courgettes", "riz", "viande"], False),
    "courgettes farcies": ("koussa_mahchi", ["courgettes", "riz", "viande"], False),
    "koussa mehche": ("koussa_mahchi", ["courgettes", "riz", "viande"], False),

    # === SHISH BARAK (OLJ) ===
    "shish barak": ("shish_barak", ["pate", "viande", "yaourt"], False),
    "ravioles au yaourt": ("shish_barak", ["pate", "viande", "yaourt"], False),
    "mante": ("shish_barak", ["pate", "viande", "yaourt"], False),
    "manti": ("shish_barak", ["pate", "viande", "yaourt"], False),
    "raviolis yaourt": ("shish_barak", ["pate", "viande", "yaourt"], False),

    # === BATABA HARRA (BASE2) ===
    "batata harra": ("batata_harra", ["pomme de terre", "ail", "citron", "piment"], True),
    "pommes de terre epicees": ("batata_harra", ["pomme de terre", "ail", "citron", "piment"], True),
    "pommes de terre \u00E9pic\u00E9es": ("batata_harra", ["pomme de terre", "ail", "citron", "piment"], True),

    # === DESSERTS (BASE2) ===
    "maamoul": ("maamoul", ["semoule", "dattes", "noix", "pistaches"], True),
    "mamoul": ("maamoul", ["semoule", "dattes", "noix", "pistaches"], True),
    "knefeh": ("knefeh", ["cheveux d'ange", "fromage", "sirop"], True),
    "kunafa": ("knefeh", ["cheveux d'ange", "fromage", "sirop"], True),
    "knafeh": ("knefeh", ["cheveux d'ange", "fromage", "sirop"], True),
    "knafe": ("knefeh", ["cheveux d'ange", "fromage", "sirop"], True),
    "baklawa": ("baklawa", ["pate filo", "noix", "pistaches", "sirop"], True),
    "baklava": ("baklawa", ["pate filo", "noix", "pistaches", "sirop"], True),
    "baklawat": ("baklawa", ["pate filo", "noix", "pistaches", "sirop"], True),

    # === HALLOUMI (BASE2) ===
    "halloumi": ("halloumi", ["fromage halloumi"], True),
    "haloumi": ("halloumi", ["fromage halloumi"], True),
    "hellim": ("halloumi", ["fromage halloumi"], True),
    "halloum": ("halloumi", ["fromage halloumi"], True),

    # === MOGHRABIEH (BASE2) ===
    "moghrabieh": ("moghrabieh", ["perles de couscous", "poulet", "pois chiches"], True),
    "moughrabieh": ("moghrabieh", ["perles de couscous", "poulet", "pois chiches"], True),
    "moghrabi": ("moghrabieh", ["perles de couscous", "poulet", "pois chiches"], True),

    # === DAWOOD BASHA (BASE2) ===
    "dawood basha": ("dawood_basha", ["boulette de viande", "tomate", "pignon"], True),
    "dawoud bacha": ("dawood_basha", ["boulette de viande", "tomate", "pignon"], True),

    # === MOULOUKHIYE (OLJ) ===
    "mouloukhiye": ("mouloukhiye", ["corete", "poulet", "riz"], False),
    "mouloukhia": ("mouloukhiye", ["corete", "poulet", "riz"], False),
    "mloukhiye": ("mouloukhiye", ["corete", "poulet", "riz"], False),
    "molokhia": ("mouloukhiye", ["corete", "poulet", "riz"], False),

    # === AUTRES ===
    "sfouf": ("sfouf", ["semoule", "curcuma", "anis"], False),
}



class DishNormalizer:
    """Maps dish variants to canonical forms with Base1/Base2 hints."""

    def __init__(self) -> None:
        self._alias_map: dict[str, tuple[str, tuple[str, ...], bool]] = {}
        self._aliases_by_canonical: dict[str, set[str]] = {}
        self._ingredients_by_canonical: dict[str, set[str]] = {}
        self._base2_only: set[str] = set()
        self._canonical_lookup: dict[str, str] = {}

        for alias, (canonical, ingredients, base2_only) in DISH_ALIASES.items():
            canonical_name = canonical.strip()
            canonical_norm = normalize_text(canonical_name)
            alias_norm = normalize_text(alias)
            if not canonical_norm or not alias_norm:
                continue
            ingredients_norm = [
                normalize_text(item) for item in ingredients if normalize_text(item)
            ]

            self._alias_map[alias_norm] = (
                canonical_name,
                tuple(ingredients_norm),
                base2_only,
            )
            self._aliases_by_canonical.setdefault(canonical_name, set()).add(alias_norm)
            if ingredients_norm:
                self._ingredients_by_canonical.setdefault(canonical_name, set()).update(ingredients_norm)
            if base2_only:
                self._base2_only.add(canonical_name)
            self._canonical_lookup.setdefault(canonical_norm, canonical_name)

        # Ensure canonical name resolves to itself
        for canonical_name in list(self._aliases_by_canonical):
            canonical_norm = normalize_text(canonical_name)
            if canonical_norm and canonical_norm not in self._alias_map:
                self._alias_map[canonical_norm] = (
                    canonical_name,
                    tuple(),
                    canonical_name in self._base2_only,
                )
                self._aliases_by_canonical[canonical_name].add(canonical_norm)
            if canonical_norm and canonical_norm not in self._canonical_lookup:
                self._canonical_lookup[canonical_norm] = canonical_name

        logger.info("Dish normalizer initialized with %d aliases", len(self._alias_map))

    def normalize(self, dish: str | None) -> str | None:
        if not dish:
            return None
        normalized = normalize_text(dish)
        if not normalized:
            return None
        if normalized in self._alias_map:
            return self._alias_map[normalized][0]
        return self._fuzzy_lookup(normalized)

    def ingredients_for(self, dish: str | None) -> list[str]:
        canonical = self.normalize(dish)
        if not canonical:
            return []
        ingredients = self._ingredients_by_canonical.get(canonical, set())
        return sorted(ingredients)

    def is_base2_only(self, dish: str | None) -> bool:
        canonical = self.normalize(dish)
        if not canonical:
            return False
        return canonical in self._base2_only

    def expand(self, dish: str | None) -> list[str]:
        canonical = self.normalize(dish)
        if not canonical:
            return []
        aliases = self._aliases_by_canonical.get(canonical, {canonical})
        return sorted(aliases)

    def expand_list(self, dishes: Iterable[str]) -> list[str]:
        expanded: list[str] = []
        seen: set[str] = set()
        for dish in dishes or []:
            for variant in self.expand(dish):
                if variant not in seen:
                    expanded.append(variant)
                    seen.add(variant)
        return expanded

    def get_all_aliases(self, canonical: str | None) -> list[str]:
        if not canonical:
            return []
        normalized = normalize_text(canonical)
        if not normalized:
            return []
        canonical_name = self._canonical_lookup.get(normalized)
        if canonical_name is None and canonical in self._aliases_by_canonical:
            canonical_name = canonical
        if not canonical_name:
            return []
        aliases = self._aliases_by_canonical.get(canonical_name, set())
        return sorted(aliases)

    def _fuzzy_lookup(self, term: str) -> str | None:
        if len(term) < 4:
            return None
        candidates = difflib.get_close_matches(
            term,
            list(self._alias_map.keys()),
            n=1,
            cutoff=0.84,
        )
        if not candidates:
            return None
        match = candidates[0]
        return self._alias_map.get(match, (None, (), False))[0]


dish_normalizer = DishNormalizer()

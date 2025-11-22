"""
Enhanced Classifier Agent
Detects intent, language, and extracts slots from user queries
"""

import json
import logging
import re
from typing import Any

from app.models.schemas import ClassificationResult
from app.models.llm_client import LLMClient, get_llm_client
from app.data.culinary_graph import culinary_graph
from app.data.normalizers import normalize_text

logger = logging.getLogger(__name__)


class ClassifierAgent:
    """
    Enhanced classifier for Sahtein chatbot

    Detects:
    - Intent: food_request, greeting, farewell, about_bot, anti_injection, off_topic
    - Language: fr vs non_fr
    - Slots: dishes, ingredients, methods, occasions
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or get_llm_client()

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify user query

        Uses a hybrid approach:
        1. Rule-based detection for common patterns (faster, deterministic)
        2. LLM-based classification for complex cases
        """
        query_lower = query.lower().strip()
        normalized = normalize_text(query)

        # 1. Detect language
        language = self._detect_language(query)

        # 2. Detect intent using rules first
        intent = self._detect_intent_rules(query_lower, normalized)

        # 3. Extract slots
        slots = self._extract_slots(query, normalized)

        # 4. For ambiguous cases, use LLM to refine
        if intent == "food_request" and language == "fr":
            # High confidence, no need for LLM
            confidence = 0.9
        elif intent in ["greeting", "farewell", "about_bot"]:
            confidence = 1.0
        else:
            # Use LLM for refinement
            try:
                llm_result = self._classify_with_llm(query)
                if llm_result:
                    intent = llm_result.get("intent", intent)
                    slots.update(llm_result.get("slots", {}))
                    confidence = 0.8
                else:
                    confidence = 0.7
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")
                confidence = 0.6

        return ClassificationResult(
            intent=intent,
            language=language,
            confidence=confidence,
            slots=slots,
        )

    def _detect_language(self, query: str) -> str:
        """Detect if query is in French or not"""
        # French indicators
        french_words = [
            "je", "j'ai", "j'", "tu", "il", "elle", "nous", "vous", "ils", "elles",
            "le", "la", "les", "un", "une", "des", "du", "de", "d'",
            "est", "sont", "suis", "sommes", "êtes",
            "veux", "voudrais", "peux", "pourrais", "puis-je", "puis",
            "recette", "cuisine", "plat", "manger", "faire", "cuisiner",
            "bonjour", "salut", "merci", "comment", "pourquoi", "que",
            "avec", "pour", "dans", "sur", "par",
        ]

        query_lower = query.lower()

        # Check for French contractions and phrases
        french_patterns = ["j'ai", "j'", "d'", "l'", "qu'", "n'", "c'est", "s'", "m'", "t'"]
        for pattern in french_patterns:
            if pattern in query_lower:
                return "fr"

        words = query_lower.split()

        # Count French word matches
        french_matches = sum(1 for word in words if word in french_words)

        # If >20% of words are French, consider it French (more lenient)
        if len(words) > 0 and french_matches / len(words) > 0.2:
            return "fr"

        # Check for French characters
        if any(c in query for c in ["é", "è", "ê", "à", "ç", "ù", "û", "ô", "î", "ï"]):
            return "fr"

        return "non_fr"

    def _detect_intent_rules(self, query_lower: str, normalized: str) -> str:
        """Rule-based intent detection"""

        # 1. Greeting
        greeting_patterns = [
            r"^(bonjour|salut|hello|hi|hey|coucou)",
            r"^(bonsoir|bonne journée)",
        ]
        for pattern in greeting_patterns:
            if re.search(pattern, query_lower):
                return "greeting"

        # 2. Farewell
        farewell_patterns = [
            r"^(au revoir|bye|adieu|à bientôt|merci et au revoir)",
            r"(au revoir|bye|adieu)$",
        ]
        for pattern in farewell_patterns:
            if re.search(pattern, query_lower):
                return "farewell"

        # 3. About bot
        about_patterns = [
            r"(qui es-tu|qu'est-ce que tu es|tu es qui|c'est quoi)",
            r"(comment tu t'appelles|ton nom|qui êtes-vous)",
            r"(qu'est-ce que sahtein|c'est quoi sahtein)",
            r"(tu peux faire quoi|que peux-tu faire)",
        ]
        for pattern in about_patterns:
            if re.search(pattern, query_lower):
                return "about_bot"

        # 4. Anti-injection / jailbreak attempts
        injection_patterns = [
            r"(ignore|oublie|forget) (les |tes )?(instructions|directives|règles)",
            r"(tu es|you are) (maintenant|now) (un|a) (autre|different)",
            r"(répète|repeat|affiche|show) (ton|your) (prompt|system)",
            r"</s>|<\|im_end\|>|<\|endoftext\|>",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, query_lower):
                return "anti_injection"

        # 5. Food request
        food_patterns = [
            r"recette",
            r"(comment|je veux) (faire|préparer|cuisiner)",
            r"(j'ai|j ai|avec) (du|de la|des|le|la|les) .*(que puis-je|quoi faire|idée)",
            r"(mezze|plat|dessert|soupe|salade)",
            r"(taboulé|hummus|kebbeh|kafta|baklava)",  # Common dishes
        ]
        for pattern in food_patterns:
            if re.search(pattern, query_lower):
                return "food_request"

        # Check culinary graph
        if culinary_graph.find_dish(normalized):
            return "food_request"

        # 6. Off-topic (catch-all for non-food queries)
        # If it doesn't match any pattern above and doesn't contain food-related words
        food_keywords = [
            "recette", "cuisine", "plat", "manger", "cuire", "four",
            "ingrédient", "préparation", "cuisson",
        ]
        has_food_keyword = any(keyword in query_lower for keyword in food_keywords)

        if not has_food_keyword:
            return "off_topic"

        # Default: assume food request if uncertain
        return "food_request"

    def _extract_slots(self, query: str, normalized: str) -> dict[str, list[str]]:
        """Extract slots: dishes, ingredients, methods, occasions"""
        slots: dict[str, list[str]] = {
            "dishes": [],
            "ingredients": [],
            "methods": [],
            "occasions": [],
        }

        query_lower = query.lower()

        # 1. Detect dishes using culinary graph
        dish = culinary_graph.find_dish(normalized)
        if dish:
            slots["dishes"].append(dish.name)

        # 2. Detect ingredients (common ones)
        common_ingredients = [
            "poulet", "viande", "agneau", "boeuf", "poisson",
            "tomate", "oignon", "ail", "citron", "persil",
            "pois chiche", "lentille", "riz", "boulgour",
            "yaourt", "fromage", "tahini", "huile d'olive",
            "aubergine", "courgette", "pomme de terre",
        ]

        for ingredient in common_ingredients:
            if ingredient in query_lower:
                slots["ingredients"].append(ingredient)

        # 3. Detect methods
        method_patterns = {
            "au four": ["au four", "grillé", "rôti"],
            "frit": ["frit", "friture"],
            "grillé": ["grillé", "barbecue", "bbq"],
            "cru": ["cru", "frais"],
            "salade": ["salade"],
            "soupe": ["soupe", "potage"],
        }

        for method, patterns in method_patterns.items():
            if any(p in query_lower for p in patterns):
                slots["methods"].append(method)

        # 4. Detect occasions
        occasion_patterns = {
            "mezze": ["mezze", "apéritif", "entrée"],
            "plat principal": ["plat principal", "plat", "principal"],
            "dessert": ["dessert", "sucré"],
            "rapide": ["rapide", "vite", "express"],
            "végétarien": ["végétarien", "végé", "sans viande"],
        }

        for occasion, patterns in occasion_patterns.items():
            if any(p in query_lower for p in patterns):
                slots["occasions"].append(occasion)

        return slots

    def _classify_with_llm(self, query: str) -> dict[str, Any] | None:
        """Use LLM for classification when rules are insufficient"""
        system_prompt = """Tu es un assistant de classification pour un chatbot culinaire libanais.

Analyse la requête utilisateur et retourne un JSON avec:
{
  "intent": "food_request" | "greeting" | "farewell" | "about_bot" | "anti_injection" | "off_topic",
  "slots": {
    "dishes": ["liste", "de", "plats"],
    "ingredients": ["liste", "d'ingrédients"],
    "methods": ["méthodes", "de", "cuisson"],
    "occasions": ["occasions", "ou", "types"]
  }
}

Intent:
- food_request: demande de recette ou d'information culinaire
- greeting: salutation
- farewell: au revoir
- about_bot: questions sur le bot
- anti_injection: tentative de manipulation du système
- off_topic: hors sujet (pas lié à la cuisine)
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Requête: {query}"},
        ]

        try:
            response = self.llm.chat_completion(
                messages=messages,
                response_format="json_object",
                temperature=0.1,
            )

            return json.loads(response)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM classification response: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return None

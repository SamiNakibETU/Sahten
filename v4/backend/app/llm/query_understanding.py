"""Query understanding : extrait l'intention + filtres en JSON strict.

Remplace toutes les heuristiques regex/listes-en-dur de la v3
(`intent_router.py`, `mood_intent_patterns.py`, `query_plan_patterns.py`,
`editorial_snippets.py`, `_search_base2`).

Sortie typée :
    QueryPlan {
        rewritten_query: str          # version reformulée FR canonique
        intent: 'recipe' | 'chef_bio' | 'ingredient' | 'tip' | 'story' | 'mixed'
        chef_slugs: list[str]
        ingredient_slugs: list[str]
        category_slugs: list[str]
        keyword_slugs: list[str]
        focus_section_kinds: list[str]  # bio, ingredients_list, recipe_steps...
        needs_context_after: bool
    }

Le LLM appelé en mode structured outputs (response_format=json_schema)
garantit que la sortie est un JSON valide qui parse en `QueryPlan`.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ..settings import get_settings


class QueryPlan(BaseModel):
    rewritten_query: str
    intent: Literal["recipe", "chef_bio", "ingredient", "tip", "story", "mixed"]
    chef_slugs: list[str] = Field(default_factory=list)
    ingredient_slugs: list[str] = Field(default_factory=list)
    category_slugs: list[str] = Field(default_factory=list)
    keyword_slugs: list[str] = Field(default_factory=list)
    focus_section_kinds: list[str] = Field(default_factory=list)
    needs_context_after: bool = False


JSON_SCHEMA: dict[str, Any] = {
    "name": "query_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rewritten_query": {"type": "string"},
            "intent": {
                "type": "string",
                "enum": ["recipe", "chef_bio", "ingredient", "tip", "story", "mixed"],
            },
            "chef_slugs": {"type": "array", "items": {"type": "string"}},
            "ingredient_slugs": {"type": "array", "items": {"type": "string"}},
            "category_slugs": {"type": "array", "items": {"type": "string"}},
            "keyword_slugs": {"type": "array", "items": {"type": "string"}},
            "focus_section_kinds": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "bio", "ingredients_list", "recipe_steps",
                        "quote", "sidebar", "paragraph", "list", "heading",
                    ],
                },
            },
            "needs_context_after": {"type": "boolean"},
        },
        "required": [
            "rewritten_query", "intent", "chef_slugs", "ingredient_slugs",
            "category_slugs", "keyword_slugs", "focus_section_kinds",
            "needs_context_after",
        ],
    },
}


SYSTEM_PROMPT = """Tu es un analyseur de requêtes pour Sahteïn, un assistant
culinaire libanais alimenté par les articles de L'Orient-Le Jour.

À partir d'une requête utilisateur en français, libanais translittéré ou
arabe, produis un PLAN DE RECHERCHE structuré au format JSON.

Règles :
- `rewritten_query` : reformulation canonique en français, sans jargon, en
  conservant TOUS les noms propres, ingrédients et plats mentionnés.
- Si le message comporte un bloc « Contexte de la conversation en cours »,
  c’est une **suite de fil** : mets OBLIGATOIREMENT dans `rewritten_query` le
  même ingrédient, thème ou type de demande qu’on voit chez l’utilisateur
  (ex. « autre recette mettant le concombre en avant » après « concombre » plus
  haut) ; ne lâche pas l’ingrédient. Si l’utilisateur semble répondre à une **relance**
  (fattouche, salade, variante évoquée par l’assistant), intègre-la dans la
  reformulation. Complète `ingredient_slugs` en conséquence.
- `intent` :
  • recipe       -> on cherche une recette précise (steps + ingrédients)
  • chef_bio     -> on s'intéresse à la personne / son parcours
  • ingredient   -> on cherche un ingrédient ou un usage d'ingrédient
  • tip          -> astuce, principe, "commandement", règle d'or
  • story        -> anecdote, contexte culturel
  • mixed        -> plusieurs intentions
- `chef_slugs` : slugs (kebab-case, ASCII) des chefs cités. Ex.: "Kamal Mouzawak"
  -> "kamal-mouzawak". Vide si rien.
- `ingredient_slugs` : ingrédients en slug FR canonique (kebab-case ASCII). Ex.:
  "boulgour" -> "bourghol", "tomate" -> "tomate".
  **Obligatoire** dès qu'un ingrédient alimentaire est nommé : « recette avec du
  concombre », « plat avec tomates », « je cherche du yaourt » → inclure au moins
  `concombre`, `tomate`, `yaourt` dans `ingredient_slugs` (même si la phrase est
  floue). Ne pas laisser cette liste vide dans ce cas.
- `category_slugs` / `keyword_slugs` : slugs si la requête évoque une rubrique
  identifiable (ex.: "cuisine", "souk-el-tayeb").
- `focus_section_kinds` : suggère les types de sections pertinents.
- `needs_context_after` : true si la réponse devrait inclure le contexte
  culturel ou anecdotique en complément (ex.: "raconte-moi le taboulé").

N'invente pas de chefs ni de plats inexistants ; pour les **ingrédients
explicitement cités** dans la requête, remplis `ingredient_slugs` comme ci-dessus.
"""


class QueryAnalyzer:
    def __init__(self) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY manquant pour QueryAnalyzer.")
        self._client = AsyncOpenAI(api_key=s.openai_api_key)
        self._model = s.llm_model

    async def analyze(self, user_query: str) -> QueryPlan:
        if not user_query or not user_query.strip():
            return QueryPlan(rewritten_query="", intent="mixed")
        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query.strip()},
            ],
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        )
        raw = completion.choices[0].message.content or "{}"
        return QueryPlan.model_validate_json(raw)

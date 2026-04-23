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

Avec un historique de conversation, l’analyseur reçoit HISTORIQUE + QUESTION ACTUELLE
et infère anaphores, fil d’ingrédient et relances (pas d’heuristique regex côté pipeline).
"""

from __future__ import annotations

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

On te fournit soit un message isolé, soit le couple **HISTORIQUE** + **QUESTION ACTUELLE**.
Lis tout l’historique (tours « Utilisateur » et « Assistant ») comme un humain
le ferait : mêmes fils, anaphores (« une autre recette » = le même thème/ingrédient
qu’avant), réponses courtes (oui, non, celle d’en haut) à des **relances** de
l’assistant (ex. fattouche, tarator), ou rejet d’une suggestion pour en demander
une autre. Ne te fie à aucun mot-clé : déduis l’intention du sens global.

Règles de sortie JSON :
- `rewritten_query` : reformulation **autonome** en français, exploitable telle
  quelle pour la recherche (BM25 + sémantique) : y injecte le ou les **ingrédients,
  plats, thèmes** implicites issus de l’historique quand la question actuelle est
  elliptique ou de continuation (ex. « autre recette au concombre » si c’était
  le fil de la conversation, même si le mot concombre n’apparaît que plus haut).
- `ingredient_slugs` : mêmes ingrédients, en slugs kebab-case ASCII, que tu as
  intégrés dans la logique (pas seulement mots de la dernière phrase). Si le fil
  est « concombre » d’après l’historique, inclut `concombre`. Ne laisse pas la
  liste vide quand l’histoire de conversation porte clairement sur un ou des
  ingrédients identifiables.
- `intent` :
  • recipe       -> on cherche une recette précise (steps + ingrédients)
  • chef_bio     -> on s'intéresse à la personne / son parcours
  • ingredient   -> on cherche un ingrédient ou un usage d'ingrédient
  • tip          -> astuce, principe, "commandement", règle d'or
  • story        -> anecdote, contexte culturel
  • mixed        -> plusieurs intentions
- `chef_slugs` : slugs (kebab-case, ASCII) des chefs cités. Ex.: "Kamal Mouzawak"
  -> "kamal-mouzawak". Vide si rien.
- `category_slugs` / `keyword_slugs` : slugs si la requête évoque une rubrique
  identifiable (ex.: "cuisine", "souk-el-tayeb").
- `focus_section_kinds` : types de sections pertinents.
- `needs_context_after` : true si un complément anecdotique / culturel est attendu.

N'invente pas de noms de chefs, plats ou ingrédients jamais évoqués ; mais tu peux
**résoudre** ce qui est implicite dans l’échange (fil suivi) sans qu’on le répète
à chaque message.
- Dernière relance de l’assistant : si l’Assistant vient de proposer
  explicitement un plat ou une piste (ex. « le fattouche », « une salade
  traditionnelle ») et que la QUESTION ACTUELLE est une suite brève
  (« une autre », « oui », etc.), mets le nom de ce plat (ou la formulation
  la plus proche) dans `rewritten_query`, en plus de l’ingrédient / thème
  (ex. « fattouche salade concombre » ou « recette fattouche concombre ») pour
  que l’index trouve l’article — ne renvoie pas seulement « autre recette
  concombre » en oubliant ce que toi l’assistant as suggéré.
"""


class QueryAnalyzer:
    def __init__(self) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY manquant pour QueryAnalyzer.")
        self._client = AsyncOpenAI(api_key=s.openai_api_key)
        self._model = s.llm_model

    async def analyze(
        self,
        user_query: str,
        *,
        conversation_history: str | None = None,
        max_history_chars: int = 14_000,
    ) -> QueryPlan:
        if not user_query or not user_query.strip():
            return QueryPlan(rewritten_query="", intent="mixed")
        uq = user_query.strip()
        h = (conversation_history or "").strip()
        if h and len(h) > max_history_chars:
            h = "…\n" + h[-max_history_chars:]
        if h:
            user_content = (
                "HISTORIQUE DE CONVERSATION (tours précédents, ordre chronologique ; "
                "l’utilisateur a déjà parlé avec l’assistant — résous les anaphores, "
                "suis le fil, note les relances proposées par l’assistant) :\n"
                f"{h}\n\n"
                "QUESTION ACTUELLE (à interpréter **à la lumière** de l’historique) :\n"
                f"{uq}"
            )
        else:
            user_content = uq
        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        )
        raw = completion.choices[0].message.content or "{}"
        return QueryPlan.model_validate_json(raw)

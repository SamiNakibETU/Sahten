"""Deuxième « cerveau » : lecture du fil de session pour la retrieval + la réponse.

Complète le QueryAnalyzer (plan structuré) sans listes d’ingrédients en dur : un
appel LLM séparé avec sortie JSON contrainte, uniquement si un historique existe.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ..settings import get_settings


class SessionFocus(BaseModel):
    """Résumé intentionnel du fil + option d’enrichir la requête documentaire."""

    search_boost_phrase: str = Field(
        default="",
        description="Phrase courte FR à ajouter à la requête de retrieval (synonymes, ingrédient implicite). Vide si inutile.",
    )
    thread_summary: str = Field(
        default="",
        description="Une ligne : fil suivi (thème, ingrédient, relance en cours).",
    )
    user_wants_different_article: bool = Field(
        default=False,
        description="True si l’utilisateur demande une variante / autre fiche que la précédente.",
    )
    suggest_broaden_corpus_search: bool = Field(
        default=False,
        description="True si les tags SQL peuvent être trop restrictifs et qu’il faut élargir au texte plein.",
    )


JSON_SCHEMA: dict[str, Any] = {
    "name": "session_focus",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "search_boost_phrase": {"type": "string"},
            "thread_summary": {"type": "string"},
            "user_wants_different_article": {"type": "boolean"},
            "suggest_broaden_corpus_search": {"type": "boolean"},
        },
        "required": [
            "search_boost_phrase",
            "thread_summary",
            "user_wants_different_article",
            "suggest_broaden_corpus_search",
        ],
    },
}


SYSTEM = """Tu résume le **fil** d’une conversation culinaire (Sahteïn / L’Orient-Le Jour).

Entrée : HISTORIQUE (Utilisateur / Assistant) + QUESTION ACTUELLE.
Sors un JSON respectant le schéma.

Règles :
- `search_boost_phrase` : seulement si ça améliore la **recherche dans des articles**
  (ingrédient ou thème implicite absent de la dernière phrase). Sinon chaîne vide.
  Pas de liste de mots séparés par des virgules ; une mini-phrase naturelle.
  **Priorité** : relis le **dernier** tour « Assistant : ». Si l’utilisateur
  enchaîne avec « une autre », « oui », « celle d’en haut » juste **après** que
  l’assistant a **nommé** un plat (ex. « fattouche », « taboulé ») en relance,
  mets **obligatoirement** ce nom (et l’ingrédient de fil) dans
  `search_boost_phrase` / `thread_summary` — sinon l’équipe aval ne pourra pas
  retrouver l’article. Ne pas l’ignorer au profit d’une requête trop générique.
- `thread_summary` : utile pour le modèle qui **répond** (coherence). Max ~200 car.
- `user_wants_different_article` : vrai si l’utilisateur veut une autre fiche / variante
  / « encore » / « plutôt autre chose » / réponse négative à une suggestion.
- `suggest_broaden_corpus_search` : mets true si l’on peut craindre que des recettes
  mentionnent un ingrédient dans le texte **sans** être taguées en base (corpus
  inhomogène) — l’orchestrateur élargira alors la recherche. Pas d’invention
  d’ingrédients : seulement ce que l’échange indique.
- Après **plusieurs** propositions de recettes (l’assistant a déjà mis des titres
  en avant), un message du type « encore une autre » doit **poursuivre** le fil
  (même ingrédient, angle salade / entrée / relance restante comme le fattouche)
  plutôt que réinitialiser : indique dans `thread_summary` qu’il s’agit d’un
  **tour N** et qu’il faut éviter de reproposer le **même type** dominant (ex.
  enchaîner encore une sauce) ou un plat **hors piste** par rapport aux relances.
- **Objection / correction** (« il n’y a pas de concombre dedans », « ce n’est pas
  ce que je cherchais », « la recette précédente n’a pas X ») : l’utilisateur
  conteste la pertinence de la **dernière fiche** mais **reste dans le fil**
  (même ingrédient ou même relance). Mets `user_wants_different_article` à **true**,
  remets **obligatoirement** dans `search_boost_phrase` l’ingrédient du fil (ex.
  concombre) **et** si l’assistant a nommé un plat en relance (fattouche, taboulé),
  inclus ce nom pour la recherche documentaire. Mets `suggest_broaden_corpus_search`
  à **true**. Résume dans `thread_summary` que l’utilisateur exige une fiche où
  l’ingrédient est **explicite** dans le texte (pas seulement accessoire).
"""


class SessionFocusAnalyzer:
    def __init__(self) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY manquant pour SessionFocusAnalyzer.")
        self._client = AsyncOpenAI(api_key=s.openai_api_key)
        self._model = s.llm_model

    async def infer(
        self,
        user_query: str,
        conversation_history: str,
        *,
        max_history_chars: int = 14_000,
    ) -> SessionFocus:
        h = (conversation_history or "").strip()
        if len(h) > max_history_chars:
            h = "…\n" + h[-max_history_chars:]
        uq = (user_query or "").strip()
        user_content = (
            f"{h}\n\n---\nQUESTION ACTUELLE :\n{uq}"
        )
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
            )
            raw = completion.choices[0].message.content or "{}"
            return SessionFocus.model_validate_json(raw)
        except Exception:  # noqa: BLE001
            return SessionFocus()


__all__ = ["SessionFocus", "SessionFocusAnalyzer"]

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
- `thread_summary` : utile pour le modèle qui **répond** (coherence). Max ~200 car.
- `user_wants_different_article` : vrai si l’utilisateur veut une autre fiche / variante
  / « encore » / « plutôt autre chose » / réponse négative à une suggestion.
- `suggest_broaden_corpus_search` : mets true si l’on peut craindre que des recettes
  mentionnent un ingrédient dans le texte **sans** être taguées en base (corpus
  inhomogène) — l’orchestrateur élargira alors la recherche. Pas d’invention
  d’ingrédients : seulement ce que l’échange indique.
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

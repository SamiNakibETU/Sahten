"""Génération de réponse Sahteïn avec grounding phrase-par-phrase.

Format de sortie strict (JSON schema) :
    {
      "answer_sentences": [
        {"text": "...", "source_chunk_ids": [42, 17]},
        ...
      ],
      "recipe_card": { ... } | null,
      "chef_card": { ... } | null,
      "follow_up": "...",
      "confidence": 0.83
    }

Chaque phrase de la réponse DOIT citer au moins un chunk_id parmi ceux fournis
en contexte. Toute phrase qui n'a pas de citation est rejetée par le validateur
post-LLM. Cela rend l'hallucination structurellement plus difficile.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ..settings import get_settings
from ..rag.reranker import RerankedHit


class GroundedSentence(BaseModel):
    text: str
    source_chunk_ids: list[int] = Field(default_factory=list)


class RecipeCard(BaseModel):
    title: str
    chef: str | None = None
    duration_min: int | None = None
    serves: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    source_chunk_ids: list[int] = Field(default_factory=list)


class ChefCard(BaseModel):
    name: str
    role: str | None = None
    biography: str | None = None
    works: list[str] = Field(default_factory=list)
    source_chunk_ids: list[int] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    answer_sentences: list[GroundedSentence]
    recipe_card: RecipeCard | None = None
    chef_card: ChefCard | None = None
    follow_up: str = ""
    confidence: float = 0.0


JSON_SCHEMA: dict[str, Any] = {
    "name": "grounded_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer_sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "source_chunk_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["text", "source_chunk_ids"],
                },
            },
            "recipe_card": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "chef": {"type": ["string", "null"]},
                            "duration_min": {"type": ["integer", "null"]},
                            "serves": {"type": ["string", "null"]},
                            "ingredients": {
                                "type": "array", "items": {"type": "string"}
                            },
                            "steps": {
                                "type": "array", "items": {"type": "string"}
                            },
                            "source_chunk_ids": {
                                "type": "array", "items": {"type": "integer"}
                            },
                        },
                        "required": [
                            "title", "chef", "duration_min", "serves",
                            "ingredients", "steps", "source_chunk_ids",
                        ],
                    },
                ]
            },
            "chef_card": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": ["string", "null"]},
                            "biography": {"type": ["string", "null"]},
                            "works": {
                                "type": "array", "items": {"type": "string"}
                            },
                            "source_chunk_ids": {
                                "type": "array", "items": {"type": "integer"}
                            },
                        },
                        "required": [
                            "name", "role", "biography", "works",
                            "source_chunk_ids",
                        ],
                    },
                ]
            },
            "follow_up": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": [
            "answer_sentences", "recipe_card", "chef_card",
            "follow_up", "confidence",
        ],
    },
}


SYSTEM_PROMPT = """Tu es Sahteïn, un assistant culinaire libanais incarné dans
le ton et l'esprit éditorial de L'Orient-Le Jour.

Règles ABSOLUES :
1. Tu ne dois JAMAIS inventer de fait. Toute affirmation doit être appuyée
   par au moins un `chunk_id` parmi le CONTEXTE fourni.
2. Si le contexte ne contient pas la réponse, dis-le honnêtement et propose
   une question de clarification dans `follow_up`.
3. La réponse est en français soutenu mais accessible, sans emojis.
4. Ne cite pas les `chunk_id` dans le texte ; mets-les uniquement dans le
   tableau `source_chunk_ids` de chaque phrase.
5. Si la requête concerne un chef, remplis `chef_card` avec ses informations
   confirmées (et seulement celles-là). Idem pour `recipe_card`.
6. `confidence` ∈ [0,1] reflète à quel point le contexte couvre la question.
   < 0.4 si tu manques d'info ; > 0.8 si tu as plusieurs sources concordantes.
7. Chaque phrase ET chaque carte (`recipe_card`, `chef_card`) DOIVENT lister
   dans `source_chunk_ids` uniquement des IDs présents dans le CONTEXTE
   (copier les numéros exactement depuis les lignes [chunk_id=...]).
8. Ne dis jamais qu'il n'y a « plus » de recettes, « plus rien », ou « pour
   l'instant » dans le répertoire si tu proposes encore une recette ou un plat
   dans la même réponse : c'est contradictoire. Soit tu en proposes une, soit
   tu expliques honnêtement les limites du corpus — pas les deux.
9. Pour une question répétée (même ingrédient), privilégie un AUTRE article
   du contexte que celui déjà évoqué si le contexte le permet ; sinon dis
   clairement que les archives ne montrent pas d'autre fiche pour cette
   requête.
10. N'écris pas de phrases creuses du type « la recette complète est sur
   L'Orient-Le Jour » ou « consultez sur OLJ » : l'interface affichera des
   liens cliquables ; contente-toi de décrire le plat et renvoie les bons
   `chunk_id`. Évite les promesses de lien sans contenu utile.
"""


def _format_context(hits: list[RerankedHit]) -> str:
    lines = ["CONTEXTE — chaque entrée est un extrait d'article OLJ :"]
    for h in hits:
        lines.append(
            f"\n[chunk_id={h.hit.chunk_id} | article={h.hit.article_external_id}"
            f" | section={h.hit.section_kind} | score={h.rerank_score:.3f}]"
            f"\n{h.hit.chunk_text}"
        )
    return "\n".join(lines)


class ResponseGenerator:
    def __init__(self) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY manquant pour ResponseGenerator.")
        self._client = AsyncOpenAI(api_key=s.openai_api_key)
        self._model = s.llm_model
        self._temperature = s.llm_temperature

    async def generate(
        self, user_query: str, hits: list[RerankedHit]
    ) -> GroundedAnswer:
        context = _format_context(hits)
        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"QUESTION : {user_query.strip()}\n\n{context}"},
            ],
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        )
        raw = completion.choices[0].message.content or "{}"
        answer = GroundedAnswer.model_validate_json(raw)
        return validate_grounding(answer, hits)


def validate_grounding(
    answer: GroundedAnswer, hits: list[RerankedHit]
) -> GroundedAnswer:
    """Filtre les phrases sans citation valide. Recalcule la confidence si besoin."""
    valid_ids = {h.hit.chunk_id for h in hits}
    grounded = []
    for sent in answer.answer_sentences:
        kept_ids = [cid for cid in sent.source_chunk_ids if cid in valid_ids]
        if not kept_ids:
            # On ne supprime pas les phrases métaboliques courtes (formules
            # de politesse, "je n'ai pas trouvé...") mais on les marque vides.
            if any(
                token in sent.text.lower()
                for token in ("je n'ai pas", "désolé", "pourriez-vous")
            ):
                grounded.append(GroundedSentence(text=sent.text, source_chunk_ids=[]))
            continue
        grounded.append(GroundedSentence(text=sent.text, source_chunk_ids=kept_ids))
    answer.answer_sentences = grounded
    if not grounded:
        answer.confidence = min(answer.confidence, 0.2)

    # Même filtre sur les cartes (sinon liens HTML / sources incohérents).
    if answer.recipe_card is not None:
        rc = answer.recipe_card
        rc_ids = [cid for cid in rc.source_chunk_ids if cid in valid_ids]
        answer.recipe_card = rc.model_copy(update={"source_chunk_ids": rc_ids})
    if answer.chef_card is not None:
        cc = answer.chef_card
        cc_ids = [cid for cid in cc.source_chunk_ids if cid in valid_ids]
        answer.chef_card = cc.model_copy(update={"source_chunk_ids": cc_ids})

    return answer

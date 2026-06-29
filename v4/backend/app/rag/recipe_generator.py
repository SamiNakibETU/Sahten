"""Génération LLM de recette en DERNIER RECOURS.

Quand un plat est demandé mais absent à la fois de l'OLJ (base 1) et de
``Data_base_2.json`` (base 2), on demande à GPT-4.1 une recette concise et
fiable, on l'affiche **clairement étiquetée comme générée (hors carnets OLJ)**,
et on la met en cache en **Postgres** (`generated_recipes`) — le système de
fichiers du conteneur Railway étant éphémère.

Garde-fous :
- Le modèle renvoie ``can_generate=false`` si le nom n'est pas un vrai plat
  (mot inventé, ingrédient seul, non-sens) → on n'affiche rien de faux.
- Jamais d'attribution à un chef ni à L'Orient-Le Jour.
"""

from __future__ import annotations

import unicodedata
from typing import Any

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..cost_tracker import get_request_cost
from ..db.models import GeneratedRecipe
from ..settings import get_settings

log = structlog.get_logger(__name__)

GENERATION_MODEL = "gpt-4.1"


def normalize_dish(name: str) -> str:
    t = unicodedata.normalize("NFKD", (name or "").lower()).replace("’", "'")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return " ".join(t.replace("'", " ").split())


class LLMRecipe(BaseModel):
    can_generate: bool
    dish_name: str = ""
    cuisine: str = ""
    serves: str = ""
    prep: str = ""
    cook: str = ""
    difficulty: str = ""
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    note: str = ""


JSON_SCHEMA: dict[str, Any] = {
    "name": "generated_recipe",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "can_generate": {"type": "boolean"},
            "dish_name": {"type": "string"},
            "cuisine": {"type": "string"},
            "serves": {"type": "string"},
            "prep": {"type": "string"},
            "cook": {"type": "string"},
            "difficulty": {"type": "string"},
            "ingredients": {"type": "array", "items": {"type": "string"}},
            "steps": {"type": "array", "items": {"type": "string"}},
            "note": {"type": "string"},
        },
        "required": [
            "can_generate", "dish_name", "cuisine", "serves", "prep", "cook",
            "difficulty", "ingredients", "steps", "note",
        ],
    },
}


SYSTEM_PROMPT = """Tu es un chef expert en cuisine libanaise et levantine.
On te donne le NOM d'un plat demandé par un utilisateur. Tu dois décider s'il
s'agit d'un vrai plat cuisinable, et si oui produire une recette CONCISE et FIABLE.

Règles :
- Si le nom n'est PAS un vrai plat (mot inventé, simple ingrédient isolé,
  non-sens, requête vague) → `can_generate` = false et laisse les autres champs vides.
- Si c'est un vrai plat (idéalement libanais/levantin/moyen-oriental, mais tout
  plat reconnu est accepté) → `can_generate` = true et remplis :
  • `dish_name` : nom propre et lisible du plat (en français quand l'usage existe).
  • `cuisine` : ex. "libanaise", "levantine", "moyen-orientale", "autre".
  • `serves` (ex. "4 personnes"), `prep` (ex. "20 min"), `cook` (ex. "30 min"),
    `difficulty` ("facile" / "moyenne" / "difficile").
  • `ingredients` : 6 à 14 lignes courtes avec quantités (ex. "300 g de semoule fine").
  • `steps` : 4 à 8 étapes courtes, claires, dans l'ordre.
  • `note` : une phrase d'astuce ou de contexte (facultatif, sinon vide).
- N'invente JAMAIS de chef, de personne, ni d'attribution à un journal ou à
  L'Orient-Le Jour. Reste sur la recette elle-même.
- Reste factuel et raisonnable sur les quantités et les temps.
"""


class RecipeGenerator:
    def __init__(self) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY manquant pour RecipeGenerator.")
        self._client = AsyncOpenAI(api_key=s.openai_api_key, timeout=45, max_retries=1)

    async def generate(self, dish_name: str) -> LLMRecipe | None:
        name = (dish_name or "").strip()
        if not name:
            return None
        completion = await self._client.chat.completions.create(
            model=GENERATION_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Plat demandé : {name}"},
            ],
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        )
        acc = get_request_cost()
        if acc is not None:
            acc.record_openai_chat_usage(
                "recipe_generator", model=GENERATION_MODEL, usage=completion.usage
            )
        raw = completion.choices[0].message.content or "{}"
        rec = LLMRecipe.model_validate_json(raw)
        if not rec.can_generate or not (rec.ingredients and rec.steps):
            return None
        return rec


_GENERATOR: RecipeGenerator | None = None


def _get_generator() -> RecipeGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = RecipeGenerator()
    return _GENERATOR


async def get_or_generate_recipe(
    session: AsyncSession, dish_name: str
) -> dict[str, Any] | None:
    """Renvoie le payload de recette (dict) depuis le cache DB, sinon le génère
    via GPT-4.1 puis le persiste. None si le plat n'est pas générable."""
    norm = normalize_dish(dish_name)
    if len(norm) < 3:
        return None

    cached = (
        await session.execute(
            select(GeneratedRecipe).where(GeneratedRecipe.dish_norm == norm)
        )
    ).scalar_one_or_none()
    if cached is not None:
        payload = dict(cached.payload or {})
        payload["_cached"] = True
        return payload

    try:
        rec = await _get_generator().generate(dish_name)
    except Exception as exc:
        log.warning("recipe_generator.generate_failed", dish=dish_name[:60], error=str(exc))
        return None
    if rec is None:
        return None

    payload: dict[str, Any] = {
        "name": rec.dish_name or dish_name.strip(),
        "cuisine": rec.cuisine,
        "serves": rec.serves,
        "prep": rec.prep,
        "cook": rec.cook,
        "difficulty": rec.difficulty,
        "ingredients": rec.ingredients,
        "steps": rec.steps,
        "note": rec.note,
    }
    row = GeneratedRecipe(
        dish_norm=norm,
        name=payload["name"],
        payload=payload,
        model=GENERATION_MODEL,
    )
    session.add(row)
    try:
        await session.commit()
    except IntegrityError:
        # Course concurrente : un autre worker a déjà inséré -> on relit.
        await session.rollback()
        cached = (
            await session.execute(
                select(GeneratedRecipe).where(GeneratedRecipe.dish_norm == norm)
            )
        ).scalar_one_or_none()
        if cached is not None:
            payload = dict(cached.payload or {})
    payload["_cached"] = False
    log.info("recipe_generator.generated", dish=payload.get("name"), norm=norm)
    return payload

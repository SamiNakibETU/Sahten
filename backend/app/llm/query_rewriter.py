"""
Réécriture courte de la requête pour le retrieval (Phase 2) — un appel LLM JSON.
Cible : intents mood / vague où le texte utilisateur n’aligne pas les titres du corpus.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..core.config import get_settings
from ..core.llm_routing import async_openai_client_for_model, provider_credentials_ok
from ..schemas.query_analysis import QueryAnalysis

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = """Tu aides un moteur de recherche de recettes (corpus L'Orient-Le Jour, cuisine libanaise et proche).
À partir de la requête utilisateur et de l’analyse (intent, tags), produis une phrase de recherche COURTE en français
qui maximise le rappel lexical : noms de plats libanais probables, ingrédients, type de plat, saison si pertinent.
Si mood_tags inclut « liban » sans mention de dessert dans la requête : oriente la phrase vers plats du quotidien libanais
(mezze, grillades, ragoûts, riz, poêlées, daoud bacha, chawarma, moussaka, kafta, etc.), pas vers pâtisseries.
Ne pas inventer de recette absente du monde réel ; rester générique (ex. « soupe réconfortante hiver », « mezze froid citron »).
Réponds UNIQUEMENT en JSON : {"retrieval_phrase": "...", "keywords": ["mot1", "mot2"]}
keywords : 3 à 8 termes utiles pour TF-IDF."""


async def maybe_rewrite_for_retrieval(
    raw_query: str,
    analysis: QueryAnalysis,
    *,
    conversation_context: Optional[str],
    model: str,
) -> Optional[str]:
    settings = get_settings()
    if not settings.enable_retrieval_query_rewrite:
        return None
    if not provider_credentials_ok(model):
        return None

    intent = analysis.intent
    if intent not in (
        "recipe_by_mood",
        "recipe_by_category",
        "recipe_by_ingredient",
        "multi_recipe",
        "recipe_by_diet",
        "menu_composition",
    ):
        return None

    user_block = f"Requête: {raw_query}\nIntent: {intent}\n"
    if analysis.mood_tags:
        user_block += f"mood_tags: {analysis.mood_tags}\n"
    if analysis.category:
        user_block += f"category: {analysis.category}\n"
    if analysis.ingredients:
        user_block += f"ingredients: {analysis.ingredients}\n"
    if analysis.dietary_restrictions:
        user_block += f"dietary: {analysis.dietary_restrictions}\n"
    if conversation_context:
        user_block += f"\nContexte conversation:\n{conversation_context[:800]}\n"

    try:
        client = async_openai_client_for_model(model)
        resp = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": user_block},
            ],
            temperature=0,
            max_tokens=120,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        phrase = (data.get("retrieval_phrase") or "").strip()
        kws = data.get("keywords") or []
        if isinstance(kws, list):
            extra = " ".join(str(x) for x in kws if x)
        else:
            extra = ""
        merged = " ".join(p for p in (phrase, extra) if p).strip()
        if len(merged) < 8:
            return None
        logger.info("Retrieval rewrite: %s", merged[:120])
        return merged
    except Exception as e:
        logger.warning("Retrieval rewrite failed: %s", e)
        return None

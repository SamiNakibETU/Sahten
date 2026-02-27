"""
Backfill OLJ Canonical Dataset
==============================

Goal: produce a *canonical* dataset usable by a durable RAG retriever.

Input:
  - v2/data_base_OLJ_enriched.json

Output:
  - v2/data/olj_canonical.json

What this script does:
  - Ensures critical fields are not empty (title, category, difficulty, search_text)
  - Normalizes taxonomy (category/difficulty)
  - Uses a single LLM call (JSON mode) to backfill missing enrichment fields *offline*
  - Deduplicates by URL

Notes:
  - This script is intended to be run manually/offline.
  - It is NOT executed by the API at runtime.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError
from unidecode import unidecode


def _slug_to_title(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"-\d+$", "", slug)  # drop trailing numeric ids
    slug = slug.replace("-", " ").strip()
    return slug[:1].upper() + slug[1:] if slug else ""


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_token(s: str) -> str:
    return _norm_text(unidecode(s or "").lower())


CANON_CATEGORY = {
    "mezze": "mezze_froid",
    "mezze froid": "mezze_froid",
    "mezze_froid": "mezze_froid",
    "mezze chaud": "mezze_chaud",
    "mezze_chaud": "mezze_chaud",
    "plat principal": "plat_principal",
    "plat_principal": "plat_principal",
    "dessert": "dessert",
    "entrée": "entree",
    "entree": "entree",
    "salade": "salade",
    "soupe": "soupe",
    "sauces": "sauces",
    "sauce": "sauces",
    "boisson": "boisson",
    "cocktail": "boisson",
    "apéro": "boisson",
    "apero": "boisson",
}

CANON_DIFFICULTY = {
    "facile": "facile",
    "moyenne": "moyenne",
    "moyen": "moyenne",
    "difficile": "difficile",
    "non spécifié": "non_specifie",
    "non specifie": "non_specifie",
    "": "non_specifie",
    None: "non_specifie",
}


class LLMBackfill(BaseModel):
    """Minimal enrichment backfill returned by the LLM (JSON mode)."""

    is_recipe: bool = Field(default=True)
    is_lebanese: bool = Field(default=True)
    cuisine_type: str = Field(default="libanaise")
    category: str = Field(default="autre")
    difficulty: str = Field(default="non_specifie")
    dish_canonical_name: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    main_ingredients: List[str] = Field(default_factory=list)


def _call_llm_backfill(model: str, api_key: str, payload: dict) -> Optional[LLMBackfill]:
    """
    Offline LLM backfill using OpenAI JSON mode.
    Returns None if OpenAI is not configured or call fails.
    """
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    client = OpenAI(api_key=api_key)

    system = (
        "Tu es un annotateur de données culinaires pour L'Orient-Le Jour. "
        "Ta mission: compléter des métadonnées manquantes pour une recette.\n\n"
        "Réponds UNIQUEMENT en JSON valide.\n"
        "Champs attendus:\n"
        "{\n"
        '  "is_recipe": true|false,\n'
        '  "is_lebanese": true|false,\n'
        '  "cuisine_type": "libanaise|méditerranéenne|..." ,\n'
        '  "category": "mezze|mezze_froid|mezze_chaud|plat principal|dessert|entrée|salade|soupe|sauces|boisson|autre",\n'
        '  "difficulty": "facile|moyenne|difficile|non_specifie",\n'
        '  "dish_canonical_name": "string|null",\n'
        '  "aliases": ["..."],\n'
        '  "main_ingredients": ["..."]\n'
        "}\n\n"
        "Règles:\n"
        "- Si c'est clairement une recette OLJ, is_recipe=true.\n"
        "- is_lebanese=true si plat libanais ou très courant au Liban.\n"
        "- Pour un plat arménien très courant au Liban (ex: su beureg), mets is_lebanese=true, cuisine_type='arménienne (Liban)'.\n"
        "- Sois conservateur: si incertain, category='autre' et difficulty='non_specifie'."
    )

    user = f"Voici la recette (données brutes):\n{json.dumps(payload, ensure_ascii=False)}"

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_completion_tokens=500,
    )

    data = json.loads(resp.choices[0].message.content or "{}")
    try:
        return LLMBackfill(**data)
    except ValidationError:
        return None


def _build_search_text(item: dict, enrichment: dict) -> str:
    title = _norm_text(item.get("title") or item.get("recipe_section_title") or "")
    chef = _norm_text(item.get("chef_name") or "")
    desc = _norm_text(item.get("recipe_description") or "")
    tags = " ".join(_norm_text(t) for t in (item.get("tags") or []) if t)
    ingredients = " ".join(_norm_text(i) for i in (item.get("ingredients") or []) if i)
    instructions = " ".join(_norm_text(i) for i in (item.get("instructions") or []) if i)
    aliases = " ".join(_norm_text(a) for a in (enrichment.get("aliases") or []) if a)
    mains = " ".join(_norm_text(m) for m in (enrichment.get("main_ingredients") or []) if m)
    return _norm_text(f"{title} {chef} {desc} {ingredients} {instructions} {tags} {aliases} {mains}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="v2/data_base_OLJ_enriched.json")
    parser.add_argument("--output", default="v2/data/olj_canonical.json")
    parser.add_argument("--model", default=None, help="OpenAI model for backfill (default from env/config)")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or uses env/core config)")
    args = parser.parse_args()

    # Project root = .../V3 (scripts live under v2/backend/scripts/)
    project_root = Path(__file__).resolve().parents[3]

    def _resolve(p: str) -> Path:
        path = Path(p)
        return (path if path.is_absolute() else (project_root / path)).resolve()

    inp = _resolve(args.input)
    out = _resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load settings (optional)
    model = args.model
    api_key = args.api_key
    try:
        from app.core.config import get_settings

        settings = get_settings()
        model = model or settings.rerank_model or settings.openai_model
        api_key = api_key or settings.openai_api_key
    except Exception:
        pass

    raw = json.loads(inp.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Expected a list JSON input")

    by_url: Dict[str, dict] = {}
    for item in raw:
        url = item.get("url")
        if not url:
            continue
        by_url[str(url)] = item

    canonical_docs: List[dict] = []

    for url, item in by_url.items():
        enrichment = item.get("enrichment") or {}
        raw_enrichment_present = bool(enrichment)

        # Ensure title isn't empty
        title = _norm_text(item.get("title") or item.get("recipe_section_title") or "")
        if not title:
            title = _slug_to_title(url)

        chef_name = _norm_text(item.get("chef_name") or "") or None

        # Determine whether enrichment is "complete enough"
        critical_missing = (
            not enrichment.get("category")
            or enrichment.get("is_recipe") is None
            or enrichment.get("is_lebanese") is None
            or not enrichment.get("difficulty")
        )

        llm_backfill: Optional[LLMBackfill] = None
        if critical_missing:
            payload = {
                "url": url,
                "title": title,
                "chef_name": chef_name,
                "recipe_description": item.get("recipe_description"),
                "ingredients": item.get("ingredients"),
                "instructions": item.get("instructions"),
                "tags": item.get("tags"),
                "category_raw": item.get("category"),
            }
            llm_backfill = _call_llm_backfill(model=model or "gpt-4o-mini", api_key=api_key or "", payload=payload)

        # Merge enrichment with LLM backfill (LLM only fills missing critical fields)
        merged = dict(enrichment)
        if llm_backfill:
            for k, v in llm_backfill.model_dump().items():
                if merged.get(k) in (None, "", [], {}):
                    merged[k] = v

        # Canonicalize category/difficulty
        raw_category = _norm_text(merged.get("category") or item.get("category") or "")
        category_canonical = CANON_CATEGORY.get(_norm_token(raw_category), "autre")

        raw_difficulty = _norm_text(merged.get("difficulty") or item.get("difficulty") or "")
        difficulty_canonical = CANON_DIFFICULTY.get(_norm_token(raw_difficulty), "non_specifie")

        is_recipe = bool(merged.get("is_recipe", True))
        is_lebanese = bool(merged.get("is_lebanese", True))
        cuisine_type = _norm_text(merged.get("cuisine_type") or "") or None

        aliases = merged.get("aliases") or []
        main_ingredients = merged.get("main_ingredients") or []
        tags = item.get("tags") or []

        search_text = _build_search_text(item, merged)
        if not search_text:
            # absolute fallback: title only
            search_text = title

        canonical_docs.append(
            {
                "url": url,
                "title": title,
                "chef_name": chef_name,
                "cuisine_type": cuisine_type,
                "is_lebanese": is_lebanese,
                "is_recipe": is_recipe,
                "category_canonical": category_canonical,
                "difficulty_canonical": difficulty_canonical,
                "tags": [t for t in tags if t],
                "main_ingredients": [i for i in main_ingredients if i],
                "aliases": [a for a in aliases if a],
                "search_text": search_text,
                "source": "olj",
                "raw_category": raw_category or None,
                "raw_difficulty": raw_difficulty or None,
                "raw_enrichment_present": raw_enrichment_present,
            }
        )

    out.write_text(json.dumps(canonical_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(canonical_docs)} canonical docs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Canonical Document Schema
=========================

V7.2 introduces a durable RAG approach that depends on a *canonical* dataset:
- No ambiguous empty strings for critical fields
- Normalized taxonomy (categories/difficulty)
- A single `search_text` field used consistently for lexical retrieval

This schema is used by:
- Offline backfill scripts to produce `v2/data/olj_canonical.json`
- Runtime retriever to load and validate documents
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


CanonicalCategory = Literal[
    "mezze_froid",
    "mezze_chaud",
    "plat_principal",
    "dessert",
    "entree",
    "salade",
    "soupe",
    "sauces",
    "boisson",
    "autre",
]

CanonicalDifficulty = Literal["facile", "moyenne", "difficile", "non_specifie"]


class CanonicalRecipeDoc(BaseModel):
    """Canonical recipe document used by retrieval."""

    # Identity
    url: HttpUrl
    title: str = Field(min_length=1)

    # Editorial metadata
    chef_name: Optional[str] = None
    cuisine_type: Optional[str] = None
    is_lebanese: bool = True
    is_recipe: bool = True

    # Normalized taxonomy
    category_canonical: CanonicalCategory = "autre"
    difficulty_canonical: CanonicalDifficulty = "non_specifie"

    # Structured fields that help retrieval
    tags: List[str] = Field(default_factory=list)
    main_ingredients: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)

    # Retrieval text (single source of truth for lexical retrieval)
    search_text: str = Field(min_length=1)

    # Optional debugging/provenance (kept lightweight)
    source: Literal["olj"] = "olj"
    raw_category: Optional[str] = None
    raw_difficulty: Optional[str] = None
    raw_enrichment_present: bool = True
    
    # Media
    image_url: Optional[str] = Field(default=None, alias="_image_url")
    
    class Config:
        populate_by_name = True  # Accept both _image_url and image_url

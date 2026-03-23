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

    # Normalized fields for exact-match indexing (filled by backfill or at load)
    title_normalized: Optional[str] = None
    aliases_normalized: List[str] = Field(default_factory=list)
    main_ingredients_normalized: List[str] = Field(default_factory=list)
    dish_name_normalized: Optional[str] = None

    # Retrieval text (single source of truth for lexical retrieval)
    search_text: str = Field(min_length=1)

    # Optional debugging/provenance (kept lightweight)
    source: Literal["olj"] = "olj"
    raw_category: Optional[str] = None
    raw_difficulty: Optional[str] = None
    raw_enrichment_present: bool = True
    
    # Media
    image_url: Optional[str] = Field(default=None, alias="_image_url")
    
    def model_post_init(self, __context) -> None:
        """Fill normalized fields from raw fields when missing."""
        from ..data.normalizers import normalize_text
        if not self.title_normalized:
            object.__setattr__(self, "title_normalized", normalize_text(self.title))
        if not self.aliases_normalized:
            object.__setattr__(
                self,
                "aliases_normalized",
                [normalize_text(a) for a in (self.aliases or []) if a],
            )
        if not self.main_ingredients_normalized:
            object.__setattr__(
                self,
                "main_ingredients_normalized",
                [normalize_text(m) for m in (self.main_ingredients or []) if m],
            )
        if not self.dish_name_normalized and self.title:
            object.__setattr__(self, "dish_name_normalized", normalize_text(self.title))

    class Config:
        populate_by_name = True  # Accept both _image_url and image_url

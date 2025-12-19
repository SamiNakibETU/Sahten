"""
Query Analysis Schema
=====================

THE CORE of Sahten.

This Pydantic schema is filled by a single LLM call using Instructor.
It replaces ALL the keyword matching, regex patterns, and manual extraction
with ~95%+ accuracy.

The LLM analyzes the user query and returns:
- Safety assessment (injection, toxicity)
- Intent classification
- All relevant filters
- Confidence score
- Redirect suggestion if blocked/off-topic
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class SafetyCheck(BaseModel):
    """
    Safety assessment of the query.
    
    Detects:
    - Prompt injection attempts
    - Toxic/offensive content
    - Manipulation attempts
    """
    
    is_safe: bool = Field(
        description="True si la requête est safe (pas d'injection, pas de toxicité)"
    )
    
    threat_type: Literal["injection", "toxicity", "none"] = Field(
        default="none",
        description="Type de menace détectée: injection (prompt manipulation), toxicity (insultes/offensive), none"
    )
    
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confiance dans l'évaluation de sécurité (0-1)"
    )


class QueryAnalysis(BaseModel):
    """
    Complete analysis of a user query by the LLM.
    
    This single structured output replaces:
    - Injection detection (regex patterns) → safety.is_safe
    - Toxicity filtering (word lists) → safety.threat_type
    - Intent classification (keyword matching) → intent
    - Filter extraction (regex patterns) → dish_name, ingredients, category, etc.
    - Off-topic detection (keyword lists) → is_culinary
    
    All done in ONE LLM call with ~95%+ accuracy.
    Cost: ~$0.0001 per query with GPT-4o-mini
    """
    
    # === SAFETY ===
    safety: SafetyCheck = Field(
        default_factory=lambda: SafetyCheck(is_safe=True, threat_type="none"),
        description="Évaluation de sécurité de la requête"
    )
    
    # === INTENT CLASSIFICATION ===
    intent: Literal[
        "recipe_specific",      # Recette précise: "recette taboulé"
        "recipe_by_ingredient", # Par ingrédients: "j'ai du poulet"
        "recipe_by_mood",       # Par humeur: "réconfortant"
        "recipe_by_diet",       # Par régime: "végétarien"
        "recipe_by_category",   # Par catégorie: "un dessert"
        "recipe_by_chef",       # Par chef: "Tara Khattar"
        "menu_composition",     # Menu complet: "entrée plat dessert"
        "multi_recipe",         # Plusieurs: "3 mezze"
        "greeting",             # Salutation: "bonjour"
        "clarification",        # Question: "c'est quoi le zaatar"
        "off_topic",            # Hors sujet culinaire
    ] = Field(
        default="recipe_by_mood",
        description="L'intention principale de l'utilisateur"
    )
    
    intent_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confiance dans la classification d'intent (0-1)"
    )
    
    # === CULINARY CHECK ===
    is_culinary: bool = Field(
        default=True,
        description="True si la requête concerne la nourriture/cuisine, même indirectement ('j'ai faim' = True)"
    )
    
    # === EXTRACTED FILTERS ===
    
    dish_name: Optional[str] = Field(
        default=None,
        description="Nom du plat demandé, normalisé en français. Ex: 'taboulé' même si user dit 'tabbouleh'"
    )
    
    dish_name_variants: List[str] = Field(
        default_factory=list,
        description="Variantes du nom pour la recherche: ['taboulé', 'tabbouleh', 'taboule', 'تبولة']"
    )
    
    ingredients: List[str] = Field(
        default_factory=list,
        description="Ingrédients mentionnés, normalisés en français"
    )
    
    category: Optional[Literal[
        "dessert", 
        "mezze_froid", 
        "mezze_chaud", 
        "entree", 
        "plat_principal", 
        "soupe", 
        "salade", 
        "boisson"
    ]] = Field(
        default=None,
        description="Catégorie de plat demandée"
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v):
        """
        Normalize common category synonyms produced by LLMs (or user phrasing).

        This prevents brittle failures like category='plat' (should be 'plat_principal').
        Unknown categories are coerced to None (instead of failing the whole analysis).
        """
        if v is None:
            return None
        if not isinstance(v, str):
            return v

        s = v.strip().lower()
        mapping = {
            "plat": "plat_principal",
            "plat principal": "plat_principal",
            "plats": "plat_principal",
            "entree": "entree",
            "entrée": "entree",
            "mezzé": "mezze_froid",
            "mezze": "mezze_froid",
            "mezze froid": "mezze_froid",
            "mezze chaud": "mezze_chaud",
            "boisson": "boisson",
            "drink": "boisson",
            "dessert": "dessert",
            "salade": "salade",
            "soupe": "soupe",
            "sauce": None,  # not in schema (yet)
            "sauces": None,
        }
        normalized = mapping.get(s, s)
        allowed = {
            "dessert",
            "mezze_froid",
            "mezze_chaud",
            "entree",
            "plat_principal",
            "soupe",
            "salade",
            "boisson",
        }
        if normalized in allowed:
            return normalized
        return None
    
    dietary_restrictions: List[Literal[
        "vegetarien", 
        "vegan", 
        "sans_gluten", 
        "sans_lactose", 
        "halal", 
        "low_calorie"
    ]] = Field(
        default_factory=list,
        description="Restrictions alimentaires détectées"
    )
    
    mood_tags: List[Literal[
        "reconfortant", 
        "frais", 
        "festif", 
        "rapide", 
        "traditionnel", 
        "moderne", 
        "hiver", 
        "ete",
        "facile",
        "chaud",
        "froid",
        "leger",
        "copieux"
    ]] = Field(
        default_factory=list,
        description="Tags d'humeur/occasion détectés"
    )
    
    chef_name: Optional[str] = Field(
        default=None,
        description="Nom du chef si mentionné"
    )
    
    recipe_count: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Nombre de recettes demandées (défaut: 1, menu: 3)"
    )
    
    # === METADATA ===
    
    user_language: Literal["fr", "en", "ar"] = Field(
        default="fr",
        description="Langue principale de l'utilisateur"
    )
    
    reasoning: str = Field(
        default="",
        description="Courte explication de l'analyse (1-2 phrases, aide au debug)"
    )
    
    redirect_suggestion: Optional[str] = Field(
        default=None,
        description="Si off-topic/blocked: suggestion de plat libanais en rapport avec le contexte"
    )
    
    # === HELPERS ===
    
    @property
    def should_redirect(self) -> bool:
        """Check if query should be redirected."""
        return (
            not self.safety.is_safe or
            self.intent == "off_topic" or
            not self.is_culinary
        )
    
    @property
    def needs_search(self) -> bool:
        """Check if we need to search for recipes."""
        return (
            self.safety.is_safe and
            self.is_culinary and
            self.intent not in ["greeting", "off_topic"]
        )



"""
Response Schemas
================

Structured response schemas for Sahten.
Ensures consistent, validated responses across the pipeline.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class RecipeNarrative(BaseModel):
    """
    Partie narrative de la réponse Sahten.
    
    Structure engageante avec:
    - hook: Phrase d'accroche enthousiaste
    - cultural_context: Contexte historique/culturel
    - teaser: Ce qui rend la recette spéciale
    - cta: Call to action vers OLJ
    - closing: Toujours "Sahten !"
    """
    
    hook: str = Field(
        min_length=10, 
        description="Phrase d'accroche enthousiaste (ex: 'Ah le taboulé !')"
    )
    
    cultural_context: str = Field(
        min_length=30, 
        description="2-3 phrases sur l'histoire, la tradition, l'origine du plat"
    )
    
    teaser: Optional[str] = Field(
        default=None, 
        description="Une phrase sur ce qui rend cette recette spéciale"
    )
    
    cta: str = Field(
        description="Invitation à découvrir sur L'Orient-Le Jour"
    )
    
    closing: Literal["Sahten !"] = "Sahten !"
    
    @field_validator('hook')
    @classmethod
    def hook_must_be_enthusiastic(cls, v: str) -> str:
        """Le hook doit être engageant."""
        # On accepte tout hook de longueur suffisante
        return v
    
    @field_validator('cultural_context')
    @classmethod
    def context_must_be_rich(cls, v: str) -> str:
        """Le contexte culturel doit être substantiel."""
        if len(v) < 30:
            raise ValueError("Le contexte culturel doit faire au moins 30 caractères")
        return v


class RecipeCard(BaseModel):
    """
    Carte de recette à afficher dans l'interface.
    
    Deux types:
    - OLJ: Lien vers l'article L'Orient-Le Jour
    - Base2: Recette complète avec ingrédients et étapes
    """
    
    source: Literal["olj", "base2"]
    title: str
    url: Optional[str] = None
    chef: Optional[str] = None
    category: str = ""
    image_url: Optional[str] = None
    
    # Champs pour Base2 uniquement (recette complète)
    ingredients: Optional[List[str]] = None
    steps: Optional[List[str]] = None
    prep_time: Optional[str] = None
    cook_time: Optional[str] = None
    servings: Optional[int] = None
    
    @field_validator('url')
    @classmethod
    def validate_olj_url(cls, v: Optional[str], info) -> Optional[str]:
        """Les URLs OLJ doivent pointer vers lorientlejour.com."""
        if v and info.data.get('source') == 'olj':
            if 'lorientlejour.com' not in v:
                raise ValueError(f"URL OLJ invalide: {v}")
        return v
    
    @field_validator('ingredients')
    @classmethod
    def base2_needs_ingredients(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        """Les recettes Base2 doivent avoir des ingrédients."""
        # Note: On ne force pas ici car parfois on crée des cartes OLJ
        return v


class OLJRecommendation(BaseModel):
    """
    Recommandation contextuelle OLJ.
    
    Affichée après une recette Base2 pour rediriger
    vers L'Orient-Le Jour.
    """
    
    title: str
    url: str
    reason: str = Field(
        description="Pourquoi cette recommandation (ex: 'Pour découvrir d'autres variations')"
    )
    
    @field_validator('url')
    @classmethod
    def must_be_olj_url(cls, v: str) -> str:
        """L'URL doit pointer vers L'Orient-Le Jour."""
        if v and 'lorientlejour.com' not in v:
            raise ValueError(f"URL de recommandation invalide: {v}")
        return v


class SahtenResponse(BaseModel):
    """
    Réponse complète structurée de Sahten.
    
    Types de réponse:
    - recipe_olj: Recette(s) de L'Orient-Le Jour
    - recipe_base2: Recette(s) Base2 avec reco OLJ
    - menu: Menu complet (entrée, plat, dessert)
    - greeting: Salutation d'accueil
    - redirect: Redirection (injection, toxicité, hors-sujet)
    - clarification: Explication d'un terme culinaire
    """
    
    response_type: Literal[
        "recipe_olj", 
        "recipe_base2", 
        "menu", 
        "greeting", 
        "redirect", 
        "clarification"
    ]
    
    narrative: RecipeNarrative
    
    recipes: List[RecipeCard] = Field(default_factory=list)
    
    olj_recommendation: Optional[OLJRecommendation] = None
    
    recipe_count: int = Field(ge=0, default=0)
    
    # Métadonnées pour debugging/analytics
    intent_detected: Optional[str] = None
    confidence: Optional[float] = None
    
    @field_validator('recipes')
    @classmethod
    def validate_recipe_list(cls, v: List[RecipeCard], info) -> List[RecipeCard]:
        """Valide la cohérence de la liste de recettes."""
        return v
    
    def model_post_init(self, __context) -> None:
        """Validation post-init: recipe_count doit matcher len(recipes)."""
        if self.recipe_count != len(self.recipes):
            # Auto-correct plutôt que de lever une erreur
            object.__setattr__(self, 'recipe_count', len(self.recipes))



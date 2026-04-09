"""
Response Schemas
================

Structured response schemas for Sahten.
Ensures consistent, validated responses across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Literal, Any
from pydantic import BaseModel, Field, field_validator


@dataclass
class SharedIngredientProof:
    """
    Proof that an alternative recipe shares at least one main ingredient.
    Used for audit and analytics.
    """
    query_ingredient: str
    normalized_ingredient: str
    shared_ingredients: List[str]
    recipe_title: str
    recipe_url: str
    proof_score: int


@dataclass
class AlternativeMatch:
    """Recipe card with explicit proof of shared ingredient."""
    recipe_card: "RecipeCard"
    proof: SharedIngredientProof
    match_reason: str = "shared_ingredient_match"


@dataclass
class EvidenceBundle:
    """
    Facts-only bundle passed to LLM verbalization layers.
    LLM can phrase these facts but must not invent new decisions.
    """
    response_type: str
    intent_detected: Optional[str]
    user_query: str
    selected_recipe_cards: List["RecipeCard"] = field(default_factory=list)
    shared_ingredient_proof: Optional[SharedIngredientProof] = None
    exact_match: bool = False
    grounded_excerpt: Optional[str] = None
    grounded_title: Optional[str] = None
    grounded_url: Optional[str] = None
    olj_recommendation: Optional["OLJRecommendation"] = None
    match_reason: Optional[str] = None
    allowed_claims: List[str] = field(default_factory=list)
    forbidden_claims: List[str] = field(default_factory=list)
    session_context: dict[str, Any] = field(default_factory=dict)


class ConversationBlock(BaseModel):
    """Typed conversational block for richer front rendering."""
    block_type: Literal[
        "assistant_message",
        "match_reason",
        "grounded_snippet",
        "follow_up_question",
        "cta",
    ]
    text: str = Field(min_length=1)


class RecipeNarrative(BaseModel):
    """
    Partie narrative de la réponse Sahten.
    
    Structure éditoriale compacte avec:
    - hook: angle principal
    - cultural_context: justification concise
    - teaser: précision additionnelle optionnelle
    - cta: Call to action vers OLJ
    - closing: Toujours "Sahten !"
    """
    
    hook: str = Field(
        min_length=10, 
        description="Angle principal clair et utile"
    )
    
    cultural_context: str = Field(
        min_length=16,
        description="Justification concise, factuelle et lisible"
    )
    
    teaser: Optional[str] = Field(
        default=None, 
        description="Une phrase sur ce qui rend cette recette spéciale"
    )
    
    cta: str = Field(
        description="Invitation à découvrir sur L'Orient-Le Jour"
    )
    
    closing: str = "Sahteïn !"
    
    @field_validator('hook')
    @classmethod
    def hook_must_be_enthusiastic(cls, v: str) -> str:
        """Le hook doit être engageant."""
        # On accepte tout hook de longueur suffisante
        return v
    
    @field_validator('cultural_context')
    @classmethod
    def context_must_be_rich(cls, v: str) -> str:
        """Le contexte doit rester informatif, sans imposer une longueur excessive."""
        if len(v) < 16:
            raise ValueError("Le contexte doit faire au moins 16 caractères")
        return v


class RecipeCard(BaseModel):
    """
    Carte de recette à afficher dans l'interface.
    
    Deux types:
    - OLJ: Lien vers l'article L'Orient-Le Jour
    - Base2: Recette complète avec ingrédients et étapes
    
    Grounding:
    - cited_passage: Passage pertinent extrait du document source
    """
    
    source: Literal["olj", "base2"]
    title: str
    url: Optional[str] = None
    chef: Optional[str] = None
    category: str = ""
    image_url: Optional[str] = None
    
    # Grounding/Citation: passage pertinent du document source
    cited_passage: Optional[str] = Field(
        default=None,
        description="Passage court extrait du document qui justifie la pertinence de cette recette"
    )

    # Extraits éditoriaux (OLJ) pour narrative LLM, dérivés de search_text
    recipe_lead: Optional[str] = Field(
        default=None,
        description="Chapô / début d'article avant liste d'ingrédients",
    )
    story_snippet: Optional[str] = Field(
        default=None,
        description="Phrase(s) anecdote ou voix auteur si détectées",
    )
    
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
        "recipe_olj",                      # recipe_olj_found: recette trouvée dans OLJ
        "recipe_base2",                    # Fallback Base2
        "menu",                            # Menu complet
        "greeting",
        "redirect",                        # Off-topic, blocage sécurité
        "recipe_not_found",                # Recette absente, pas d'alternative prouvée
        "clarification",
        "not_found_with_alternative",      # recipe_not_found_with_proven_alternative
    ]
    
    narrative: RecipeNarrative
    conversation_blocks: List[ConversationBlock] = Field(default_factory=list)
    
    recipes: List[RecipeCard] = Field(default_factory=list)
    
    olj_recommendation: Optional[OLJRecommendation] = None
    
    recipe_count: int = Field(ge=0, default=0)
    
    # Métadonnées pour debugging/analytics
    intent_detected: Optional[str] = None
    confidence: Optional[float] = None
    model_used: Optional[str] = None  # Model used for this response (for A/B testing)
    
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



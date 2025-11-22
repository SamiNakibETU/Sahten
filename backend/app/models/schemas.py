"""
Pydantic schemas for Sahtein 3.0
All data models used throughout the application
"""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl, ConfigDict


# ============================================================================
# Data Models - Base 1 (OLJ Articles)
# ============================================================================

class RecipeArticle(BaseModel):
    """OLJ recipe article from Base 1"""

    article_id: str
    title: str
    normalized_title: str
    slug: str
    url: str
    chef: str | None = None
    author: str | None = None
    section: str = "Liban à table"
    tags: list[str] = Field(default_factory=list)
    publish_date: datetime | None = None
    modified_date: datetime | None = None
    popularity_score: float = 0.0
    short_summary: str = ""
    description: str = ""
    anecdote: str = ""
    tips: list[str] = Field(default_factory=list)

    # For link resolution
    is_editor_pick: bool = False


# ============================================================================
# Data Models - Base 2 (Structured Recipes)
# ============================================================================

class Ingredient(BaseModel):
    """Structured ingredient with quantity"""

    nom: str
    quantite: float | int | None = None
    unite: str | None = None

    def to_text(self) -> str:
        """Convert to human-readable text"""
        if self.quantite and self.unite:
            return f"{self.quantite} {self.unite} de {self.nom}"
        elif self.quantite:
            return f"{self.quantite} {self.nom}"
        else:
            return self.nom


class StructuredRecipe(BaseModel):
    """Structured recipe from Base 2"""

    recipe_id: str
    name: str
    normalized_name: str
    category: str = ""
    ingredients: list[Ingredient]
    steps: list[str] = Field(alias="etapes", default_factory=list)
    servings: int | None = Field(alias="nombre_de_personnes", default=None)
    prep_time: str | None = Field(alias="temps_preparation", default=None)
    cook_time: str | None = Field(alias="temps_cuisson", default=None)
    difficulty: str | None = Field(alias="difficulte", default=None)
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# Data Models - Golden Examples
# ============================================================================

class GoldenExample(BaseModel):
    """Golden example from test dataset"""

    id: str
    scenario: str
    title: str
    user_query: str
    response: str
    expected_intent: str | None = None
    expected_url: str | None = None
    metadata: dict = Field(default_factory=dict)


# ============================================================================
# RAG Pipeline Models
# ============================================================================

class ClassificationResult(BaseModel):
    """Result from ClassifierAgent"""

    intent: Literal[
        "food_request",
        "greeting",
        "farewell",
        "about_bot",
        "anti_injection",
        "off_topic"
    ]
    language: Literal["fr", "non_fr"]
    confidence: float = 1.0
    slots: dict[str, list[str]] = Field(default_factory=dict)
    # slots structure:
    # - dishes: list[str]
    # - ingredients: list[str]
    # - methods: list[str]  (e.g., "au four", "salade", "dessert")
    # - occasions: list[str]  (e.g., "mezze", "dinner")


class QueryPlan(BaseModel):
    """Query understanding and retrieval plan"""

    need_type: Literal[
        "recipe_by_ingredients",
        "recipe_by_name",
        "suggestions",
        "off_topic",
        "greeting",
        "about_bot"
    ]
    primary_dish: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    language: Literal["fr", "non_fr"]
    retrieval_query: str
    link_query: str | None = None


class RetrievalCandidate(BaseModel):
    """A single retrieval candidate"""

    source: Literal["olj", "base2"]
    content: str
    score: float
    metadata: dict = Field(default_factory=dict)

    # For OLJ candidates
    article_id: str | None = None

    # For Base 2 candidates
    recipe_id: str | None = None


class LinkResolutionResult(BaseModel):
    """Result from LinkResolver"""

    primary_article: RecipeArticle | None = None
    suggested_articles: list[RecipeArticle] = Field(default_factory=list)
    strategy: str = "none"  # e.g., "exact_match", "similarity", "fallback"
    confidence: float = 0.0


class ScenarioContext(BaseModel):
    """Context for scenario selection"""

    scenario_id: int
    scenario_name: str
    use_base: Literal["olj", "base2", "mixed", "none"]
    show_full_recipe: bool = False  # True only for Base 2 scenarios
    include_link: bool = True


class ChatResponse(BaseModel):
    """Final response to user"""

    html: str
    scenario_id: int
    scenario_name: str
    used_base: Literal["olj", "base2", "mixed", "none"]
    primary_url: str | None = None
    debug_info: dict = Field(default_factory=dict)


# ============================================================================
# API Models
# ============================================================================

class ChatRequest(BaseModel):
    """API request for chat endpoint"""

    message: str = Field(..., min_length=1, max_length=500)
    conversation_id: str | None = None
    debug: bool = False


class ChatResponseAPI(BaseModel):
    """API response for chat endpoint"""

    html: str
    scenario_id: int | None = None
    primary_url: str | None = None
    debug_info: dict | None = None


# ============================================================================
# Index Models (for retrieval)
# ============================================================================

class ContentDocument(BaseModel):
    """Document in content index"""

    doc_id: str
    source: Literal["olj", "base2"]
    content: str
    metadata: dict = Field(default_factory=dict)


class LinkDocument(BaseModel):
    """Document in link index (article-level only)"""

    article: RecipeArticle
    searchable_text: str  # Combined normalized_title + tags + chef

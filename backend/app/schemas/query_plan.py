"""
Plan de requête unifié (LLM-first) : une seule structure pour retrieval + génération.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

PlanTask = Literal[
    "named_dish",
    "browse_corpus",
    "by_ingredient",
    "by_category",
    "by_chef",
    "by_diet",
    "menu",
    "multi",
    "clarify_term",
    "greeting",
    "about_bot",
    "off_topic",
]

CuisineScope = Literal["lebanese_olj", "any", "non_lebanese_named"]

CourseHint = Literal["any", "entree", "mezze", "plat", "dessert"]

ConstraintTag = Literal[
    "season_hiver",
    "season_ete",
    "season_automne",
    "season_printemps",
    "social_famille",
    "social_convivial",
    "time_rapide",
    "time_facile",
    "diet_vegetarien",
    "diet_vegan",
    "diet_sans_gluten",
]

MoodTagLiteral = Literal[
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
    "convivial",
    "froid",
    "leger",
    "copieux",
    "liban",
]

DietLiteral = Literal["vegetarien", "vegan", "sans_gluten", "sans_lactose", "halal", "low_calorie"]

CategoryLiteral = Literal[
    "dessert",
    "mezze_froid",
    "mezze_chaud",
    "entree",
    "plat_principal",
    "soupe",
    "salade",
    "boisson",
]


class QueryPlan(BaseModel):
    """Sortie structurée stricte de l'analyseur LLM (contrat SOTA)."""

    task: PlanTask = Field(description="Type principal de la demande")
    cuisine_scope: CuisineScope = Field(
        default="lebanese_olj",
        description="Périmètre : corpus OLJ libanais, tout, ou plat étranger nommé",
    )
    course: CourseHint = Field(
        default="any",
        description="Moment / type de plat ciblé pour le retrieval",
    )
    dish_name: str = Field(
        default="",
        description="Plat normalisé si named_dish ou clarify_term (sinon vide)",
    )
    dish_variants: List[str] = Field(default_factory=list)
    ingredients: List[str] = Field(default_factory=list)
    inferred_main_ingredients: List[str] = Field(default_factory=list)
    category: str = Field(
        default="",
        description="Catégorie canonique ou chaîne vide",
    )
    chef_name: str = Field(default="")
    recipe_count: int = Field(default=1, ge=1, le=10)
    constraints: List[ConstraintTag] = Field(default_factory=list)
    mood_tags: List[MoodTagLiteral] = Field(default_factory=list)
    dietary_restrictions: List[DietLiteral] = Field(default_factory=list)
    retrieval_focus: str = Field(
        default="",
        description="Phrase courte pour TF-IDF / embeddings",
    )
    needs_clarification: bool = Field(default=False)
    clarification_question: str = Field(default="")
    safety_is_safe: bool = Field(default=True)
    safety_threat_type: Literal["injection", "toxicity", "none"] = Field(default="none")
    reasoning: str = Field(default="")
    is_culinary: bool = Field(default=True)
    redirect_suggestion: str = Field(default="")
    intent_confidence: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("recipe_count", mode="before")
    @classmethod
    def _coerce_recipe_count(cls, v: Any) -> int:
        if v is None:
            return 1
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            return 1

    @field_validator("intent_confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: Any) -> float:
        if v is None:
            return 0.9
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.9

    @field_validator(
        "dish_variants",
        "ingredients",
        "inferred_main_ingredients",
        "mood_tags",
        "dietary_restrictions",
        "constraints",
        mode="before",
    )
    @classmethod
    def _coerce_lists(cls, v: Any) -> Any:
        if v is None:
            return []
        return v

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, v: Any) -> str:
        if v is None:
            return ""
        if not isinstance(v, str):
            return ""
        s = v.strip().lower()
        mapping = {
            "plat": "plat_principal",
            "plat principal": "plat_principal",
            "plats": "plat_principal",
            "entrée": "entree",
            "entree": "entree",
            "mezzé": "mezze_froid",
            "mezze": "mezze_froid",
            "mezze froid": "mezze_froid",
            "mezze chaud": "mezze_chaud",
        }
        s = mapping.get(s, s)
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
        return s if s in allowed else ""


QUERY_PLAN_OPENAI_SCHEMA: dict[str, Any] = {
    "name": "sahten_query_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "enum": [
                    "named_dish",
                    "browse_corpus",
                    "by_ingredient",
                    "by_category",
                    "by_chef",
                    "by_diet",
                    "menu",
                    "multi",
                    "clarify_term",
                    "greeting",
                    "about_bot",
                    "off_topic",
                ],
            },
            "cuisine_scope": {
                "type": "string",
                "enum": ["lebanese_olj", "any", "non_lebanese_named"],
            },
            "course": {
                "type": "string",
                "enum": ["any", "entree", "mezze", "plat", "dessert"],
            },
            "dish_name": {"type": "string"},
            "dish_variants": {"type": "array", "items": {"type": "string"}},
            "ingredients": {"type": "array", "items": {"type": "string"}},
            "inferred_main_ingredients": {"type": "array", "items": {"type": "string"}},
            "category": {
                "type": "string",
                "description": "dessert, plat_principal, entree, mezze_froid, mezze_chaud, soupe, salade, boisson, ou vide",
            },
            "chef_name": {"type": "string"},
            "recipe_count": {"type": "integer"},
            "constraints": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "season_hiver",
                        "season_ete",
                        "season_automne",
                        "season_printemps",
                        "social_famille",
                        "social_convivial",
                        "time_rapide",
                        "time_facile",
                        "diet_vegetarien",
                        "diet_vegan",
                        "diet_sans_gluten",
                    ],
                },
            },
            "mood_tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
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
                        "convivial",
                        "froid",
                        "leger",
                        "copieux",
                        "liban",
                    ],
                },
            },
            "dietary_restrictions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["vegetarien", "vegan", "sans_gluten", "sans_lactose", "halal", "low_calorie"],
                },
            },
            "retrieval_focus": {"type": "string"},
            "needs_clarification": {"type": "boolean"},
            "clarification_question": {"type": "string"},
            "safety_is_safe": {"type": "boolean"},
            "safety_threat_type": {
                "type": "string",
                "enum": ["injection", "toxicity", "none"],
            },
            "reasoning": {"type": "string"},
            "is_culinary": {"type": "boolean"},
            "redirect_suggestion": {"type": "string"},
            "intent_confidence": {"type": "number"},
        },
        "required": [
            "task",
            "cuisine_scope",
            "course",
            "dish_name",
            "dish_variants",
            "ingredients",
            "inferred_main_ingredients",
            "category",
            "chef_name",
            "recipe_count",
            "constraints",
            "mood_tags",
            "dietary_restrictions",
            "retrieval_focus",
            "needs_clarification",
            "clarification_question",
            "safety_is_safe",
            "safety_threat_type",
            "reasoning",
            "is_culinary",
            "redirect_suggestion",
            "intent_confidence",
        ],
        "additionalProperties": False,
    },
}

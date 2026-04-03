"""Projection QueryPlan → QueryAnalysis (compat pipeline existant)."""

from __future__ import annotations

from typing import Optional

from .query_analysis import QueryAnalysis, SafetyCheck
from .query_plan import QueryPlan


def query_plan_to_analysis(plan: QueryPlan) -> QueryAnalysis:
    safety = SafetyCheck(
        is_safe=plan.safety_is_safe,
        threat_type=plan.safety_threat_type,
        confidence=1.0,
    )

    intent_map = {
        "named_dish": "recipe_specific",
        "browse_corpus": "recipe_by_mood",
        "by_ingredient": "recipe_by_ingredient",
        "by_category": "recipe_by_category",
        "by_chef": "recipe_by_chef",
        "by_diet": "recipe_by_diet",
        "menu": "menu_composition",
        "multi": "multi_recipe",
        "clarify_term": "clarification",
        "greeting": "greeting",
        "about_bot": "about_bot",
        "off_topic": "off_topic",
    }
    intent = intent_map[plan.task]  # type: ignore[assignment]

    dish_name: Optional[str] = None
    if plan.task == "named_dish":
        dish_name = plan.dish_name.strip() or None
    elif plan.task == "clarify_term":
        dish_name = plan.dish_name.strip() or None

    variants = [x.strip() for x in plan.dish_variants if x and str(x).strip()]
    if plan.task == "named_dish" and dish_name:
        if dish_name not in variants:
            variants = [dish_name, *variants]

    mood = list(plan.mood_tags)
    for c in plan.constraints:
        if c == "season_hiver":
            for t in ("hiver", "reconfortant", "chaud"):
                if t not in mood:
                    mood.append(t)  # type: ignore[arg-type]
        elif c == "season_ete":
            for t in ("ete", "frais", "leger"):
                if t not in mood:
                    mood.append(t)  # type: ignore[arg-type]
        elif c == "season_automne" and "reconfortant" not in mood:
            mood.append("reconfortant")
        elif c == "season_printemps":
            for t in ("frais", "leger"):
                if t not in mood:
                    mood.append(t)  # type: ignore[arg-type]
        elif c == "social_famille":
            for t in ("convivial", "copieux"):
                if t not in mood:
                    mood.append(t)  # type: ignore[arg-type]
        elif c == "social_convivial" and "convivial" not in mood:
            mood.append("convivial")
        elif c == "time_rapide" and "rapide" not in mood:
            mood.append("rapide")
        elif c == "time_facile" and "facile" not in mood:
            mood.append("facile")

    if plan.task == "browse_corpus" and plan.cuisine_scope == "lebanese_olj" and "liban" not in mood:
        mood.append("liban")
    if plan.task == "browse_corpus" and "traditionnel" not in mood:
        mood.append("traditionnel")

    diet = list(plan.dietary_restrictions)
    for c in plan.constraints:
        if c == "diet_vegetarien" and "vegetarien" not in diet:
            diet.append("vegetarien")
        elif c == "diet_vegan" and "vegan" not in diet:
            diet.append("vegan")
        elif c == "diet_sans_gluten" and "sans_gluten" not in diet:
            diet.append("sans_gluten")

    category: Optional[str] = plan.category.strip() or None
    if plan.task in ("by_category", "multi", "menu") and category is None:
        if plan.course == "dessert":
            category = "dessert"
        elif plan.course == "plat":
            category = "plat_principal"
        elif plan.course == "entree":
            category = "entree"
        elif plan.course == "mezze":
            category = "mezze_froid"

    inferred = [x.strip() for x in plan.inferred_main_ingredients if x and str(x).strip()]
    ingredients = [x.strip() for x in plan.ingredients if x and str(x).strip()]

    chef = plan.chef_name.strip() or None
    redirect = plan.redirect_suggestion.strip() or None

    return QueryAnalysis(
        safety=safety,
        intent=intent,
        intent_confidence=plan.intent_confidence,
        is_culinary=plan.is_culinary,
        dish_name=dish_name,
        dish_name_variants=variants,
        ingredients=ingredients,
        inferred_main_ingredients=inferred,
        category=category,  # type: ignore[arg-type]
        dietary_restrictions=diet,
        mood_tags=mood[:20],  # type: ignore[arg-type]
        chef_name=chef,
        recipe_count=plan.recipe_count,
        reasoning=plan.reasoning or f"QueryPlan.task={plan.task}",
        redirect_suggestion=redirect,
        plan=plan,
    )

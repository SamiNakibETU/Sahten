"""
Response Composer
Generates French HTML responses according to scenario and editorial constraints
"""

import logging
import random
from typing import Literal

from app.models.schemas import (
    QueryPlan,
    ClassificationResult,
    ScenarioContext,
    LinkResolutionResult,
    RetrievalCandidate,
    StructuredRecipe,
    RecipeArticle,
)
from app.data.loaders import data_cache

logger = logging.getLogger(__name__)


class ResponseComposer:
    """
    Composes French HTML responses respecting editorial constraints

    Key rules:
    - French only
    - HTML format (<p>, <br>, <a>)
    - 1-3 emojis max, food/emotion related, no flags
    - ~100 words except Base 2 full recipes
    - OLJ recipes: storytelling only, NO ingredient lists/steps, MUST link
    - Base 2 recipes: full recipe allowed, clarify it's not OLJ
    - Links: only https://www.lorientlejour.com
    """

    # Food and emotion emojis (no flags)
    ALLOWED_EMOJIS = [
        "🍽️", "🥗", "🍴", "👨‍🍳", "👩‍🍳", "🌿", "🥙", "🥘", "🍲",
        "😋", "😊", "👌", "✨", "💚", "🌟", "🎉",
    ]

    def __init__(self):
        pass

    def compose(
        self,
        scenario: ScenarioContext,
        query_plan: QueryPlan,
        classification: ClassificationResult,
        link_result: LinkResolutionResult,
        retrieval_candidates: list[RetrievalCandidate] | None = None,
    ) -> str:
        """
        Compose HTML response based on scenario

        Returns HTML string ready for frontend display
        """
        scenario_id = scenario.scenario_id

        # Route to appropriate composer
        if scenario_id == 1:
            return self._compose_olj_recipe(link_result, query_plan)
        elif scenario_id == 2:
            return self._compose_base2_recipe(link_result, retrieval_candidates, query_plan)
        elif scenario_id == 3:
            return self._compose_no_match_fallback(link_result)
        elif scenario_id == 4:
            return self._compose_greeting(link_result)
        elif scenario_id == 5:
            return self._compose_about_bot(link_result)
        elif scenario_id == 6:
            return self._compose_off_topic_redirect(link_result)
        elif scenario_id == 7:
            return self._compose_non_french()
        elif scenario_id == 8:
            return self._compose_ingredient_suggestions(link_result, retrieval_candidates)
        else:
            return self._compose_fallback()

    def _compose_olj_recipe(
        self,
        link_result: LinkResolutionResult,
        query_plan: QueryPlan,
    ) -> str:
        """Scenario 1: OLJ recipe available - storytelling only, no full recipe"""
        if not link_result.primary_article:
            return self._compose_fallback()

        article = link_result.primary_article
        emoji = self._pick_emoji()

        # Build storytelling response
        html_parts = [f"<p>{emoji} <strong>{article.title}</strong></p>"]

        # Add description/anecdote
        if article.description:
            desc = article.description[:180]
            html_parts.append(f"<p>{desc}</p>")
        elif article.anecdote:
            html_parts.append(f"<p>{article.anecdote}</p>")

        # Add chef attribution if available
        if article.chef:
            html_parts.append(f"<p>Une recette de {article.chef}.</p>")

        # MUST include link
        html_parts.append(
            f'<p><a href="{article.url}">Découvrez la recette complète ici</a></p>'
        )

        # Optional: suggest another recipe
        if link_result.suggested_articles:
            suggested = link_result.suggested_articles[0]
            html_parts.append(
                f'<p>Vous aimerez aussi : <a href="{suggested.url}">{suggested.title}</a></p>'
            )

        return "\n".join(html_parts)

    def _compose_base2_recipe(
        self,
        link_result: LinkResolutionResult,
        retrieval_candidates: list[RetrievalCandidate] | None,
        query_plan: QueryPlan,
    ) -> str:
        """Scenario 2: Base 2 structured recipe with OLJ suggestion"""
        # Find best Base 2 candidate
        base2_candidate = None
        if retrieval_candidates:
            for candidate in retrieval_candidates:
                if candidate.source == "base2" and candidate.recipe_id:
                    base2_candidate = candidate
                    break

        if not base2_candidate:
            return self._compose_fallback()

        # Load full recipe
        recipes = data_cache.get_structured_recipes()
        recipe = None
        for r in recipes:
            if r.recipe_id == base2_candidate.recipe_id:
                recipe = r
                break

        if not recipe:
            return self._compose_fallback()

        emoji = self._pick_emoji()
        html_parts = [f"<p>{emoji} <strong>{recipe.name}</strong></p>"]

        # Metadata
        meta_parts = []
        if recipe.servings:
            meta_parts.append(f"{recipe.servings} personnes")
        if recipe.prep_time:
            meta_parts.append(f"Préparation : {recipe.prep_time}")
        if recipe.difficulty:
            meta_parts.append(f"Difficulté : {recipe.difficulty}")

        if meta_parts:
            html_parts.append(f"<p><em>{' | '.join(meta_parts)}</em></p>")

        # Ingredients
        html_parts.append("<p><strong>Ingrédients :</strong><br>")
        for ing in recipe.ingredients[:10]:  # Limit to 10 for display
            html_parts.append(f"• {ing.to_text()}<br>")
        html_parts.append("</p>")

        # Steps (abbreviated)
        html_parts.append("<p><strong>Préparation :</strong><br>")
        for i, step in enumerate(recipe.steps[:5], 1):  # Show first 5 steps
            html_parts.append(f"{i}. {step[:100]}{'...' if len(step) > 100 else ''}<br>")
        html_parts.append("</p>")

        # OLJ suggestion
        if link_result.primary_article:
            article = link_result.primary_article
            html_parts.append(
                f'<p>Découvrez aussi sur L\'Orient-Le Jour : '
                f'<a href="{article.url}">{article.title}</a></p>'
            )

        return "\n".join(html_parts)

    def _compose_no_match_fallback(self, link_result: LinkResolutionResult) -> str:
        """Scenario 3: No match, fallback OLJ article"""
        emoji = random.choice(["😊", "🍽️"])

        html_parts = [
            f"<p>{emoji} Je n'ai pas trouvé exactement ce que vous cherchez, mais voici une suggestion !</p>"
        ]

        if link_result.primary_article:
            article = link_result.primary_article
            html_parts.append(
                f'<p><a href="{article.url}"><strong>{article.title}</strong></a></p>'
            )
            if article.description:
                html_parts.append(f"<p>{article.description[:120]}...</p>")

        return "\n".join(html_parts)

    def _compose_greeting(self, link_result: LinkResolutionResult) -> str:
        """Scenario 4: Greeting"""
        greetings = [
            "Bonjour ! 😊 Je suis Sahtein, votre assistant culinaire libanais.",
            "Salut ! 🍽️ Ravie de vous rencontrer. Je suis Sahtein, spécialiste de la cuisine libanaise.",
            "Bonjour ! 👨‍🍳 Bienvenue chez Sahtein, votre guide de la gastronomie libanaise.",
        ]

        html_parts = [f"<p>{random.choice(greetings)}</p>"]
        html_parts.append(
            "<p>Demandez-moi une recette, des suggestions avec vos ingrédients, "
            "ou des idées de mezze ! 🌿</p>"
        )

        # Suggest a recipe
        if link_result.primary_article:
            article = link_result.primary_article
            html_parts.append(
                f'<p>Pour commencer : <a href="{article.url}">{article.title}</a></p>'
            )

        return "\n".join(html_parts)

    def _compose_about_bot(self, link_result: LinkResolutionResult) -> str:
        """Scenario 5: About bot"""
        emoji = random.choice(["😊", "🍽️", "👨‍🍳"])

        html_parts = [
            f"<p>{emoji} Je suis Sahtein, votre assistant culinaire pour la cuisine libanaise "
            "et méditerranéenne orientale.</p>",
            "<p>Je vous aide à découvrir les recettes de L'Orient-Le Jour, "
            "et je peux vous suggérer des plats selon vos envies ou vos ingrédients.</p>",
        ]

        # Example recipe
        if link_result.primary_article:
            article = link_result.primary_article
            html_parts.append(
                f'<p>Par exemple : <a href="{article.url}">{article.title}</a></p>'
            )

        return "\n".join(html_parts)

    def _compose_off_topic_redirect(self, link_result: LinkResolutionResult) -> str:
        """Scenario 6: Off-topic, redirect to cuisine"""
        emoji = random.choice(["😊", "🍴"])

        redirects = [
            f"{emoji} Je suis spécialisé dans la cuisine libanaise et méditerranéenne. "
            "Puis-je vous suggérer une recette ?",
            f"{emoji} Ma spécialité, c'est la gastronomie libanaise ! "
            "Que diriez-vous de découvrir un délicieux mezze ?",
        ]

        html_parts = [f"<p>{random.choice(redirects)}</p>"]

        # Suggest a recipe
        if link_result.primary_article:
            article = link_result.primary_article
            html_parts.append(
                f'<p>Voici une suggestion : <a href="{article.url}">{article.title}</a></p>'
            )

        return "\n".join(html_parts)

    def _compose_non_french(self) -> str:
        """Scenario 7: Non-French query"""
        return (
            "<p>😊 Bonjour ! Je réponds uniquement en français.</p>"
            "<p>Pour découvrir nos recettes libanaises, posez-moi votre question en français !</p>"
        )

    def _compose_ingredient_suggestions(
        self,
        link_result: LinkResolutionResult,
        retrieval_candidates: list[RetrievalCandidate] | None,
    ) -> str:
        """Scenario 8: Multiple suggestions based on ingredients"""
        emoji = random.choice(["😋", "👨‍🍳"])

        html_parts = [
            f"<p>{emoji} Avec ces ingrédients, vous pouvez préparer plusieurs plats !</p>"
        ]

        # Show top 3 suggestions
        if retrieval_candidates:
            for i, candidate in enumerate(retrieval_candidates[:3], 1):
                if candidate.source == "base2":
                    name = candidate.metadata.get("name", "Recette")
                    html_parts.append(f"<p>{i}. {name}</p>")
                elif candidate.source == "olj":
                    title = candidate.metadata.get("title", "Recette")
                    url = candidate.metadata.get("url")
                    if url:
                        html_parts.append(f'<p>{i}. <a href="{url}">{title}</a></p>')

        # OLJ suggestion
        if link_result.primary_article:
            article = link_result.primary_article
            html_parts.append(
                f'<p>Sur L\'Orient-Le Jour : <a href="{article.url}">{article.title}</a></p>'
            )

        return "\n".join(html_parts)

    def _compose_fallback(self) -> str:
        """Generic fallback response"""
        return (
            "<p>😊 Désolé, je n'ai pas pu traiter votre demande.</p>"
            "<p>Demandez-moi une recette libanaise ou méditerranéenne, "
            "et je serai ravi de vous aider ! 🍽️</p>"
        )

    def _pick_emoji(self, count: int = 1) -> str:
        """Pick random allowed emoji(s)"""
        emojis = random.sample(self.ALLOWED_EMOJIS, min(count, len(self.ALLOWED_EMOJIS)))
        return " ".join(emojis)

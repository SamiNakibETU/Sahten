"""
Response Generator V7.1
=======================

Generates engaging narrative responses using LLM.
Called AFTER retrieval to create the final response text.

KEY: The narrative must acknowledge the USER'S REQUEST, not just describe the recipe.

Pipeline:
1. QueryAnalyzer → intent + filters + original query
2. Retriever → recipes
3. ResponseGenerator → engaging narrative that CONNECTS user request to recipe
"""

import json
import logging
from typing import List, Optional

from openai import AsyncOpenAI

from ..schemas.query_analysis import QueryAnalysis
from ..schemas.responses import RecipeNarrative, RecipeCard

logger = logging.getLogger(__name__)


NARRATIVE_SYSTEM_PROMPT = """Tu es Sahten, le chatbot culinaire libanais de L'Orient-Le Jour.

# TA MISSION
Générer une réponse qui CONNECTE la demande de l'utilisateur aux recettes trouvées.

# RÈGLE CRITIQUE
Ta réponse doit TOUJOURS faire le lien entre:
1. Ce que l'utilisateur a DEMANDÉ (sa requête originale)
2. Ce que tu lui PROPOSES (les recettes trouvées)

# FORMAT (JSON)
{
  "hook": "Phrase qui RÉPOND à la demande de l'utilisateur",
  "cultural_context": "2-3 phrases de contexte culturel sur le plat proposé",
  "teaser": "Pourquoi cette recette correspond à sa demande",
  "cta": "Invitation vers L'Orient-Le Jour"
}

# EXEMPLES PAR TYPE DE REQUÊTE

## REQUÊTE PAR MOOD (ex: "recette pour l'hiver", "plat réconfortant")
{
  "hook": "Pour te réchauffer cet hiver, j'ai exactement ce qu'il te faut !",
  "cultural_context": "La Mouloukhiyé est LE plat réconfortant par excellence au Liban. Ce ragoût de corète, mijoté pendant des heures, réchauffe les cœurs depuis des générations.",
  "teaser": "Tara Khattar en a fait sa spécialité avec sa version au poulet et bœuf.",
  "cta": "Découvre sa recette sur L'Orient-Le Jour"
}

## REQUÊTE PAR INGRÉDIENT (ex: "recette avec du yaourt")
{
  "hook": "Du yaourt ? J'ai plusieurs idées délicieuses pour toi !",
  "cultural_context": "Au Liban, le yaourt est un ingrédient star. On le retrouve dans les sauces tarator, les marinades, et même certains desserts.",
  "teaser": "Alan Geaam a créé des sauces au yaourt incroyables que tu vas adorer.",
  "cta": "Découvre ses recettes sur L'Orient-Le Jour"
}

## REQUÊTE SPÉCIFIQUE (ex: "recette taboulé")
{
  "hook": "Ah le taboulé ! Tu as excellent goût.",
  "cultural_context": "Le vrai taboulé libanais est surtout du PERSIL avec un peu de boulgour — pas l'inverse ! Chaque famille a sa recette secrète.",
  "teaser": "Kamal Mouzawak, fondateur de Souk el-Tayeb, en a fait sa signature.",
  "cta": "Découvre sa technique sur L'Orient-Le Jour"
}

## REQUÊTE FACILE/RAPIDE (ex: "plat facile")
{
  "hook": "Quelque chose de facile ? Parfait, j'ai ce qu'il te faut !",
  "cultural_context": "La cuisine libanaise a plein de recettes simples et savoureuses. Pas besoin d'être un chef pour régaler.",
  "teaser": "Cette recette est accessible même aux débutants.",
  "cta": "Lance-toi avec cette recette sur L'Orient-Le Jour"
}

## MENU COMPOSITION (ex: "entrée plat dessert")
{
  "hook": "Un menu complet ? Yalla, je te compose un festin libanais !",
  "cultural_context": "Au Liban, un repas c'est sacré. On commence par les mezze, on enchaîne avec un plat généreux, et on finit toujours sur une note sucrée.",
  "teaser": "Voici une sélection qui impressionnera tes invités.",
  "cta": "Découvre ces recettes sur L'Orient-Le Jour"
}

## BASE2 FALLBACK (recette complète fournie)
{
  "hook": "[Réponse à la demande] Voici une recette classique que je te donne en entier !",
  "cultural_context": "[Contexte culturel du plat]",
  "teaser": "Je te donne tous les détails : ingrédients et étapes.",
  "cta": "Et pour plus d'inspiration, explore L'Orient-Le Jour"
}

# RÈGLES STRICTES
- Le hook doit RÉPONDRE à la demande, pas juste nommer le plat
- JAMAIS commencer par "Ah, [nom du plat]" si l'utilisateur n'a pas demandé ce plat spécifiquement
- JAMAIS de listes numérotées
- TOUJOURS contextualiser POURQUOI cette recette répond à la demande
"""


class ResponseGenerator:
    """
    Generates narrative responses using LLM.
    
    KEY: Uses the original user query to generate contextually relevant responses.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        from ..core.config import get_settings
        settings = get_settings()
        
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate_narrative(
        self,
        user_query: str,  # ADDED: Original user query
        analysis: QueryAnalysis,
        recipes: List[RecipeCard],
        is_base2_fallback: bool = False,
    ) -> RecipeNarrative:
        """
        Generate engaging narrative for the response.
        
        Args:
            user_query: ORIGINAL user query (critical for context!)
            analysis: Query analysis result
            recipes: Retrieved recipes
            is_base2_fallback: True if using Base2 (not OLJ)
            
        Returns:
            RecipeNarrative with hook, context, teaser, cta
        """
        # Durable offline fallback: if no API key, don't attempt network calls.
        if not self.api_key:
            return self._fallback_narrative(user_query, recipes, is_base2_fallback)

        # Build context for LLM - INCLUDING USER QUERY
        context = {
            "user_query": user_query,  # CRITICAL: What the user actually asked
            "intent": analysis.intent,
            "mood_tags": analysis.mood_tags,
            "ingredients": analysis.ingredients,
            "category": analysis.category,
            "dish_name": analysis.dish_name,
            "is_base2": is_base2_fallback,
            "recipes": [
                {"title": r.title, "chef": r.chef, "source": r.source, "category": r.category}
                for r in recipes[:3]
            ] if recipes else []
        }
        
        # Build specific instructions based on intent
        instructions = []
        
        if analysis.intent == "recipe_by_mood":
            instructions.append(f"L'utilisateur cherche: '{user_query}'. Réponds à son ENVIE, pas au nom du plat.")
        elif analysis.intent == "recipe_by_ingredient":
            instructions.append(f"L'utilisateur a des ingrédients: {analysis.ingredients}. Montre comment la recette les utilise.")
        elif analysis.intent == "menu_composition":
            instructions.append("L'utilisateur veut un menu complet. Présente les 3 plats comme un ensemble cohérent.")
        elif analysis.intent == "recipe_specific":
            instructions.append(f"L'utilisateur a demandé spécifiquement: {analysis.dish_name}. Tu peux être enthousiaste sur CE plat.")
        
        if is_base2_fallback:
            instructions.append("IMPORTANT: C'est une recette de notre base classique. Mentionne que tu donnes la recette complète + invite à explorer OLJ pour plus.")
        
        user_prompt = f"""Requête utilisateur: "{user_query}"

Contexte: {json.dumps(context, ensure_ascii=False)}

{chr(10).join(instructions) if instructions else ""}

Génère une réponse narrative qui CONNECTE la demande de l'utilisateur aux recettes proposées."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_completion_tokens=350,
            )
            
            data = json.loads(response.choices[0].message.content)
            
            return RecipeNarrative(
                hook=data.get("hook", "Excellente idée !"),
                cultural_context=data.get("cultural_context", "La cuisine libanaise est riche en saveurs."),
                teaser=data.get("teaser"),
                cta=data.get("cta", "Découvre sur L'Orient-Le Jour"),
                closing="Sahten !"
            )
            
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            return self._fallback_narrative(user_query, recipes, is_base2_fallback)
    
    def _fallback_narrative(
        self,
        user_query: str,
        recipes: List[RecipeCard],
        is_base2: bool
    ) -> RecipeNarrative:
        """Fallback narrative if LLM fails."""
        if not recipes:
            return RecipeNarrative(
                hook="Je n'ai pas trouvé exactement ce que tu cherches !",
                cultural_context="Mais la cuisine libanaise est si riche qu'il y a forcément quelque chose qui va te plaire.",
                teaser=None,
                cta="Explore nos recettes sur L'Orient-Le Jour",
                closing="Sahten !"
            )
        
        title = recipes[0].title
        
        # Contextual fallback based on what seems like a mood/ingredient request
        if any(word in user_query.lower() for word in ["hiver", "chaud", "réconfortant", "froid"]):
            return RecipeNarrative(
                hook=f"Pour répondre à ton envie, je te propose {title} !",
                cultural_context="La cuisine libanaise regorge de plats adaptés à chaque saison et chaque humeur.",
                teaser="Cette recette devrait parfaitement correspondre à ce que tu cherches.",
                cta="Découvre les détails sur L'Orient-Le Jour" if not is_base2 else "Voici la recette complète, et explore plus sur L'Orient-Le Jour",
                closing="Sahten !"
            )
        
        return RecipeNarrative(
            hook=f"J'ai trouvé quelque chose pour toi : {title} !",
            cultural_context="La cuisine libanaise regorge de saveurs et de traditions. Ce plat en est un parfait exemple.",
            teaser="Une recette authentique qui devrait te plaire.",
            cta="Retrouve tous les détails sur L'Orient-Le Jour" if not is_base2 else "Voici la recette, et explore plus sur L'Orient-Le Jour",
            closing="Sahten !"
        )
    
    def generate_greeting(self) -> RecipeNarrative:
        """Generate greeting response (no LLM needed)."""
        return RecipeNarrative(
            hook="Marhaba ! Je suis Sahten, ton guide culinaire libanais.",
            cultural_context="Je suis là pour te faire découvrir les trésors de la cuisine libanaise à travers les recettes de L'Orient-Le Jour. Du houmous onctueux aux baklavas dorés, en passant par les kebbés croustillants — c'est tout un voyage !",
            teaser="Dis-moi ce qui te ferait plaisir : une recette précise, des idées selon tes ingrédients, ou un menu complet.",
            cta="Je suis tout ouïe",
            closing="Sahten !"
        )
    
    def generate_redirect(self, suggestion: Optional[str] = None) -> RecipeNarrative:
        """Generate redirect response for off-topic/blocked."""
        return RecipeNarrative(
            hook="Hmm, ce n'est pas vraiment mon domaine !",
            cultural_context="Mais au Liban, on dit que tout se règle autour d'un bon repas. La cuisine libanaise, c'est plus qu'une nourriture — c'est un art de vivre.",
            teaser=suggestion or "Que dirais-tu d'un délicieux mezze ?",
            cta="Laisse-moi te faire découvrir nos recettes",
            closing="Sahten !"
        )
    
    def generate_clarification(self, term: str) -> RecipeNarrative:
        """Generate clarification response for ingredient/term questions."""
        clarifications = {
            "zaatar": "Le zaatar est un mélange d'herbes (thym, origan, sumac, sésame) typiquement libanais. On le tartine sur du pain avec de l'huile d'olive pour les manaïch.",
            "sumac": "Le sumac est une épice pourpre au goût citronné, indispensable sur le fattouch et les grillades libanaises.",
            "tahini": "Le tahini est une pâte de sésame crémeuse, base du houmous et de nombreuses sauces. Son goût légèrement amer équilibre les plats.",
            "labneh": "Le labneh est un fromage frais crémeux obtenu en égouttant du yaourt. Incontournable du petit-déjeuner libanais !",
            "boulgour": "Le boulgour est du blé concassé précuit, base du taboulé et du kebbé. Plus nutritif que le riz.",
        }
        
        term_lower = term.lower() if term else ""
        context = None
        for key, value in clarifications.items():
            if key in term_lower:
                context = value
                break
        
        if context:
            return RecipeNarrative(
                hook=f"Ah, le {term.split()[-1] if term else 'terme'} ! Une merveille de la cuisine libanaise.",
                cultural_context=context,
                teaser="Tu veux une recette qui utilise cet ingrédient ?",
                cta="Je peux te montrer comment l'utiliser",
                closing="Sahten !"
            )
        
        return RecipeNarrative(
            hook="Bonne question !",
            cultural_context="La cuisine libanaise a beaucoup d'ingrédients spécifiques. Donne-moi plus de détails et je t'explique.",
            teaser="Ou je peux te suggérer une recette !",
            cta="Dis-moi ce que tu veux découvrir",
            closing="Sahten !"
        )

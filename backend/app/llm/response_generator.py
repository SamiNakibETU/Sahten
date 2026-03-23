"""
Response Generator V9.0
=======================

- Structured Outputs (json_schema strict)
- Prompts : consignes positives, liste d'interdits minimale (petits modeles)
- Ton : conseiller culinaire, vouvoiement, serviable
"""

import json
import logging
import re
from typing import Any, List, Optional

from openai import AsyncOpenAI
from unidecode import unidecode

from ..core.config import get_settings
from ..schemas.query_analysis import QueryAnalysis
from ..schemas.responses import (
    RecipeNarrative,
    RecipeCard,
    SharedIngredientProof,
    EvidenceBundle,
)

logger = logging.getLogger(__name__)

# Accroche contractuelle produit (recette demandée absente + alternative prouvée).
EXACT_ALTERNATIVE_HOOK = (
    "Je suis désolé, mais je n'ai pas cette recette dans mes carnets. "
    "Mais pour me faire pardonner je peux te proposer"
)

RESPONSE_JSON_SCHEMA = {
    "name": "sahten_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "hook": {
                "type": "string",
                "description": "Pour response_type not_found_with_alternative : le serveur remplace par la phrase contractuelle EXACT_ALTERNATIVE_HOOK ; vous pouvez mettre un placeholder court. Sinon : accroche au vous, 1 phrase.",
            },
            "detail": {
                "type": "string",
                "description": "1-2 phrases, vous. Si alternative : 1re phrase = titre exact (selected_recipes[0].title) + auteur (chef/cheffe selon le prenom — ex. Carla, Liza, Tara, Yasmina = cheffe femme, ne pas ecrire « un chef ») ; 2e phrase = lien concret avec la demande. Sinon : fait utile. Donnees JSON uniquement.",
            },
            "cta": {
                "type": "string",
                "description": "Invitez a consulter la recette sur L'Orient-Le Jour (vous).",
            },
        },
        "required": ["hook", "detail", "cta"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """\
# Role & objectif

Vous etes Sahten, conseiller culinaire pour L'Orient-Le Jour.
Vous orientez vers des recettes publiees sur lorientlejour.com.
Objectif : reponse courte, utile, professionnelle et chaleureuse — un detail concret qui donne envie d'ouvrir la fiche.

# Langue & ton (obligatoire)

- Adressez-vous avec **vous** dans `detail` et `cta` (pas de tutoiement), sauf la phrase d'accroche **alternative** fixe imposee par le produit (voir plus bas — elle contient « te proposer »).
- Style : expert accessible, serviable, phrases courtes.

# Ce que vous faites (regles positives)

- Une a deux phrases dans `detail`, puis `cta`. Pas de long developpement.
- Ancrez-vous dans **selected_recipes** et **cited_passage** : noms des plats, auteurs, ingredients ou techniques mentionnes dans les donnees.
- **Accord chef / cheffe** : si le champ `chef` contient un prenom typiquement feminin (Carla, Liza, Tara, Yasmina, Joanna, Lea, Emilie, etc.), ecrivez **cheffe** ou tournure neutre (« la recette de Carla Rebeiz », « proposee par Carla Rebeiz ») — **interdit** : « un chef » pour une femme. Si le genre est ambigu, forme neutre sans « chef » : « par [Nom complet] ».
- Si la demande est vague (« rapide », « leger »), reliez-la a un fait precis de la recette (temps, nombre d'ingredients, type de cuisson).
- Si plusieurs recettes : une phrase courte par fiche, chaque phrase commence par le titre (ou le chef) pour que le lecteur fasse le lien avec les cartes.
- **Alternative** (plat demande absent des donnees) : le `hook` est **remplace cote serveur** par la phrase exacte (accents compris) : « Je suis desole, mais je n'ai pas cette recette dans mes carnets. Mais pour me faire pardonner je peux te proposer » — dans `detail` vous enchainez tout de suite avec la **recette libanaise** (titre + **cheffe/chef** correct selon le prenom, voir regle ci-dessus), ingredient principal en commun, lien concret. Pas d'encyclopedie sur le plat demande. **Ne commencez pas** le `detail` par une autre accroche type « Je vous propose » sans avoir laisse le serveur afficher le hook contractuel (le hook dans le JSON est ignore pour le texte final mais le `detail` doit suivre logiquement).

# Rester factuel (anti-hallucination)

- Ne citez ni recette, ni chef, ni ingredient qui ne figurent pas dans le JSON utilisateur.
- N'inventez pas d'accompagnement ni d'etape : seulement ce qui est dans les donnees.

# A eviter (liste courte — les petits modeles saturent avec de longues interdictions)

- Generalites pays : formulations du type « Au Liban, on… », « La cuisine libanaise regorge… » — preferez parler de **la fiche** ou du **chef**.
- Cours sur le plat mondial demande s'il n'est pas dans selected_recipes (pas d'article Wikipedia).
- Etiqueter un plat international comme « classique libanais » s'il ne l'est pas.
- Adjectifs creux sans fait : authentique, inoubliable, explosion de saveurs, experience culinaire.

# Format de sortie

JSON strict : `hook`, `detail`, `cta` — tous les champs en francais avec vouvoiement.

# Exemples (tous au vous)

## Plat exact
User: {"user_query": "taboule", "selected_recipes": [{"title": "Le vrai taboule de Kamal Mouzawak", "chef": "Kamal Mouzawak"}]}
Response: {"hook": "Pour le taboule, voici une reference du site.", "detail": "Kamal Mouzawak met le persil au centre : peu de boulgour, beaucoup d'herbes fraiches — c'est sa signature.", "cta": "Retrouvez la recette sur L'Orient-Le Jour"}

## Plat exact + facile
User: {"user_query": "recette facile a faire", "selected_recipes": [{"title": "Les crevettes guacamole a la libanaise de Liza Asseily", "chef": "Liza Asseily"}]}
Response: {"hook": "Voici une option simple sur le site.", "detail": "Les crevettes guacamole de Liza Asseily : peu de cuisson, avocat et crevettes au citron et a la grenade — les etapes sont dans l'article.", "cta": "Consultez la recette sur L'Orient-Le Jour"}

## Alternative
User: {"user_query": "blanquette de veau", "selected_recipes": [{"title": "La Mouloukhiye de Tara Khattar", "chef": "Tara Khattar"}], "shared_ingredient_proof": {"shared_ingredients": ["viande mijotee"]}}
Response: {"hook": "Je suis désolé, mais je n'ai pas cette recette dans mes carnets. Mais pour me faire pardonner je peux te proposer", "detail": "La mouloukhiye de Tara Khattar : viandes mijotees longtemps dans une sauce aux feuilles de mouloukhiye — meme logique de cuisson lente que votre blanquette.", "cta": "La recette est sur L'Orient-Le Jour"}

## Alternative etrangere
User: {"user_query": "fajitas", "selected_recipes": [{"title": "Le poulet citron zaatar de Carla Rebeiz", "chef": "Carla Rebeiz"}], "shared_ingredient_proof": {"shared_ingredients": ["poulet"]}}
Response: {"hook": "Je suis désolé, mais je n'ai pas cette recette dans mes carnets. Mais pour me faire pardonner je peux te proposer", "detail": "Le poulet citron zaatar de la cheffe Carla Rebeiz : poulet marine puis grille — autre assaisonnement (zaatar, citron), meme type de cuisson.", "cta": "Decouvrez la fiche sur L'Orient-Le Jour"}

## Requete vague
User: {"user_query": "un plat facile a faire", "selected_recipes": [{"title": "Les courgettes aux tomates de teta de Karim Haidar", "chef": "Karim Haidar"}]}
Response: {"hook": "Voici un plat peu charge en technique.", "detail": "Les courgettes aux tomates de teta de Karim Haidar : trois legumes, une poele, recette de famille telle qu'elle est publiee.", "cta": "Les etapes sont sur L'Orient-Le Jour"}

## Menu
User: {"user_query": "un menu libanais", "intent": "menu_composition", "recipe_count": 3, "selected_recipes": [{"title": "Salade fattouch", "category": "entree"}, {"title": "Daoud Bacha", "category": "plat_principal"}, {"title": "Maamoul", "category": "dessert"}]}
Response: {"hook": "Proposition de menu en trois temps :", "detail": "Entree : fattouch, salade au sumac. Plat : Daoud Bacha, boulettes sauce tomate. Dessert : maamoul fourres aux noix ou pistaches.", "cta": "Chaque recette est sur L'Orient-Le Jour"}
"""

ALTERNATIVE_SYSTEM_PROMPT = """\
# Role

Vous etes Sahten, conseiller culinaire L'Orient-Le Jour. Le plat demande n'est pas dans les donnees : vous proposez une **recette libanaise publiee** qui a du sens par rapport a la demande (au moins un **ingredient principal** en commun — voir shared_ingredient_proof).

# Accroche obligatoire (hook)

Le champ `hook` doit etre **exactement** cette phrase (copiez-la telle quelle, avec accents) :
« Je suis désolé, mais je n'ai pas cette recette dans mes carnets. Mais pour me faire pardonner je peux te proposer »

# Contraintes absolues pour `detail`

1. PREMIERE phrase : titre exact de la recette proposee + auteur (genre correct : prenom feminin → cheffe / « par [Prenom Nom] » ; masculin ou inconnu → chef / « par [Nom] »).
   Formule : « [Titre] de [Auteur] : » (ou equivalent avec chef/cheffe).
   JAMAIS commencer par « Tu veux », « Je te propose », « En cuisine », « Au Liban », ni par une phrase sur le plat demande avant le titre OLJ.

2. DEUXIEME phrase : lien concret avec la demande de l'utilisateur.
   Utiliser UNIQUEMENT les donnees de `shared_ingredient_proof.shared_ingredients` et le texte de `cited_passage` (si present dans le JSON utilisateur) pour formuler ce lien.
   JAMAIS inventer ni utiliser vos connaissances generales sur le plat demande (origine, histoire, pays).

3. INTERDICTIONS absolues dans `detail` :
   - Toute mention du pays ou region d'origine du plat demande (Mexique, Italie, Inde, etc.)
   - Les formules : « on aime », « on retrouve », « cuisine libanaise permet », « variation », « inspiration », « fusion », « invitation au voyage »
   - Tout ce qui ne vient pas des champs `selected_recipes`, `shared_ingredient_proof` ou `cited_passage`

`detail` et `cta` : vouvoiement (« vous », « votre »).

# Exemple MAUVAIS (a eviter)

detail: « Le veau est souvent mijote en cuisine libanaise... Meme si ce n'est pas exactement... »

# Exemple CORRECT

detail: « La mouloukhiye de Tara Khattar : poulet mijote dans une sauce aux feuilles de mouloukhiye — meme logique de cuisson lente que votre blanquette de veau. »

# Sortie

JSON strict : hook, detail, cta.

# Exemples JSON

User: {"user_query": "hachis parmentier ?", "selected_recipes": [{"title": "Les fusilli a la creme de citron de Sabyl Ghoussoub", "chef": "Sabyl Ghoussoub"}], "shared_ingredient_proof": {"shared_ingredients": ["creme", "plat"]}}
Response: {"hook": "Je suis désolé, mais je n'ai pas cette recette dans mes carnets. Mais pour me faire pardonner je peux te proposer", "detail": "Les fusilli a la creme de citron de Sabyl Ghoussoub : pates en sauce cremeuse — registre reconfortant proche de votre idee, sans viande hachee comme dans un hachis.", "cta": "Retrouvez la recette sur L'Orient-Le Jour"}

User: {"user_query": "raclette", "selected_recipes": [{"title": "La raclette au celeri-rave facon tartiflette du chef Youssef Akiki", "chef": "Youssef Akiki"}], "shared_ingredient_proof": {"shared_ingredients": ["fromage"]}}
Response: {"hook": "Je suis désolé, mais je n'ai pas cette recette dans mes carnets. Mais pour me faire pardonner je peux te proposer", "detail": "La raclette au celeri-rave facon tartiflette du chef Youssef Akiki : fromage fondu au four sur legumes — meme principe que la raclette, avec celeri-rave a la place des pommes de terre.", "cta": "Consultez la fiche sur L'Orient-Le Jour"}
"""


def _normalize_for_banned_substrings(text: str) -> str:
    """Uniformise ponctuation / accents pour detecter les tournures interdites (ex. « Au Liban, on aime »)."""
    t = unidecode(text.lower())
    t = re.sub(r"[''`´]+", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# Filet leger (petits modeles : preferer consignes positives dans le prompt).
# Les gardes principales restent : citation du titre, longueur max, detail alternatif.
BANNED_PATTERNS = (
    "la cuisine libanaise regorge",
    "au liban on",
    "en liban on",
    "classique libanais",
    "typique du liban",
    "tu as envie",
    "tu veux",
    "tu cherches",
    "je te comprends",
    "bien que d origine",
    "originaire de",
    "originaires de",
    "explosion de saveurs",
    "experience culinaire",
    "laisse toi tenter",
    "meme si la recette",
    "une autre inspiration",
    "envie d un plat",
    "te transportera",
    "je te propose une recette",
    "plat savoureux",
    "veritable classique",
    "parfait pour",
    "ideal pour",
)


class ResponseGenerator:
    """Single LLM call with Structured Outputs + fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = settings.openai_api_key if api_key is None else api_key
        self.model = model or settings.openai_model
        self.fallback_model = fallback_model or settings.fallback_model
        self.client = AsyncOpenAI(api_key=self.api_key, timeout=12.0)

    def _build_user_payload(
        self, evidence: EvidenceBundle, analysis: Optional[QueryAnalysis] = None
    ) -> str:
        payload: dict[str, Any] = {
            "user_query": evidence.user_query,
            "response_type": evidence.response_type,
            "intent_detected": evidence.intent_detected,
            "selected_recipes": [
                {
                    "title": r.title,
                    "chef": r.chef,
                    "category": r.category,
                    "cited_passage": (r.cited_passage[:200] if r.cited_passage else None),
                }
                for r in evidence.selected_recipe_cards
            ],
        }
        if analysis is not None:
            payload["intent"] = analysis.intent
            payload["mood_tags"] = list(analysis.mood_tags or [])
            payload["recipe_count"] = analysis.recipe_count
            payload["category"] = analysis.category
            payload["dish_name"] = analysis.dish_name
        if evidence.shared_ingredient_proof:
            payload["shared_ingredient_proof"] = {
                "shared_ingredients": evidence.shared_ingredient_proof.shared_ingredients[:4],
                "match_reason": evidence.match_reason,
            }
        return json.dumps(payload, ensure_ascii=False)

    async def _call_llm(
        self,
        system: str,
        user_content: str,
        *,
        model_override: Optional[str] = None,
        max_tokens: int = 220,
    ) -> dict:
        model = model_override or self.model
        resp = await self.client.chat.completions.create(
            model=model,
            response_format={
                "type": "json_schema",
                "json_schema": RESPONSE_JSON_SCHEMA,
            },
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.25,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    def _data_to_narrative(self, data: dict, is_alternative: bool = False) -> RecipeNarrative:
        detail = data.get("detail", "")
        cta = data.get("cta", "Consultez la recette sur L'Orient-Le Jour")
        if is_alternative:
            # Phrase contractuelle produit (independante de la sortie LLM).
            hook = EXACT_ALTERNATIVE_HOOK
        else:
            hook = data.get("hook", "Voici une suggestion issue de nos recettes publiées.")
        return RecipeNarrative(
            hook=hook,
            cultural_context=detail,
            teaser=None,
            cta=cta,
            closing="Sahten !",
        )

    def _validate(
        self,
        narrative: RecipeNarrative,
        evidence: EvidenceBundle,
        analysis: Optional[QueryAnalysis] = None,
    ) -> bool:
        raw = " ".join(
            p for p in [narrative.hook, narrative.cultural_context or "", narrative.cta] if p
        )
        text = unidecode(raw.lower())
        ban_space = _normalize_for_banned_substrings(raw)
        n_cards = len(evidence.selected_recipe_cards)
        max_chars = 900 if n_cards > 1 or evidence.response_type == "menu" else 560
        if len(text) > max_chars:
            return False
        if any(pat in ban_space for pat in BANNED_PATTERNS):
            logger.info("Validation failed: banned pattern found in: %s", ban_space[:120])
            return False
        if evidence.response_type == "not_found_with_alternative":
            if narrative.hook.strip() != EXACT_ALTERNATIVE_HOOK.strip():
                logger.info("Validation failed: alternative hook must match EXACT_ALTERNATIVE_HOOK")
                return False
        if not evidence.selected_recipe_cards:
            return True
        if n_cards <= 1:
            card0 = evidence.selected_recipe_cards[0]
            title_words = [
                w
                for w in unidecode((card0.title or "").lower()).split()
                if len(w) >= 4
            ]
            # Alternative : le plat propose doit etre nomme dans le detail, pas seulement dans le CTA.
            if evidence.response_type == "not_found_with_alternative" and title_words:
                detail_l = unidecode((narrative.cultural_context or "").lower())
                if not any(w in detail_l for w in title_words[:6]):
                    logger.info(
                        "Validation failed: alternative detail must cite proposed recipe (title tokens)"
                    )
                    return False
                first_sentence = detail_l.split(".")[0]
                if not any(w in first_sentence for w in title_words[:3]):
                    logger.info(
                        "Validation failed: detail first sentence must contain recipe title (title tokens)"
                    )
                    return False
            if title_words and not any(w in text for w in title_words[:3]):
                return False
            return True
        for card in evidence.selected_recipe_cards:
            title_l = unidecode((card.title or "").lower())
            words = [w for w in title_l.split() if len(w) >= 3][:6]
            chef_ok = False
            if card.chef:
                chef_part = unidecode(card.chef.lower()).split()[0]
                chef_ok = len(chef_part) >= 3 and chef_part in text
            if not (chef_ok or any(w in text for w in words[:4])):
                return False
        return True

    async def _generate_with_fallback(
        self,
        system: str,
        evidence: EvidenceBundle,
        fallback_fn,
        is_alternative: bool = False,
        analysis: Optional[QueryAnalysis] = None,
    ) -> RecipeNarrative:
        if not self.api_key:
            return fallback_fn()

        user_content = self._build_user_payload(evidence, analysis)
        n_cards = len(evidence.selected_recipe_cards)
        max_tok = 360 if n_cards > 1 or evidence.response_type == "menu" else 220

        for model_name in [self.model, self.fallback_model]:
            try:
                data = await self._call_llm(
                    system,
                    user_content,
                    model_override=model_name,
                    max_tokens=max_tok,
                )
                narrative = self._data_to_narrative(data, is_alternative)
                if self._validate(narrative, evidence, analysis):
                    return narrative
                logger.warning("Validation failed for model %s, trying next", model_name)
            except Exception as e:
                logger.error("LLM call failed (%s): %s", model_name, e)

        return fallback_fn()

    # ── Public API ──────────────────────────────────────────────────

    async def generate_narrative(
        self,
        user_query: str,
        analysis: QueryAnalysis,
        recipes: List[RecipeCard],
        is_base2_fallback: bool = False,
        session_context: Optional[dict] = None,
    ) -> RecipeNarrative:
        take = min(8, len(recipes))
        cards = recipes[:take]
        if analysis.intent == "menu_composition":
            response_type = "menu"
        elif is_base2_fallback:
            response_type = "recipe_base2"
        else:
            response_type = "recipe_olj"

        evidence = EvidenceBundle(
            response_type=response_type,
            intent_detected=analysis.intent,
            user_query=user_query,
            selected_recipe_cards=cards,
            allowed_claims=[],
            forbidden_claims=[],
            session_context=session_context or {},
        )
        return await self._generate_with_fallback(
            SYSTEM_PROMPT,
            evidence,
            lambda: self._fallback_narrative(user_query, cards),
            analysis=analysis,
        )

    async def generate_proven_alternative_narrative(
        self,
        *,
        user_query: str,
        alternative: RecipeCard,
        proof: SharedIngredientProof,
        session_context: Optional[dict] = None,
        match_reason: Optional[str] = None,
    ) -> RecipeNarrative:
        evidence = EvidenceBundle(
            response_type="not_found_with_alternative",
            intent_detected="recipe_specific",
            user_query=user_query,
            selected_recipe_cards=[alternative],
            shared_ingredient_proof=proof,
            match_reason=match_reason,
            allowed_claims=[],
            forbidden_claims=[],
            session_context=session_context or {},
        )
        return await self._generate_with_fallback(
            ALTERNATIVE_SYSTEM_PROMPT,
            evidence,
            lambda: self._fallback_alternative(user_query, alternative, proof),
            is_alternative=True,
            analysis=None,
        )

    # ── Deterministic fallbacks ─────────────────────────────────────

    def _fallback_narrative(
        self,
        user_query: str,
        recipes: List[RecipeCard],
    ) -> RecipeNarrative:
        if not recipes:
            return RecipeNarrative(
                hook="Je n'ai pas cette recette dans mes fiches.",
                cultural_context=(
                    "Indiquez un plat, un ingrédient ou une contrainte (temps, régime) "
                    "et je relance une recherche ciblée."
                ),
                teaser=None,
                cta="Parcourez les recettes sur L'Orient-Le Jour",
                closing="Sahten !",
            )
        if len(recipes) > 1:
            parts = []
            for r in recipes[:6]:
                t = r.title or "Recette"
                c = f" ({r.chef})" if r.chef else ""
                parts.append(f"{t}{c}.")
            return RecipeNarrative(
                hook="Voici la sélection :",
                cultural_context=" ".join(parts),
                teaser=None,
                cta="Chaque recette est sur L'Orient-Le Jour",
                closing="Sahten !",
            )
        r = recipes[0]
        title = r.title or "cette recette"
        chef = r.chef or ""
        chef_part = f" de {chef}" if chef else ""
        return RecipeNarrative(
            hook=f"Voici {title}{chef_part}.",
            cultural_context="Les étapes détaillées sont dans l'article.",
            teaser=None,
            cta="Consultez la recette sur L'Orient-Le Jour",
            closing="Sahten !",
        )

    def _fallback_alternative(
        self,
        user_query: str,
        alternative: RecipeCard,
        proof: SharedIngredientProof,
    ) -> RecipeNarrative:
        shared = [i for i in proof.shared_ingredients if i]
        shared_label = ", ".join(shared[:2]) if shared else proof.normalized_ingredient or ""
        alt_title = alternative.title or "cette recette"
        chef = alternative.chef or ""
        chef_part = f" de {chef}" if chef else ""
        q_short = (user_query or "").strip()
        if len(q_short) > 100:
            q_short = q_short[:97] + "…"
        recipe_label = f"{alt_title}{chef_part}"
        demande = f"Concernant « {q_short} »" if q_short else "Pour votre recherche"
        return RecipeNarrative(
            hook=EXACT_ALTERNATIVE_HOOK,
            cultural_context=(
                f"{recipe_label} — {demande}, point commun avec votre idée : {shared_label}."
                if shared_label
                else f"{recipe_label} — {demande}, les précisions sont dans la fiche."
            ),
            teaser=None,
            cta="Consultez cette recette sur L'Orient-Le Jour",
            closing="Sahten !",
        )

    # ── Static responses (no LLM) ──────────────────────────────────

    def generate_greeting(self) -> RecipeNarrative:
        return RecipeNarrative(
            hook="Je m'appelle Sahtein, je suis le chef robot de L'Orient-Le Jour.",
            cultural_context=(
                "Si vous cherchez une recette libanaise, je veux vous aider : "
                "dites-moi ce que vous voulez manger (plat précis, ingrédient, envie rapide ou végétarienne…)."
            ),
            teaser="Exemples : taboulé, houmous, « recette au poulet ».",
            cta="Je vous écoute",
            closing="Sahten !",
        )

    def generate_hors_cuisine_libanaise(
        self, *, user_query: str, requested_dish: Optional[str] = None
    ) -> RecipeNarrative:
        label = (requested_dish or user_query or "ça").strip()
        return RecipeNarrative(
            hook=f"Pas de {label} dans mes fiches.",
            cultural_context=(
                "Je me concentre sur la cuisine libanaise publiée dans L'Orient-Le Jour "
                "(mezze, plats, desserts du pays)."
            ),
            teaser="Proposez une autre envie et je cherche dans le corpus.",
            cta="Parcourez la rubrique Cuisine sur L'Orient-Le Jour",
            closing="Sahten !",
        )

    def generate_about_bot(self) -> RecipeNarrative:
        return RecipeNarrative(
            hook=(
                "Je suis Sahtein, le robot de L'Orient-Le Jour à votre service pour toutes les questions "
                "relatives à la cuisine libanaise, ou aux recettes (libanaises ou non) que des chefs libanais, "
                "professionnels ou amateurs, ont bien voulu partager avec nous."
            ),
            cultural_context=(
                "Je m'appuie sur les fiches publiées sur lorientlejour.com pour des réponses courtes, "
                "utiles et fidèles aux articles."
            ),
            teaser="Demandez un plat, un ingrédient ou un type de menu.",
            cta="Parcourez les recettes sur L'Orient-Le Jour",
            closing="Sahten !",
        )

    def generate_redirect(self, suggestion: Optional[str] = None) -> RecipeNarrative:
        return RecipeNarrative(
            hook="Cette question relève d'autres domaines que celui de la cuisine — ce n'est pas mon expertise.",
            cultural_context=(
                "Si vous le souhaitez, revenons aux recettes de L'Orient-Le Jour : un plat, un ingrédient ou un menu."
            ),
            teaser=suggestion or "Un mezze, un plat, un dessert ?",
            cta="Proposez un sujet autour de la cuisine",
            closing="Sahten !",
        )

    def generate_clarification(
        self, term: str, grounding: Optional[tuple[str, str, str]] = None
    ) -> RecipeNarrative:
        if grounding:
            excerpt, title, url = grounding
            return RecipeNarrative(
                hook=f"D'après nos articles : {title}",
                cultural_context=excerpt[:400] + ("…" if len(excerpt) > 400 else ""),
                teaser="Souhaitez-vous une recette qui utilise cet ingrédient ?",
                cta="Lisez l'article complet sur L'Orient-Le Jour",
                closing="Sahten !",
            )

        clarifications = {
            "zaatar": "Le zaatar, c'est thym, origan, sumac et sésame. On le tartine sur du pain avec de l'huile d'olive pour les manaïch.",
            "sumac": "Le sumac, c'est une épice pourpre au goût citronné. Indispensable sur le fattouch et les grillades.",
            "tahini": "Le tahini, c'est de la pâte de sésame. Base du houmous et de beaucoup de sauces libanaises.",
            "labneh": "Le labneh, c'est du yaourt égoutté — crémeux, un peu acide. On le mange au petit-déj avec de l'huile d'olive.",
            "boulgour": "Le boulgour, c'est du blé concassé précuit. Base du taboulé et du kebbé.",
        }

        term_lower = term.lower() if term else ""
        for key, value in clarifications.items():
            if key in term_lower:
                return RecipeNarrative(
                    hook=f"Pour {term.split()[-1] if term else 'ce terme'}, voici une synthèse.",
                    cultural_context=value,
                    teaser="Souhaitez-vous une recette qui l'utilise ?",
                    cta="Découvrez nos recettes sur L'Orient-Le Jour",
                    closing="Sahten !",
                )

        return RecipeNarrative(
            hook="Bonne question !",
            cultural_context="Précisez un peu votre demande et je vous réponds.",
            teaser=None,
            cta="Indiquez ce que vous souhaitez découvrir",
            closing="Sahten !",
        )

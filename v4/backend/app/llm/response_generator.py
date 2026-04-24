"""Génération de réponse Sahteïn avec grounding phrase-par-phrase.

Format de sortie strict (JSON schema) :
    {
      "answer_sentences": [
        {"text": "...", "source_chunk_ids": [42, 17]},
        ...
      ],
      "recipe_card": { ... } | null,
      "recipe_card_secondary": { ... } | null,
      "chef_card": { ... } | null,
      "follow_up": "...",
      "confidence": 0.83
    }

Chaque phrase de la réponse DOIT citer au moins un chunk_id parmi ceux fournis
en contexte. Toute phrase qui n'a pas de citation est rejetée par le validateur
post-LLM. Cela rend l'hallucination structurellement plus difficile.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

import structlog

from ..settings import get_settings
from ..rag.reranker import RerankedHit

log = structlog.get_logger(__name__)


class GroundedSentence(BaseModel):
    text: str
    source_chunk_ids: list[int] = Field(default_factory=list)


class RecipeCard(BaseModel):
    title: str
    chef: str | None = None
    duration_min: int | None = None
    serves: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    source_chunk_ids: list[int] = Field(default_factory=list)


class ChefCard(BaseModel):
    name: str
    role: str | None = None
    biography: str | None = None
    works: list[str] = Field(default_factory=list)
    source_chunk_ids: list[int] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    answer_sentences: list[GroundedSentence]
    recipe_card: RecipeCard | None = None
    recipe_card_secondary: RecipeCard | None = None
    chef_card: ChefCard | None = None
    follow_up: str = ""
    confidence: float = 0.0


JSON_SCHEMA: dict[str, Any] = {
    "name": "grounded_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer_sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "source_chunk_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["text", "source_chunk_ids"],
                },
            },
            "recipe_card": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "chef": {"type": ["string", "null"]},
                            "duration_min": {"type": ["integer", "null"]},
                            "serves": {"type": ["string", "null"]},
                            "ingredients": {
                                "type": "array", "items": {"type": "string"}
                            },
                            "steps": {
                                "type": "array", "items": {"type": "string"}
                            },
                            "source_chunk_ids": {
                                "type": "array", "items": {"type": "integer"}
                            },
                        },
                        "required": [
                            "title", "chef", "duration_min", "serves",
                            "ingredients", "steps", "source_chunk_ids",
                        ],
                    },
                ]
            },
            "recipe_card_secondary": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "chef": {"type": ["string", "null"]},
                            "duration_min": {"type": ["integer", "null"]},
                            "serves": {"type": ["string", "null"]},
                            "ingredients": {
                                "type": "array", "items": {"type": "string"}
                            },
                            "steps": {
                                "type": "array", "items": {"type": "string"}
                            },
                            "source_chunk_ids": {
                                "type": "array", "items": {"type": "integer"}
                            },
                        },
                        "required": [
                            "title", "chef", "duration_min", "serves",
                            "ingredients", "steps", "source_chunk_ids",
                        ],
                    },
                ]
            },
            "chef_card": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": ["string", "null"]},
                            "biography": {"type": ["string", "null"]},
                            "works": {
                                "type": "array", "items": {"type": "string"}
                            },
                            "source_chunk_ids": {
                                "type": "array", "items": {"type": "integer"}
                            },
                        },
                        "required": [
                            "name", "role", "biography", "works",
                            "source_chunk_ids",
                        ],
                    },
                ]
            },
            "follow_up": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": [
            "answer_sentences", "recipe_card", "recipe_card_secondary",
            "chef_card", "follow_up", "confidence",
        ],
    },
}


SYSTEM_PROMPT = """Tu es Sahteïn, un assistant culinaire libanais incarné dans
le ton et l'esprit éditorial de L'Orient-Le Jour.

Règles ABSOLUES :
1. Tu ne dois JAMAIS inventer de fait. Toute affirmation doit être appuyée
   par au moins un `chunk_id` parmi le CONTEXTE fourni.
2. Si le contexte ne contient pas la réponse, dis-le honnêtement et propose
   une question de clarification dans `follow_up`.
3. La réponse est en français soutenu mais accessible, sans emojis.
4. Ne cite pas les `chunk_id` dans le texte ; mets-les uniquement dans le
   tableau `source_chunk_ids` de chaque phrase.
5. Si la requête concerne un chef, remplis `chef_card` avec ses informations
   confirmées (et seulement celles-là). Idem pour `recipe_card`.
6. `confidence` ∈ [0,1] reflète à quel point le contexte couvre la question.
   < 0.4 si tu manques d'info ; > 0.8 si tu as plusieurs sources concordantes.
7. Chaque phrase ET chaque carte (`recipe_card`, `recipe_card_secondary`,
   `chef_card`) DOIVENT lister dans `source_chunk_ids` uniquement des IDs
   présents dans le CONTEXTE (copier les numéros exactement depuis les lignes
   [chunk_id=...]). Les `source_chunk_ids` d’une carte doivent appartenir à
   **un seul article** (même `article=` dans le CONTEXTE).
8. Ne dis jamais qu'il n'y a « plus » de recettes, « plus rien », ou « pour
   l'instant » dans le répertoire si tu proposes encore une recette ou un plat
   dans la même réponse : c'est contradictoire. Soit tu en proposes une, soit
   tu expliques honnêtement les limites du corpus — pas les deux.
9. Pour une question répétée (même ingrédient), privilégie un AUTRE article
   du contexte que celui déjà évoqué si le contexte le permet ; sinon dis
   clairement que les archives ne montrent pas d'autre fiche pour cette
   requête. Si les lignes du CONTEXTE portent sur **au moins deux** numéros
   `article=` distincts, il existe d’autre(s) fiche(s) dans ce contexte : ne
   dis pas qu’il n’y a « aucune autre recette » sur cet ingrédient sans
   t’appuyer sur **un** de ces extraits (carte + citations).
10. **Trafic vers l'article (obligatoire pour les recettes)** : le widget ne doit
   PAS reproduire la recette complète (ni liste d'ingrédients détaillée, ni étapes
   numérotées copiées du contexte). Objectif : donner envie d'ouvrir la fiche sur
   L'Orient-Le Jour. Dans `answer_sentences`, 2 à 4 phrases maximum : accroche
   (ambiance, occasion, idée du plat), sans recopier les quantités ni la marche
   à suivre. Dans chaque carte recette, mets **`ingredients` et `steps` à des
   tableaux vides `[]`** — le titre, chef, durée, portions peuvent rester si utiles ;
   le lien article est affiché par l'interface.
11. N'utilise JAMAIS les formulations « mon répertoire », « plus d'autres recettes
   pour l'instant », « explorez nos recettes sur L'Orient-Le Jour » : tu n'es
   pas un site web, tu cites des extraits d'archives. Si le CONTEXTE est vide ou
   insuffisant, dis-le sans inventer de politesse marketing.
12. Si la requête est une demande de recette, termine souvent `follow_up` par une
   question qui ouvre vers un autre plat ou une variante, sans redonner la recette.
13. Si tu expliques que les archives **ne contiennent pas** de réponse adaptée,
   qu’aucune recette ne correspond, ou que le contenu est **insuffisant** pour la
   demande : mets **`source_chunk_ids: []` sur chaque phrase**, laisse
   `recipe_card`, `recipe_card_secondary` et `chef_card` à **null**, et mets
   `confidence` ≤ 0.35.
   **Ne cite aucun chunk** dans ce cas — sinon la liste « Sources » afficherait des
   articles non pertinents à côté d’un texte qui dit qu’il n’y a pas de fiche.
14. Si un **HISTORIQUE DU CHAT** est fourni et que la QUESTION ACTUELLE est courte
   (« oui », « non », « la première », « autre chose », etc.), **rattache-la au
   dernier échange** (ta proposition précédente et la question de l’utilisateur).
   Ne réponds pas que la question est « incomplète » ou « ambiguë » si l’historique
   permet de la comprendre. Utilise aussi **tes relances précédentes** (ex. fattouche,
   salade libanaise) : si l’utilisateur semble y répondre, ne repose pas la même
   question mot pour mot ; fais évoluer `follow_up` en cohérence avec le fil.
15. **Une seconde recette = une seconde carte + citations strictes** : si tu
   évoques **deux articles ou deux plats distincts** présents dans le CONTEXTE,
   mets la première fiche dans `recipe_card` et la seconde dans
   `recipe_card_secondary` (titres et `source_chunk_ids` alignés sur **leur**
   extrait chacun). Chaque phrase du texte qui parle du plat A doit avoir des
   `source_chunk_ids` issus de l’article A ; idem pour le plat B. **Interdit** de
   mélanger dans une même phrase deux chefs ou deux plats sans citer les **bons**
   chunks pour chacun.
16. Si le CONTEXTE ne contient **qu’un seul** article pertinent, ne décris **pas**
   une autre recette, un autre chef ou une « variante » nommée qui ne figure pas
   dans les extraits : pas de `recipe_card_secondary`, et pas de mention
   détaillée hors corpus.
17. **Recette demandée absente, alternative pertinente** : si l'utilisateur cite un
   **nom de plat ou de fiche très précis** (ex. « ta recette de X ») et que ce
   plat **n'est pas** dans le CONTEXTE, mais qu'un autre extrait **reste réellement**
   pertinent (même ingrédient principal, même esprit), tu peux alors commencer par
   cette phrase **exacte** :
   « Je suis désolé, mais je n'ai pas cette recette dans mes carnets. Mais pour me faire pardonner je peux te proposer »
   puis présenter la recette retenue (chunks de **cet** article).
   **Interdiction stricte** d'utiliser cette accroche pour « une autre [recette] »,
   « encore », « autre idée » ou toute **continuation d'un ingrédient / thème**
   (ex. après « concombre ») : dans ce cas c'est « une fiche de plus sur le
   **même thème** » ou, si vraiment aucun autre extrait, une explication claire
   **sans** phrase « carnets ». Ne pivote pas vers un ingrédient non demandé
   (ex. petits pois) si l’HISTORIQUE indique clairement un autre fil (ex. concombre).
18. **Cohérence avec ta dernière relance** : si, dans l’HISTORIQUE, **tu as toi-même
   nommé** un plat (ex. fattouche) en fin de message comme piste, et
   l’utilisateur enchaîne dans ce sens (« une autre », « oui »…), ne dis
   **pas** qu’il n’existe **aucune** autre recette [ingrédient] **uniquement
   sur la base d’extraits** en contredisant cette piste. Vérifie d’abord le
   CONTEXTE : s’il contient un extrait lié à ce nom de plat, cite-le. S’il n’y
   en a **vraiment** pas dans les extraits, explique le gap **sans nier** que tu
   venais d’en parler, ni affirmer un vide absolu sur tout le site — en restant
   honnête sur la limite des seuls **chunks** vus ici.
19. **Premier tour, demande vaste (ex. un ingrédient)** : si le CONTEXTE mélange
   plusieurs fiches (sauces, salades, plats) **pertinentes** pour cette demande, ne
   mets **pas** en `recipe_card` une **sauce** par habitude quand la même recherche
   offre clairement une **salade**, entrée ou plat complet au même ingrédient : **varie
   le type** (privilégie un angle différent d’une simple sauce de dip), sauf si
   l’utilisateur a demandé explicitement une sauce.
20. **Enchaînements « une autre / encore une autre »** : d’après l’HISTORIQUE, note
   quels titres de fiches **tu as déjà proposés** en `recipe_card` / cartes. Pour le
   tour suivant, choisis d’abord une fiche d’un **autre registre** (salade, mezzé)
   alignée sur **ta dernière relance** (ex. fattouche) **si** des chunks le
   portent, avant un plat où l’ingrédient n’est qu’accessoire ou une recette qui
   **casse** le fil (ex. plat créatif hors piste salade). Le fil reste ingrédient +
   type de piste tant que les extraits le permettent.
21. **Objection « ce n’est pas le bon plat / pas l’ingrédient dedans »** : l’utilisateur
   reste dans la conversation (ex. fil **concombre**). Ne déclenche **pas** la
   règle 13 (tout vide) si le CONTEXTE contient encore des extraits mentionnant
   l’ingrédient ou une **salade** (fattouche, etc.) évoquée en relance. Réponds avec
   au moins **une phrase** citant un `chunk_id` qui montre une fiche **pertinente**
   (salade au concombre, autre mezze) — ou dis honnêtement que **ces seuls extraits**
   ne décrivent pas encore assez le plat, **sans** contredire que le site peut
   avoir d’autres fiches hors contexte.
"""


def _format_context(hits: list[RerankedHit]) -> str:
    lines = ["CONTEXTE — chaque entrée est un extrait d'article OLJ :"]
    for h in hits:
        title = (h.hit.article_title or "").replace("\n", " ")[:160]
        lines.append(
            f"\n[chunk_id={h.hit.chunk_id} | article={h.hit.article_external_id}"
            f" | titre={title}"
            f" | section={h.hit.section_kind} | score={h.rerank_score:.3f}]"
            f"\n{h.hit.chunk_text}"
        )
    return "\n".join(lines)


class ResponseGenerator:
    def __init__(self) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY manquant pour ResponseGenerator.")
        self._client = AsyncOpenAI(api_key=s.openai_api_key)
        self._model = s.llm_model
        self._temperature = s.llm_temperature

    async def generate(
        self,
        user_query: str,
        hits: list[RerankedHit],
        *,
        conversation_history: str | None = None,
        session_thread_summary: str | None = None,
    ) -> GroundedAnswer:
        if not hits:
            # Pas d'appel LLM : sans chunks le schéma JSON serait rempli d'inventions
            # (« plus de recettes dans mon répertoire », etc.).
            return GroundedAnswer(
                answer_sentences=[
                    GroundedSentence(
                        text=(
                            "Je n’ai trouvé aucun extrait d’article correspondant "
                            "dans les archives indexées pour cette question. Les "
                            "recherches par ingrédient passent par le texte des "
                            "recettes : reformulez ou précisez un plat, un chef ou "
                            "un mot-clé."
                        ),
                        source_chunk_ids=[],
                    )
                ],
                recipe_card=None,
                recipe_card_secondary=None,
                chef_card=None,
                follow_up=(
                    "Souhaitez-vous essayer un autre ingrédient ou le nom d’un plat "
                    "libanais ?"
                ),
                confidence=0.05,
            )

        context = _format_context(hits)
        n_articles = len({h.hit.article_external_id for h in hits})
        user_parts: list[str] = []
        uq0 = (user_query or "").lower()
        if n_articles > 1 and any(
            w in uq0 for w in ("autre", "encore", "préfér", "prefere", "préfr", "suite")
        ):
            user_parts.append(
                f"Note (session) : le CONTEXTE ci-dessous contient **{n_articles}** "
                f"articles (`article=`) distincts. Pour « une autre recette » sur le "
                f"même fil, tu dois t’appuyer sur un article **différent** de celui "
                f"évoqué en dernier dans l’HISTORIQUE si un tel extrait s’y prête, "
                f"et citer les bons `chunk_id`."
            )
        if conversation_history and conversation_history.strip():
            user_parts.append(
                "HISTORIQUE DU CHAT (tours précédents complets : questions, "
                "tes propositions de plats, et surtout tes relances / follow_up "
                "en fin de message — l’utilisateur peut y répondre dans un tour "
                "suivant). Le message actuel de l’utilisateur est la ligne "
                "QUESTION ci-dessous :\n"
                + conversation_history.strip()
            )
        if session_thread_summary and session_thread_summary.strip():
            user_parts.append(
                "Synthèse du fil (module session, à prendre en compte pour la "
                "cohérence, sans contredire le CONTEXTE) :\n"
                + session_thread_summary.strip()
            )
        user_parts.append(f"QUESTION : {user_query.strip()}")
        user_parts.append(context)
        user_content = "\n\n".join(user_parts)
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
            )
            raw = completion.choices[0].message.content or "{}"
            answer = GroundedAnswer.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("response_generator.openai_failed", error=str(exc))
            return GroundedAnswer(
                answer_sentences=[
                    GroundedSentence(
                        text=(
                            "Je n’ai pas pu générer une réponse structurée pour "
                            "l’instant (service de langage indisponible ou réponse "
                            "inattendue). Réessayez dans quelques instants."
                        ),
                        source_chunk_ids=[],
                    )
                ],
                recipe_card=None,
                recipe_card_secondary=None,
                chef_card=None,
                follow_up="Réessayez dans un petit moment.",
                confidence=0.0,
            )
        return validate_grounding(answer, hits)


def validate_grounding(
    answer: GroundedAnswer, hits: list[RerankedHit]
) -> GroundedAnswer:
    """Filtre les phrases sans citation valide. Recalcule la confidence si besoin."""
    valid_ids = {h.hit.chunk_id for h in hits}
    grounded = []
    for sent in answer.answer_sentences:
        kept_ids = [cid for cid in sent.source_chunk_ids if cid in valid_ids]
        if not kept_ids:
            # Réponses très courtes (accord / refus) : pas de chunk explicite.
            tnorm = sent.text.strip().lower().replace("’", "'")
            if tnorm in ("oui", "non", "ok", "merci", "d'accord"):
                grounded.append(GroundedSentence(text=sent.text, source_chunk_ids=[]))
                continue
            # Phrases « pas de résultat » / honnêteté sur le corpus : garder le texte
            # sans citation (évite de supprimer tout le paragraphe).
            low = sent.text.lower()
            if any(
                token in low
                for token in (
                    "je n'ai pas",
                    "je n'ai rien",
                    "j'ai rien",
                    "rien trouvé",
                    "assez solide",
                    "désolé",
                    "pourriez-vous",
                    "reformul",
                    "préciser",
                    "les archives",
                    "aucun extrait",
                    "ne proposent pas",
                    "pas de recette",
                    "insuffisant",
                    "indexées",
                    "n'y a pas",
                    "pas de ",
                    "dedans",
                    "ne contient pas",
                    "manque",
                )
            ):
                grounded.append(GroundedSentence(text=sent.text, source_chunk_ids=[]))
                continue
            # Phrase perdue = trou fréquent : préférer une phrase explicative courte
            # plutôt que tout supprimer (le HTML tombait sur un message générique).
            if len(sent.text.strip()) >= 24:
                grounded.append(
                    GroundedSentence(
                        text=sent.text.strip(),
                        source_chunk_ids=[],
                    )
                )
            continue
        grounded.append(GroundedSentence(text=sent.text, source_chunk_ids=kept_ids))
    answer.answer_sentences = grounded
    if not grounded:
        answer.confidence = min(answer.confidence, 0.2)

    # Même filtre sur les cartes (sinon liens HTML / sources incohérents).
    # Ne pas exposer ingrédients / étapes dans le widget : trafic vers l'article.
    if answer.recipe_card is not None:
        rc = answer.recipe_card
        rc_ids = [cid for cid in rc.source_chunk_ids if cid in valid_ids]
        if not rc_ids:
            answer.recipe_card = None
        else:
            answer.recipe_card = rc.model_copy(
                update={
                    "source_chunk_ids": rc_ids,
                    "ingredients": [],
                    "steps": [],
                }
            )
    if answer.recipe_card_secondary is not None:
        rc2 = answer.recipe_card_secondary
        rc2_ids = [cid for cid in rc2.source_chunk_ids if cid in valid_ids]
        if not rc2_ids:
            answer.recipe_card_secondary = None
        else:
            answer.recipe_card_secondary = rc2.model_copy(
                update={
                    "source_chunk_ids": rc2_ids,
                    "ingredients": [],
                    "steps": [],
                }
            )
    if answer.chef_card is not None:
        cc = answer.chef_card
        cc_ids = [cid for cid in cc.source_chunk_ids if cid in valid_ids]
        if not cc_ids:
            answer.chef_card = None
        else:
            answer.chef_card = cc.model_copy(update={"source_chunk_ids": cc_ids})

    # Même article ou même titre : une seule carte (évite doublon visuel dans le widget).
    if answer.recipe_card is not None and answer.recipe_card_secondary is not None:

        def _article_id_for(rc: RecipeCard) -> int | None:
            want = {cid for cid in rc.source_chunk_ids if cid in valid_ids}
            for h in hits:
                if h.hit.chunk_id in want:
                    return int(h.hit.article_external_id)
            return None

        def _norm_recipe_title(t: str) -> str:
            return " ".join(t.strip().lower().replace("’", "'").split())

        a1, a2 = _article_id_for(answer.recipe_card), _article_id_for(
            answer.recipe_card_secondary
        )
        if a1 is not None and a2 is not None and a1 == a2:
            answer.recipe_card_secondary = None
        else:
            t1 = _norm_recipe_title(answer.recipe_card.title)
            t2 = _norm_recipe_title(answer.recipe_card_secondary.title)
            if t1 and t2 and t1 == t2:
                answer.recipe_card_secondary = None

    return answer

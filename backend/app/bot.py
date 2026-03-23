"""Orchestration : analyse → retrieval/rerank → génération HTML ; mémoire de session optionnelle."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, List

from unidecode import unidecode

from .core.config import get_settings
from .core.model_selector import get_model_for_request
from .core.intent_router import route_intent_deterministic
from .core.safety import safety_gate_check
from .llm.query_analyzer import QueryAnalyzer
from .llm.response_generator import ResponseGenerator
from .rag.retriever import HybridRetriever
from .rag.session_manager import SessionManager, is_continuation
from .schemas.query_analysis import QueryAnalysis
from .schemas.responses import (
    OLJRecommendation,
    RecipeNarrative,
    RecipeCard,
    SahtenResponse,
    EvidenceBundle,
    ConversationBlock,
)

logger = logging.getLogger(__name__)

# Style « Golden Database » / éditorial OLJ : ouverture systématique (sauf si déjà présente).
_EDITORIAL_SALUTATION = "Bonjour,"


@dataclass
class SahtenConfig:
    enable_safety_check: bool = True
    enable_narrative_generation: bool = True


class SahtenBot:
    """
    Main Sahten chatbot.
    
    Supports dynamic model selection per-request.
    Includes session memory to avoid repeating recipes.
    """
    
    def __init__(self, config: Optional[SahtenConfig] = None):
        self.config = config or SahtenConfig()
        settings = get_settings()
        
        # Default model from settings
        self.default_model = settings.openai_model
        
        # Initialize components with default model
        self.analyzer = QueryAnalyzer(model=self.default_model)
        self.retriever = HybridRetriever()
        self.generator = ResponseGenerator(model=self.default_model)
        
        # Session memory (TTL 30 min, max 2000 sessions)
        self.session_manager = SessionManager(max_sessions=2000, ttl_minutes=30)

        logger.info("SahtenBot initialized (default_model=%s, session_memory=enabled)", self.default_model)

    def _get_analyzer(self, model: str) -> QueryAnalyzer:
        """Get analyzer for specific model (cached or new)."""
        if model == self.default_model:
            return self.analyzer
        return QueryAnalyzer(model=model)
    
    def _get_generator(self, model: str) -> ResponseGenerator:
        """Get generator for specific model (cached or new)."""
        if model == self.default_model:
            return self.generator
        return ResponseGenerator(model=model)

    def _build_conversation_blocks(
        self,
        *,
        narrative: RecipeNarrative,
        evidence: EvidenceBundle,
    ) -> List[ConversationBlock]:
        blocks: List[ConversationBlock] = []
        main_parts = [narrative.hook, narrative.cultural_context]
        if narrative.teaser and evidence.response_type in {"not_found_with_alternative", "recipe_not_found"}:
            main_parts.append(narrative.teaser)
        assistant_text = " ".join(part.strip() for part in main_parts if part).strip()
        # Salutation éditoriale — sauf alternative contractuelle : le brief exige que l'accroche
        # « Je suis désolé… je peux te proposer » soit en tête du message (sans « Bonjour » devant).
        if assistant_text and evidence.response_type != "not_found_with_alternative":
            first_word = assistant_text.split(None, 1)[0].lower().rstrip(",!?.")
            if first_word not in {"bonjour", "marhaba", "salut"}:
                assistant_text = f"{_EDITORIAL_SALUTATION}\n\n{assistant_text}"
        if evidence.response_type == "not_found_with_alternative":
            # Hook contractuel + détail : ne pas tronquer l'accroche obligatoire
            max_chars = 920
        elif evidence.response_type == "menu" or len(evidence.selected_recipe_cards or []) > 1:
            max_chars = 820
        else:
            max_chars = 360
        if len(assistant_text) > max_chars:
            assistant_text = assistant_text[: max_chars - 1].rstrip() + "…"
        if assistant_text:
            blocks.append(ConversationBlock(block_type="assistant_message", text=assistant_text))
        if evidence.grounded_excerpt:
            blocks.append(ConversationBlock(block_type="grounded_snippet", text=evidence.grounded_excerpt))
        if evidence.session_context.get("is_continuation"):
            blocks.append(
                ConversationBlock(
                    block_type="follow_up_question",
                    text="Je peux vous proposer une variante plus ciblée si vous le souhaitez.",
                )
            )
        if narrative.cta:
            blocks.append(ConversationBlock(block_type="cta", text=narrative.cta))
        return blocks

    def _apply_scenario_metadata(
        self,
        *,
        message: str,
        response: SahtenResponse,
        analysis: Optional[QueryAnalysis],
        debug_info: Optional[dict],
        trace_meta: dict,
        retrieval_debug: Optional[dict] = None,
    ) -> None:
        """Enrichit trace_meta + debug_info (scénario démo / link_resolution)."""
        from .core.scenario_metadata import build_api_scenario_metadata, merge_scenario_into_debug

        sm = build_api_scenario_metadata(
            message=message,
            response=response,
            analysis=analysis,
            trace_meta=trace_meta,
            retrieval_debug=retrieval_debug or {},
        )
        trace_meta["api_scenario"] = sm
        if debug_info is not None:
            merge_scenario_into_debug(debug_info, sm)

    async def chat(
        self,
        message: str,
        *,
        debug: bool = False,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[SahtenResponse, Optional[dict], dict]:
        """
        Process a chat message.
        
        Args:
            message: User's message
            debug: Include debug info in response
            model: Override model selection (None = use default/A/B)
            request_id: Request ID for A/B testing
            session_id: Session ID for conversation memory
        
        Returns:
            (SahtenResponse, debug_info, trace_meta)
            trace_meta: always present, for observability (routing_source, safety_blocked, etc.)
        """
        # Determine model to use
        effective_model = get_model_for_request(
            request_id or "default",
            model
        )
        
        # Get components for this model
        analyzer = self._get_analyzer(effective_model)
        generator = self._get_generator(effective_model)
        
        # Get or create session for memory
        session = None
        session_context: dict = {}
        if session_id:
            session = self.session_manager.get_or_create(session_id)
            logger.debug("Session %s: %d turns, %d recipes proposed", 
                        session_id, len(session.conversation_history), len(session.recipes_proposed))
            session_context = session.get_context_for_continuation()
            session_context["is_continuation"] = is_continuation(message)
        
        trace_meta: dict = {}
        step_timings_ms: dict[str, int] = {}
        started_at = time.perf_counter()

        def finalize_trace() -> dict:
            if "total" not in step_timings_ms:
                step_timings_ms["total"] = int((time.perf_counter() - started_at) * 1000)
            if step_timings_ms:
                trace_meta["timings_ms"] = step_timings_ms
            return trace_meta

        # 0) SafetyGate deterministic (before any LLM)
        safety_result = safety_gate_check(message)
        if safety_result.blocked:
            trace_meta["safety_blocked"] = True
            narrative = generator.generate_redirect(safety_result.redirect_suggestion)
            evidence = EvidenceBundle(
                response_type="redirect",
                intent_detected="off_topic",
                user_query=message,
                session_context=session_context,
            )
            resp_s = SahtenResponse(
                response_type="redirect",
                narrative=narrative,
                conversation_blocks=self._build_conversation_blocks(narrative=narrative, evidence=evidence),
                recipes=[],
                recipe_count=0,
                intent_detected="off_topic",
                confidence=1.0,
                model_used=effective_model,
            )
            dbg_s = {"safety_gate": "blocked", "model": effective_model} if debug else None
            meta_s = finalize_trace()
            self._apply_scenario_metadata(
                message=message,
                response=resp_s,
                analysis=None,
                debug_info=dbg_s,
                trace_meta=meta_s,
            )
            return resp_s, dbg_s, meta_s

        settings = get_settings()
        use_cache = (
            settings.enable_response_cache
            and not debug
            and (session is None or not session.has_history())
        )

        def maybe_cache_put(resp: SahtenResponse, dbg: Optional[dict], meta: dict) -> None:
            if not use_cache or resp.response_type == "redirect":
                return
            from .core.response_cache import get_response_cache

            try:
                rcopy = resp.model_copy(deep=True)
            except AttributeError:
                import copy

                rcopy = copy.deepcopy(resp)
            get_response_cache().put(message, effective_model, (rcopy, dbg, dict(meta)))

        if use_cache:
            from .core.response_cache import get_response_cache

            hit = get_response_cache().get(message, effective_model)
            if hit is not None:
                resp_h, dbg_h, meta_h = hit
                total_ms = int((time.perf_counter() - started_at) * 1000)
                out_meta = {**meta_h, "cache_hit": True}
                tm = dict(out_meta.get("timings_ms") or {})
                tm["total"] = total_ms
                out_meta["timings_ms"] = tm
                dbg_out = dbg_h if debug else None
                self._apply_scenario_metadata(
                    message=message,
                    response=resp_h,
                    analysis=None,
                    debug_info=dbg_out,
                    trace_meta=out_meta,
                    retrieval_debug=(dbg_h or {}).get("retrieval") if isinstance(dbg_h, dict) else None,
                )
                return resp_h, dbg_out, out_meta

        # 1) Analyze query: deterministic router first, LLM only for ambiguous
        analysis = route_intent_deterministic(message)
        routing_source = "deterministic" if analysis is not None else "llm"
        trace_meta["routing_source"] = routing_source
        if analysis is None:
            analyze_started = time.perf_counter()
            analysis = await analyzer.analyze(message)
            step_timings_ms["analysis"] = int((time.perf_counter() - analyze_started) * 1000)

        if not analysis.safety.is_safe:
            narrative = generator.generate_redirect(analysis.redirect_suggestion)
            evidence = EvidenceBundle(
                response_type="redirect",
                intent_detected=analysis.intent,
                user_query=message,
                session_context=session_context,
            )
            resp_u = SahtenResponse(
                response_type="redirect",
                narrative=narrative,
                conversation_blocks=self._build_conversation_blocks(narrative=narrative, evidence=evidence),
                recipes=[],
                recipe_count=0,
                intent_detected=analysis.intent,
                confidence=analysis.intent_confidence,
                model_used=effective_model,
            )
            dbg_u = {"analysis": analysis.model_dump(), "model": effective_model} if debug else None
            meta_u = finalize_trace()
            self._apply_scenario_metadata(
                message=message,
                response=resp_u,
                analysis=analysis,
                debug_info=dbg_u,
                trace_meta=meta_u,
            )
            return resp_u, dbg_u, meta_u

        if analysis.intent == "greeting":
            narrative = generator.generate_greeting()
            evidence = EvidenceBundle(
                response_type="greeting",
                intent_detected=analysis.intent,
                user_query=message,
                session_context=session_context,
            )
            resp = SahtenResponse(
                response_type="greeting",
                narrative=narrative,
                conversation_blocks=self._build_conversation_blocks(narrative=narrative, evidence=evidence),
                recipes=[],
                recipe_count=0,
                intent_detected=analysis.intent,
                confidence=analysis.intent_confidence,
                model_used=effective_model,
            )
            dbg = {"analysis": analysis.model_dump(), "model": effective_model} if debug else None
            meta = finalize_trace()
            self._apply_scenario_metadata(
                message=message,
                response=resp,
                analysis=analysis,
                debug_info=dbg,
                trace_meta=meta,
            )
            maybe_cache_put(resp, dbg, meta)
            return resp, dbg, meta

        if analysis.intent == "about_bot":
            narrative = generator.generate_about_bot()
            olj_reco_ab = OLJRecommendation(
                title="Recettes L'Orient-Le Jour",
                url="https://www.lorientlejour.com/cuisine-liban-a-table",
                reason="Toutes les recettes Sahten viennent de la rubrique Cuisine",
            )
            evidence = EvidenceBundle(
                response_type="greeting",
                intent_detected=analysis.intent,
                user_query=message,
                olj_recommendation=olj_reco_ab,
                session_context=session_context,
            )
            resp_ab = SahtenResponse(
                response_type="greeting",
                narrative=narrative,
                conversation_blocks=self._build_conversation_blocks(narrative=narrative, evidence=evidence),
                recipes=[],
                olj_recommendation=olj_reco_ab,
                recipe_count=0,
                intent_detected=analysis.intent,
                confidence=analysis.intent_confidence,
                model_used=effective_model,
            )
            dbg_ab = {"analysis": analysis.model_dump(), "model": effective_model} if debug else None
            meta_ab = finalize_trace()
            self._apply_scenario_metadata(
                message=message,
                response=resp_ab,
                analysis=analysis,
                debug_info=dbg_ab,
                trace_meta=meta_ab,
            )
            maybe_cache_put(resp_ab, dbg_ab, meta_ab)
            return resp_ab, dbg_ab, meta_ab

        if analysis.intent == "off_topic" or not analysis.is_culinary:
            narrative = generator.generate_redirect(analysis.redirect_suggestion)
            evidence = EvidenceBundle(
                response_type="redirect",
                intent_detected=analysis.intent,
                user_query=message,
                session_context=session_context,
            )
            resp_o = SahtenResponse(
                response_type="redirect",
                narrative=narrative,
                conversation_blocks=self._build_conversation_blocks(narrative=narrative, evidence=evidence),
                recipes=[],
                recipe_count=0,
                intent_detected=analysis.intent,
                confidence=analysis.intent_confidence,
                model_used=effective_model,
            )
            dbg_o = {"analysis": analysis.model_dump(), "model": effective_model} if debug else None
            meta_o = finalize_trace()
            self._apply_scenario_metadata(
                message=message,
                response=resp_o,
                analysis=analysis,
                debug_info=dbg_o,
                trace_meta=meta_o,
            )
            return resp_o, dbg_o, meta_o

        if analysis.intent == "clarification":
            term = analysis.dish_name or message
            grounding = self.retriever.get_grounding_for_term(term)
            olj_reco = None
            if grounding:
                _, title, url = grounding
                olj_reco = OLJRecommendation(
                    title=title,
                    url=url,
                    reason="Lis l'article complet sur L'Orient-Le Jour",
                )
            narrative = generator.generate_clarification(term, grounding=grounding)
            evidence = EvidenceBundle(
                response_type="clarification",
                intent_detected=analysis.intent,
                user_query=message,
                grounded_excerpt=grounding[0] if grounding else None,
                grounded_title=grounding[1] if grounding else None,
                grounded_url=grounding[2] if grounding else None,
                olj_recommendation=olj_reco,
                allowed_claims=["Citer des extraits OLJ et répondre brièvement"],
                forbidden_claims=["Inventer un fait non ancré"],
                session_context=session_context,
            )
            resp = SahtenResponse(
                response_type="clarification",
                narrative=narrative,
                conversation_blocks=self._build_conversation_blocks(narrative=narrative, evidence=evidence),
                recipes=[],
                olj_recommendation=olj_reco,
                recipe_count=0,
                intent_detected=analysis.intent,
                confidence=analysis.intent_confidence,
                model_used=effective_model,
            )
            dbg = {"analysis": analysis.model_dump(), "model": effective_model} if debug else None
            meta = finalize_trace()
            self._apply_scenario_metadata(
                message=message,
                response=resp,
                analysis=analysis,
                debug_info=dbg,
                trace_meta=meta,
            )
            maybe_cache_put(resp, dbg, meta)
            return resp, dbg, meta

        # 2) Retrieve + rerank
        # Exclude recipes already proposed in this session
        exclude_urls = session.recipes_proposed if session else []
        exclude_set = set(exclude_urls)

        # ExactDishResolver: deterministic lookup before full retrieval (guarantees OLJ when in base)
        recipes: List[RecipeCard] = []
        is_base2 = False
        retrieval_debug = None
        if analysis.intent == "recipe_specific" and analysis.dish_name:
            exact = self.retriever.resolve_exact_dish(
                dish_name=analysis.dish_name,
                dish_name_variants=analysis.dish_name_variants,
                exclude_urls=exclude_set,
            )
            if exact:
                recipes = [exact]
                retrieval_debug = {"exact_match": True, "source": "ExactDishResolver"}
                trace_meta["exact_match"] = True

        if not recipes:
            retrieval_started = time.perf_counter()
            recipes, is_base2, retrieval_debug = await self.retriever.search_with_rerank(
                analysis, raw_query=message, debug=debug, exclude_urls=exclude_urls
            )
            step_timings_ms["retrieval"] = int((time.perf_counter() - retrieval_started) * 1000)
            if retrieval_debug and retrieval_debug.get("rerank_shortcircuit"):
                trace_meta["rerank_shortcircuit"] = True

        # Semantic guard: for specific dish requests, verify the returned recipe
        # actually matches the requested dish (prevents returning unrelated Lebanese
        # recipes when a non-Lebanese dish is requested, e.g. "lasagne" → kebbé).
        if (
            recipes
            and analysis.intent == "recipe_specific"
            and analysis.dish_name
            and not is_base2
        ):
            dish_variants = {analysis.dish_name.lower()} | {
                v.lower() for v in (analysis.dish_name_variants or [])
            }
            dish_variants_ascii = {unidecode(v) for v in dish_variants}

            def _recipe_matches_dish(recipe_title: str) -> bool:
                title_lower = recipe_title.lower()
                title_ascii = unidecode(title_lower)
                return any(
                    variant in title_lower or variant in title_ascii
                    for variant in dish_variants | dish_variants_ascii
                )

            if not any(_recipe_matches_dish(r.title or "") for r in recipes):
                logger.info(
                    "Semantic guard: dish '%s' not found in recipe titles %s — returning not-found",
                    analysis.dish_name,
                    [r.title for r in recipes],
                )
                recipes = []

        # Sélection OLJ hors-sujet : retenter Base2 uniquement pour plats canoniquement « Base2 only »
        # (fattouche, houmous, etc.) — pas pour des demandes type bœuf bourguignon (alternative prouvée).
        if (
            not recipes
            and analysis.intent == "recipe_specific"
            and analysis.dish_name
        ):
            from .data.dish_normalizer import dish_normalizer

            if dish_normalizer.is_base2_only(analysis.dish_name):
                base2_retry = self.retriever._search_base2(analysis)
                if base2_retry:
                    recipes = base2_retry[: analysis.recipe_count]
                    is_base2 = True
                    trace_meta["base2_after_semantic_guard"] = True

        if not recipes:
            # Try to find a Lebanese alternative sharing at least one main ingredient
            exclude_urls = set(session.recipes_proposed if session else [])
            alt_match = self.retriever.get_alternative_by_shared_ingredient(
                raw_query=message,
                analysis=analysis,
                exclude_urls=exclude_urls,
                inferred_ingredients=analysis.inferred_main_ingredients,
            )
            if alt_match is None and analysis.intent in (
                "recipe_specific",
                "recipe_by_ingredient",
            ):
                alt_match = self.retriever.get_category_fallback_match(
                    analysis, message, exclude_urls
                )
            if alt_match and (analysis.intent == "recipe_specific" or analysis.intent == "recipe_by_ingredient"):
                alternative = alt_match.recipe_card
                generation_started = time.perf_counter()
                narrative = await generator.generate_proven_alternative_narrative(
                    user_query=message,
                    alternative=alternative,
                    proof=alt_match.proof,
                    session_context=session_context,
                    match_reason=alt_match.match_reason,
                )
                step_timings_ms["generation"] = int((time.perf_counter() - generation_started) * 1000)
                trace_meta["shared_ingredient_proof"] = {
                    "query_ingredient": alt_match.proof.query_ingredient,
                    "shared_ingredients": alt_match.proof.shared_ingredients,
                    "proof_score": alt_match.proof.proof_score,
                }
                logger.info(
                    "Proven alternative: query_ingredient=%s shared=%s",
                    alt_match.proof.query_ingredient,
                    alt_match.proof.shared_ingredients,
                )
                debug_info = (
                    {
                        "analysis": analysis.model_dump(),
                        "retrieval": retrieval_debug,
                        "model": effective_model,
                        "shared_ingredient_proof": trace_meta["shared_ingredient_proof"],
                    }
                    if debug
                    else None
                )
                resp = SahtenResponse(
                    response_type="not_found_with_alternative",
                    narrative=narrative,
                    conversation_blocks=self._build_conversation_blocks(
                        narrative=narrative,
                        evidence=EvidenceBundle(
                            response_type="not_found_with_alternative",
                            intent_detected=analysis.intent,
                            user_query=message,
                            selected_recipe_cards=[alternative],
                            shared_ingredient_proof=alt_match.proof,
                            match_reason=alt_match.match_reason,
                            session_context=session_context,
                        ),
                    ),
                    recipes=[alternative],
                    recipe_count=1,
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                    model_used=effective_model,
                )
                meta = finalize_trace()
                self._apply_scenario_metadata(
                    message=message,
                    response=resp,
                    analysis=analysis,
                    debug_info=debug_info,
                    trace_meta=meta,
                    retrieval_debug=retrieval_debug,
                )
                maybe_cache_put(resp, debug_info, meta)
                return resp, debug_info, meta
            # recipe_not_found_without_proven_alternative: pas de recette, pas d'ingrédient commun prouvé
            olj_reco = self.retriever.get_olj_recommendation(analysis)
            narrative = RecipeNarrative(
                hook="Je n'ai pas la fiche exacte pour cela dans le corpus OLJ.",
                cultural_context=(
                    "Reformulez avec un ingrédient ou un plat libanais "
                    "(mezze, plat du jour, dessert du pays) : je pourrai cibler une recette publiée."
                ),
                teaser="Vous pouvez aussi parcourir la rubrique Cuisine sur le site.",
                cta="Parcourez les recettes sur L'Orient-Le Jour",
                closing="Sahten !",
            )
            resp = SahtenResponse(
                response_type="recipe_not_found",
                narrative=narrative,
                conversation_blocks=self._build_conversation_blocks(
                    narrative=narrative,
                    evidence=EvidenceBundle(
                        response_type="recipe_not_found",
                        intent_detected=analysis.intent,
                        user_query=message,
                        olj_recommendation=olj_reco,
                        session_context=session_context,
                    ),
                ),
                recipes=[],
                olj_recommendation=olj_reco,
                recipe_count=0,
                intent_detected=analysis.intent,
                confidence=analysis.intent_confidence,
                model_used=effective_model,
            )
            dbg_nf = (
                {"analysis": analysis.model_dump(), "retrieval": retrieval_debug, "model": effective_model}
                if debug
                else None
            )
            meta = finalize_trace()
            self._apply_scenario_metadata(
                message=message,
                response=resp,
                analysis=analysis,
                debug_info=dbg_nf,
                trace_meta=meta,
                retrieval_debug=retrieval_debug,
            )
            maybe_cache_put(resp, dbg_nf, meta)
            return resp, dbg_nf, meta

        # 3) Generate narrative
        if self.config.enable_narrative_generation:
            generation_started = time.perf_counter()
            narrative = await generator.generate_narrative(
                user_query=message,
                analysis=analysis,
                recipes=recipes,
                is_base2_fallback=is_base2,
                session_context=session_context,
            )
            step_timings_ms["generation"] = int((time.perf_counter() - generation_started) * 1000)
        else:
            narrative = generator.generate_redirect(
                "Mode sans génération narrative activé."
            )

        olj_reco = None
        if is_base2:
            olj_reco = self.retriever.get_olj_recommendation(
                analysis, exclude_titles=[r.title for r in recipes]
            )

        if analysis.intent == "menu_composition":
            response_type = "menu"
        elif is_base2:
            response_type = "recipe_base2"
        else:
            response_type = "recipe_olj"

        resp = SahtenResponse(
            response_type=response_type,
            narrative=narrative,
            conversation_blocks=self._build_conversation_blocks(
                narrative=narrative,
                evidence=EvidenceBundle(
                    response_type=response_type,
                    intent_detected=analysis.intent,
                    user_query=message,
                    selected_recipe_cards=recipes,
                    exact_match=bool(trace_meta.get("exact_match")),
                    olj_recommendation=olj_reco,
                    session_context=session_context,
                ),
            ),
            recipes=recipes,
            olj_recommendation=olj_reco,
            recipe_count=len(recipes),
            intent_detected=analysis.intent,
            confidence=analysis.intent_confidence,
            model_used=effective_model,
        )
        
        # Save session with new recipes
        if session and recipes:
            primary_url = recipes[0].url if recipes else None
            session.add_turn(
                user_message=message,
                intent=analysis.intent,
                primary_dish=analysis.dish_name,
                ingredients=analysis.ingredients,
                recipe_url=primary_url,
                response_summary=f"{len(recipes)} recettes proposées",
            )
            self.session_manager.save(session)
        
        dbg = (
            {"analysis": analysis.model_dump(), "retrieval": retrieval_debug, "model": effective_model}
            if debug
            else None
        )
        meta = finalize_trace()
        self._apply_scenario_metadata(
            message=message,
            response=resp,
            analysis=analysis,
            debug_info=dbg,
            trace_meta=meta,
            retrieval_debug=retrieval_debug,
        )
        maybe_cache_put(resp, dbg, meta)
        return resp, dbg, meta


# Singleton instance
_bot_instance: Optional[SahtenBot] = None


def get_bot() -> SahtenBot:
    """Get singleton bot instance."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = SahtenBot()
    return _bot_instance


def reload_bot() -> SahtenBot:
    """Force reload bot instance (after config change)."""
    global _bot_instance
    _bot_instance = SahtenBot()
    return _bot_instance

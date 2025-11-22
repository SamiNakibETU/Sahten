"""
Main RAG Pipeline
Orchestrates all components to generate editorial-compliant responses
"""

import logging
from typing import Optional

from app.models.schemas import (
    ChatResponse,
    QueryPlan,
    ClassificationResult,
)
from app.rag.classifier_agent import ClassifierAgent
from app.rag.query_planner_agent import QueryPlannerAgent
from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.rag.link_resolver import LinkResolver
from app.rag.scenario_alignment import ScenarioAligner
from app.rag.response_composer import ResponseComposer
from app.rag.content_guard import ContentGuard
from app.data.content_index import ContentIndex
from app.data.link_index import LinkIndex
from app.models.llm_client import LLMClient, get_llm_client
from app.models.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates all components

    Flow:
    1. Classify query → ClassificationResult
    2. Plan query → QueryPlan
    3. Retrieve content → RetrievalCandidates
    4. Rerank results → Top candidates
    5. Resolve article link → LinkResolutionResult
    6. Align scenario → ScenarioContext
    7. Compose response → HTML
    8. Validate & sanitize → Final HTML
    """

    def __init__(
        self,
        content_index: ContentIndex,
        link_index: LinkIndex,
        llm_client: Optional[LLMClient] = None,
    ):
        # Indexes
        self.content_index = content_index
        self.link_index = link_index

        # LLM
        self.llm = llm_client or get_llm_client()

        # Agents
        self.classifier = ClassifierAgent(llm_client=self.llm)
        self.planner = QueryPlannerAgent()

        # RAG components
        self.retriever = Retriever(content_index=content_index)
        self.reranker = Reranker()
        self.link_resolver = LinkResolver(link_index=link_index)

        # Response generation
        self.scenario_aligner = ScenarioAligner()
        self.response_composer = ResponseComposer()
        self.content_guard = ContentGuard()

        logger.info("RAG Pipeline initialized")

    def process(self, user_message: str, debug: bool = False) -> ChatResponse:
        """
        Process a user message and return a complete response

        Args:
            user_message: The user's query
            debug: If True, include debug info in response

        Returns:
            ChatResponse with HTML, scenario, and debug info
        """
        try:
            debug_info = {} if debug else None

            # Step 1: Classify query
            logger.debug(f"Step 1: Classifying query: {user_message[:50]}...")
            classification = self.classifier.classify(user_message)

            if debug:
                debug_info["classification"] = {
                    "intent": classification.intent,
                    "language": classification.language,
                    "confidence": classification.confidence,
                    "slots": classification.slots,
                }

            # Step 2: Plan query
            logger.debug("Step 2: Planning query...")
            query_plan = self.planner.plan(classification, user_message)

            if debug:
                debug_info["query_plan"] = {
                    "need_type": query_plan.need_type,
                    "primary_dish": query_plan.primary_dish,
                    "ingredients": query_plan.ingredients,
                    "retrieval_query": query_plan.retrieval_query,
                    "link_query": query_plan.link_query,
                }

            # Step 3: Retrieve content (if needed)
            retrieval_candidates = None
            if query_plan.need_type not in ["greeting", "about_bot", "off_topic"]:
                logger.debug("Step 3: Retrieving content...")
                retrieval_candidates = self.retriever.retrieve(
                    query_plan, top_k=settings.retrieval_top_k
                )

                if debug:
                    debug_info["retrieval"] = {
                        "num_candidates": len(retrieval_candidates) if retrieval_candidates else 0,
                        "top_sources": [
                            c.source for c in (retrieval_candidates or [])[:3]
                        ],
                    }

                # Step 4: Rerank (if we have candidates)
                if retrieval_candidates:
                    logger.debug("Step 4: Reranking...")
                    retrieval_candidates = self.reranker.rerank(
                        retrieval_candidates,
                        query_plan,
                        top_k=settings.rerank_top_k,
                    )

                    if debug:
                        debug_info["rerank"] = {
                            "num_after_rerank": len(retrieval_candidates),
                            "top_scores": [
                                round(c.score, 3) for c in retrieval_candidates[:3]
                            ],
                        }

            # Step 5: Resolve article link
            logger.debug("Step 5: Resolving article link...")
            link_result = self.link_resolver.resolve(
                query_plan, retrieval_candidates=retrieval_candidates
            )

            if debug:
                debug_info["link_resolution"] = {
                    "strategy": link_result.strategy,
                    "confidence": round(link_result.confidence, 3),
                    "has_primary": link_result.primary_article is not None,
                    "num_suggested": len(link_result.suggested_articles),
                }

            # Step 6: Align scenario
            logger.debug("Step 6: Aligning scenario...")
            scenario = self.scenario_aligner.align(
                classification, query_plan, link_result, retrieval_candidates
            )

            if debug:
                debug_info["scenario"] = {
                    "scenario_id": scenario.scenario_id,
                    "scenario_name": scenario.scenario_name,
                    "use_base": scenario.use_base,
                    "show_full_recipe": scenario.show_full_recipe,
                }

            # Step 7: Compose response
            logger.debug("Step 7: Composing response...")
            html_response = self.response_composer.compose(
                scenario, query_plan, classification, link_result, retrieval_candidates
            )

            if debug:
                debug_info["composition"] = {
                    "html_length": len(html_response),
                    "word_count": self.content_guard._count_words(html_response),
                }

            # Step 8: Validate & sanitize
            logger.debug("Step 8: Validating and sanitizing...")
            validation_result = self.content_guard.validate(html_response, scenario)

            if not validation_result.is_valid:
                logger.warning(f"Validation errors: {validation_result.errors}")
                # Try to sanitize
                html_response = self.content_guard.sanitize(html_response, scenario)

                # Re-validate
                validation_result = self.content_guard.validate(html_response, scenario)

            if debug:
                debug_info["validation"] = {
                    "is_valid": validation_result.is_valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                }

            # Build final response
            primary_url = (
                link_result.primary_article.url
                if link_result.primary_article
                else None
            )

            return ChatResponse(
                html=html_response,
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.scenario_name,
                used_base=scenario.use_base,
                primary_url=primary_url,
                debug_info=debug_info if debug else {},
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)

            # Return error response
            return ChatResponse(
                html="<p>😊 Désolé, une erreur est survenue. Veuillez réessayer.</p>",
                scenario_id=0,
                scenario_name="error",
                used_base="none",
                primary_url=None,
                debug_info={"error": str(e)} if debug else {},
            )


# Global pipeline instance (will be initialized on startup)
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Get global pipeline instance"""
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")
    return _pipeline


def initialize_pipeline(
    content_index: ContentIndex,
    link_index: LinkIndex,
    llm_client: Optional[LLMClient] = None,
) -> RAGPipeline:
    """Initialize global pipeline instance"""
    global _pipeline
    _pipeline = RAGPipeline(
        content_index=content_index,
        link_index=link_index,
        llm_client=llm_client,
    )
    logger.info("Global pipeline initialized")
    return _pipeline

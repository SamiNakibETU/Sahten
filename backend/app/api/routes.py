"""
API Routes for Sahtein 3.0
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.models.schemas import ChatRequest, ChatResponseAPI

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponseAPI)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for Sahtein chatbot

    Takes a user message and returns an HTML-formatted response
    with links to OLJ articles when relevant.
    """
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")

        # Get RAG pipeline
        from app.rag.pipeline import get_pipeline

        pipeline = get_pipeline()

        # Process message through pipeline
        response = pipeline.process(request.message, debug=request.debug)

        # Convert to API response
        return ChatResponseAPI(
            html=response.html,
            scenario_id=response.scenario_id,
            primary_url=response.primary_url,
            debug_info=response.debug_info if request.debug else None,
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur est survenue. Veuillez réessayer.",
        )


@router.get("/status")
async def status():
    """API status endpoint"""
    try:
        from app.rag.pipeline import get_pipeline
        from app.models.config import settings
        pipeline = get_pipeline()

        return {
            "status": "operational",
            "version": settings.app_version,
            "components": {
                "data_loaders": "ready",
                "content_index": "ready" if pipeline.content_index.is_built else "not_built",
                "link_index": "ready" if pipeline.link_index.is_built else "not_built",
                "rag_pipeline": "ready",
            },
            "stats": {
                "content_docs": len(pipeline.content_index),
                "link_articles": len(pipeline.link_index),
            }
        }
    except Exception as e:
        return {
            "status": "initializing",
            "message": str(e),
        }

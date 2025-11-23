"""
Sahtein 3.1 - Main FastAPI Application
Lebanese culinary chatbot for L'Orient-Le Jour
Production-ready with all P0 fixes implemented
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes import router as api_router
from app.models.config import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Debug mode: {settings.debug}")

    # Initialize data loaders and indexes
    logger.info("Loading data and building indexes...")
    from app.data.loaders import data_cache
    from app.data.content_index import ContentIndex
    from app.data.link_index import LinkIndex
    from app.rag.pipeline import initialize_pipeline

    # Load data
    articles = data_cache.get_olj_articles()
    recipes = data_cache.get_structured_recipes()
    logger.info(f"Loaded {len(articles)} OLJ articles and {len(recipes)} recipes")

    # Build content index
    content_index = ContentIndex()
    content_index.add_olj_articles(articles)
    content_index.add_structured_recipes(recipes)
    content_index.build()
    logger.info(f"Content index built with {len(content_index)} documents")

    # Build link index
    link_index = LinkIndex()
    link_index.add_articles(articles)
    link_index.build()
    logger.info(f"Link index built with {len(link_index)} articles")

    # Initialize RAG pipeline
    initialize_pipeline(content_index, link_index)
    logger.info("RAG pipeline initialized and ready")

    yield

    logger.info(f"Shutting down {settings.app_name}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Lebanese culinary chatbot powered by RAG",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include API routes
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Serve the frontend UI"""
    # Path to frontend HTML file
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"

    # If frontend exists, serve it
    if frontend_path.exists():
        return FileResponse(frontend_path)

    # Otherwise, return JSON status
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "message": "Frontend not found. Please ensure frontend/index.html exists.",
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )

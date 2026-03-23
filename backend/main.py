"""Point d'entrée FastAPI : API recettes, fichiers statiques frontend, webhook CMS."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

# Import components
from app.api.routes import router as api_router
from app.api.webhook import router as webhook_router
from app.core.config import get_settings

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Default Model: {settings.openai_model}")
    logger.info(f"A/B Testing: {'ON' if settings.enable_ab_testing else 'OFF'}")
    logger.info(f"Embeddings: {'ON' if settings.enable_embeddings else 'OFF'}")
    
    # Pre-warm the bot to avoid cold start latency
    try:
        from app.bot import get_bot
        bot = get_bot()
        logger.info(f"Bot ready. Loaded {len(bot.retriever.olj_docs)} OLJ docs.")
    except Exception as e:
        logger.warning(f"Bot pre-warming failed (will retry on request): {e}")
    
    yield
    
    logger.info("Shutting down Sahten.")


# Create FastAPI application
# Disable Swagger docs in production for security
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API assistant recettes L'Orient-Le Jour",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
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
app.include_router(webhook_router, prefix=settings.api_prefix)


# --- Static File Serving ---
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"


@app.get("/css/{filename}")
async def get_css(filename: str):
    file_path = FRONTEND_PATH / "css" / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path, media_type="text/css")
    return Response(status_code=404)


@app.get("/js/{filename}")
async def get_js(filename: str):
    file_path = FRONTEND_PATH / "js" / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path, media_type="application/javascript")
    return Response(status_code=404)


@app.get("/img/{filename}")
async def get_img(filename: str):
    file_path = FRONTEND_PATH / "img" / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return Response(status_code=404)


@app.get("/assets/{filename}")
async def get_assets(filename: str):
    file_path = FRONTEND_PATH / "assets" / filename
    if file_path.exists() and file_path.is_file():
        suffix = file_path.suffix.lower()
        media_types = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "application/octet-stream")
        return FileResponse(file_path, media_type=media_type)
    return Response(status_code=404)


@app.get("/")
async def root():
    """Démo complète (même UI que le dev local sur index.html) : accueil + exemples + sélecteur de modèle."""
    frontend_index = FRONTEND_PATH / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)

    widget_index = FRONTEND_PATH / "widget.html"
    if widget_index.exists():
        return FileResponse(widget_index)

    return {
        "name": settings.app_name,
        "status": "running",
        "message": "Frontend index.html not found.",
    }


@app.get("/embed")
async def embed_widget():
    """Référence widget allégée (intégration tiers) — préférer la copie du markup depuis frontend/widget.html."""
    widget_index = FRONTEND_PATH / "widget.html"
    if widget_index.exists():
        return FileResponse(widget_index)
    return {"error": "widget.html not found"}


@app.get("/test")
async def test_ui():
    """Alias historique : même page que / (index.html)."""
    frontend_index = FRONTEND_PATH / "index.html"

    if frontend_index.exists():
        return FileResponse(frontend_index)

    return {
        "error": "Test UI not found",
        "message": "Please create frontend/index.html",
    }


@app.get("/dashboard")
async def dashboard():
    """Serve the analytics dashboard."""
    dashboard_path = FRONTEND_PATH / "dashboard.html"
    
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    
    return {
        "error": "Dashboard not found",
        "message": "Please create frontend/dashboard.html",
    }


@app.get("/admin")
async def admin_panel():
    """Panneau admin recettes (liste canonique)."""
    admin_path = FRONTEND_PATH / "admin.html"
    if admin_path.exists():
        return FileResponse(admin_path)
    return {"error": "admin.html not found"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": settings.app_version,
        "model": settings.openai_model,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )

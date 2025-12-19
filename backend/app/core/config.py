"""
Sahten Configuration
====================

Centralized application settings.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # App info
    app_name: str = "Sahten"
    app_version: str = "1.0.0"
    
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    
    # Data paths (relative to v2/ directory)
    olj_data_path: str = "data_base_OLJ_enriched.json"
    base2_data_path: str = "Data_base_2.json"
    # Canonical dataset (generated offline)
    olj_canonical_path: str = "data/olj_canonical.json"
    
    # Feature flags
    enable_safety_check: bool = True
    enable_narrative_generation: bool = True
    
    # Retrieval settings
    max_results: int = 5
    olj_score_threshold: float = 0.5

    # RAG settings
    retrieve_top_k: int = 20
    rerank_top_k: int = 10
    rerank_model: str = "gpt-4o-mini"
    enable_embeddings: bool = True
    embedding_provider: str = "mock"  # "openai" or "mock"
    embedding_model: str = "text-embedding-3-small"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api"
    debug: bool = False
    
    # CORS - Allow all origins for Vercel deployment
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = False  # Must be False with wildcard origins
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

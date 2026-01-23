"""
Sahten MVP Configuration
========================

Centralized application settings with flexible model selection,
A/B testing support, and optional embeddings.

This is the SINGLE source of truth for all configuration.
Do NOT create additional config files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # ============================================================================
    # APP INFO
    # ============================================================================
    app_name: str = "Sahten"
    app_version: str = "2.1.0-dev"
    
    # ============================================================================
    # OPENAI CONFIGURATION
    # ============================================================================
    openai_api_key: str = ""
    
    # Default model - can be overridden via API request or A/B testing
    # Options: "gpt-4.1-nano" (economique), "gpt-4o-mini" (qualite)
    openai_model: str = "gpt-4.1-nano"
    
    # LLM settings (for legacy compatibility with loaders/llm_client)
    llm_provider: Literal["openai", "anthropic", "mock"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 500
    anthropic_api_key: str = ""
    
    # ============================================================================
    # MODEL SELECTION & A/B TESTING
    # ============================================================================
    # Enable A/B testing between models
    enable_ab_testing: bool = False
    
    # Models to test (when A/B testing is enabled)
    ab_test_model_a: str = "gpt-4.1-nano"
    ab_test_model_b: str = "gpt-4o-mini"
    
    # Ratio for A/B split (0.5 = 50% each)
    ab_test_ratio: float = 0.5
    
    # ============================================================================
    # DATA PATHS (relative to Sahten_MVP/ directory)
    # ============================================================================
    olj_data_path: str = "data_base_OLJ_enriched.json"
    base2_data_path: str = "Data_base_2.json"
    olj_canonical_path: str = "data/olj_canonical.json"
    
    # Legacy compatibility paths (computed properties would be cleaner but Pydantic...)
    @property
    def data_dir(self) -> Path:
        """Root directory for data files."""
        return Path(__file__).parent.parent.parent.parent
    
    @property
    def olj_recipes_path(self) -> Path:
        """Path to OLJ recipes JSON."""
        return self.data_dir / self.olj_data_path
    
    @property
    def base2_recipes_path(self) -> Path:
        """Path to Base2 recipes JSON."""
        return self.data_dir / self.base2_data_path
    
    @property
    def golden_examples_path(self) -> Path:
        """Path to golden examples (for evaluation)."""
        return self.data_dir / "golden_data_base.json"
    
    # Feature flags for legacy loaders
    use_enriched_data: bool = True
    
    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================
    enable_safety_check: bool = True
    enable_narrative_generation: bool = True
    
    # ============================================================================
    # RETRIEVAL SETTINGS
    # ============================================================================
    max_results: int = 5
    olj_score_threshold: float = 0.5
    retrieve_top_k: int = 20
    rerank_top_k: int = 10
    rerank_model: str = "gpt-4o-mini"
    
    # ============================================================================
    # EMBEDDINGS CONFIGURATION
    # ============================================================================
    # OFF by default - TF-IDF + LLM reranker is sufficient for 145 recipes
    # Enable when: 1000+ recipes OR frequent semantic queries ("quelque chose de frais")
    enable_embeddings: bool = False
    
    # Provider: "openai" (real embeddings) or "mock" (deterministic, for testing)
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # ============================================================================
    # CMS WEBHOOK CONFIGURATION
    # ============================================================================
    # Secret for webhook authentication (set in production)
    webhook_secret: str = ""
    
    # Auto-enrich new recipes via LLM when received from CMS
    auto_enrich_on_webhook: bool = True
    
    # ============================================================================
    # OLJ CMS API CONFIGURATION
    # ============================================================================
    # Base URL for OLJ CMS API
    olj_api_base: str = "https://api.lorientlejour.com/cms"
    
    # API key for authentication (set via OLJ_API_KEY env var)
    olj_api_key: str = ""
    
    # ============================================================================
    # API SETTINGS
    # ============================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api"
    debug: bool = False
    
    # ============================================================================
    # CORS - Allow all origins for Vercel/Railway deployment
    # ============================================================================
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = False
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]
    
    # ============================================================================
    # UPSTASH REDIS (for persistent logging)
    # ============================================================================
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_available_models() -> List[str]:
    """Return list of available models for UI dropdown."""
    return ["gpt-4.1-nano", "gpt-4o-mini"]

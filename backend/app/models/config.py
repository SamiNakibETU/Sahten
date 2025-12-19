"""
Configuration management for Sahtein 3.1
Handles environment variables and application settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application
    app_name: str = "Sahtein 3.1"
    app_version: str = "3.1.0"
    debug: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api"

    # CORS
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # Data paths (relative to project root)
    data_dir: Path = Path(__file__).parent.parent.parent.parent
    olj_recipes_path: Path = (data_dir.parent / "data_base_OLJ_final.json")
    base2_recipes_path: Path = data_dir / "Data_base_2.json"
    golden_examples_path: Path = data_dir / "golden_data_base.json"

    # Trace logging
    trace_log_dir: Path = Path(__file__).parent.parent.parent / "logs"
    trace_log_path: Path = trace_log_dir / "chat_traces.jsonl"

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "mock"] = "openai"
    llm_model: str = "gpt-4o-mini"  # For OpenAI
    llm_temperature: float = 0.1
    llm_max_tokens: int = 500
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Retrieval settings
    retrieval_top_k: int = 10
    rerank_top_k: int = 3
    min_similarity_threshold: float = 0.35
    
    # --- New V6 Scoring & Retrieval Boosts ---
    exact_match_boost: float = 2.0
    olj_boost: float = 1.5
    known_chef_boost: float = 1.2
    
    # --- New V6 Business Rules ---
    max_results: int = 5
    display_mode: Literal["single", "carousel", "list"] = "carousel"
    always_recommend_olj: bool = True
    filter_non_recipes: bool = True
    prefer_lebanese: bool = True
    use_enriched_data: bool = True
    
    # Known Chefs (for boosting)
    known_chefs: list[str] = [
        "Kamal Mouzawak",
        "Alan Geaam",
        "Tara Khattar",
        "Aline Kamakian",
        "Hussein Hadid",
        "Joe Barza",
        "Bethany Kehdy",
        "Greg Malouf",
    ]

    # Content guard settings
    max_response_words: int = 150  # ~100 words target, allow buffer
    max_response_words_recipe: int = 500  # For full Base 2 recipes
    max_emojis: int = 3
    allowed_emoji_categories: list[str] = ["food", "emotion", "celebration"]

    # Recipe Output Mode (Legacy but kept for compat)
    recipe_output_mode: Literal["strict_single", "balanced", "exploration"] = "exploration"

    # Editorial constraints
    default_language: str = "fr"
    allowed_url_domain: str = "https://www.lorientlejour.com"
    cuisine_focus: list[str] = ["Lebanese", "Mediterranean", "Middle Eastern"]

    # Embeddings
    enable_embeddings: bool = True
    embedding_provider: Literal["openai", "mock"] = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_cache_dir: Path = data_dir / "embedding_cache"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
config = settings # Alias for V6 code compatibility

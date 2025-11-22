"""
Configuration management for Sahtein 3.1
Handles environment variables and application settings
"""

from pydantic_settings import BaseSettings
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
    olj_recipes_path: Path = data_dir / "olj_recette_liban_a_table.json"
    base2_recipes_path: Path = data_dir / "Data_base_2.json"
    golden_examples_path: Path = data_dir / "golden_data_base.json"

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "mock"] = "mock"
    llm_model: str = "gpt-4o-mini"  # For OpenAI
    llm_temperature: float = 0.1
    llm_max_tokens: int = 500
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Retrieval settings
    retrieval_top_k: int = 10
    rerank_top_k: int = 3
    min_similarity_threshold: float = 0.3

    # Content guard settings
    max_response_words: int = 150  # ~100 words target, allow buffer
    max_response_words_recipe: int = 500  # For full Base 2 recipes
    max_emojis: int = 3
    allowed_emoji_categories: list[str] = ["food", "emotion", "celebration"]

    # Editorial constraints
    default_language: str = "fr"
    allowed_url_domain: str = "https://www.lorientlejour.com"
    cuisine_focus: list[str] = ["Lebanese", "Mediterranean", "Middle Eastern"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

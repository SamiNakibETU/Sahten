"""Configuration centralisée (pydantic-settings).

Toutes les variables sont lues depuis l'environnement (et `.env` si présent).
Aucune valeur par défaut secrète : tout secret manquant en production lève
une erreur explicite à l'instanciation de `Settings`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Réglages applicatifs typés."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"

    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://sahten:sahten@localhost:5432/sahten",  # type: ignore[arg-type]
        description="DSN SQLAlchemy async (asyncpg).",
    )
    redis_url: RedisDsn = Field(
        default="redis://localhost:6379/0",  # type: ignore[arg-type]
        description="Redis pour cache + queue arq.",
    )
    arq_queue: str = "sahten_ingest"

    openai_api_key: str = ""
    llm_model: str = "gpt-4.1"
    llm_temperature: float = 0.2

    # text-embedding-3-small (1536 dim) : compatible HNSW pgvector
    # (limite 2000 dim) sans recourir à `halfvec`. Largement suffisant
    # pour le corpus OLJ/WhiteBeard. Pour passer à `large` (3072 dim),
    # il faudra basculer la colonne vector → halfvec dans la migration
    # et adapter le retriever (cast `::halfvec(3072)`).
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    enable_local_rerank_fallback: bool = True
    local_rerank_model: str = "BAAI/bge-reranker-v2-m3"

    olj_api_base: str = "https://api.lorientlejour.com/cms"
    olj_api_key: str = ""
    webhook_secret: str = ""

    rag_hybrid_top_k_vector: int = 50
    rag_hybrid_top_k_lexical: int = 50
    rag_rerank_top_k: int = 8
    rag_min_rerank_score: float = 0.20
    rag_rrf_k: int = 60
    # Retrieval : si peu de candidats avec filtres SQL (tags), fusion avec une passe « texte seul »
    rag_retrieval_widen_enabled: bool = True
    rag_retrieval_widen_min_hits: int = 8
    # +candidats quand on exclut des articles session (diversité)
    rag_retrieval_extra_limit_per_excluded: int = 12
    # Avant rerank : entrelace les chunks par article (évite qu’un seul article sature le haut)
    rag_prererank_interleave: bool = True

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"log_level invalide: {v}")
        return v

    def require_production_secrets(self) -> None:
        """Lève si on tourne en prod sans les secrets critiques."""
        if self.app_env != "production":
            return
        missing: list[str] = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.olj_api_key:
            missing.append("OLJ_API_KEY")
        if not self.webhook_secret:
            missing.append("WEBHOOK_SECRET")
        if missing:
            raise RuntimeError(f"Production: secrets manquants: {', '.join(missing)}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Ne pas appeler require_production_secrets() ici : le chargement de l’app
    # (dont /healthz) doit pouvoir démarrer sans bloquer sur les secrets.
    return Settings()

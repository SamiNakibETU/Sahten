"""
Sahten RAG Module
=================

Importer les sous-modules explicitement (ex. `from app.rag.retriever import HybridRetriever`)
pour éviter de charger TF-IDF / index au simple `import app.rag`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["HybridRetriever"]

if TYPE_CHECKING:
    from .retriever import HybridRetriever as HybridRetriever


def __getattr__(name: str) -> Any:
    if name == "HybridRetriever":
        from .retriever import HybridRetriever

        return HybridRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

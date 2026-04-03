"""
Cache disque pour la matrice d’embeddings du corpus OLJ (évite un recompute à chaque cold start).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _fingerprint_docs(urls: list[str], model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    for u in urls:
        h.update(u.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()[:32]


def cache_paths(cache_dir: Path, fingerprint: str) -> tuple[Path, Path]:
    base = cache_dir / f"olj_emb_{fingerprint}"
    return base.with_suffix(".npz"), base.with_suffix(".meta.json")


def try_load_embedding_matrix(
    *,
    cache_dir: Path,
    urls: list[str],
    embedding_model: str,
    expected_shape_n: int,
) -> Optional[np.ndarray]:
    fp = _fingerprint_docs(urls, embedding_model)
    npz_path, meta_path = cache_paths(cache_dir, fp)
    if not npz_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fp:
            return None
        if meta.get("doc_count") != len(urls) or meta.get("doc_count") != expected_shape_n:
            return None
        data = np.load(npz_path)
        arr = data["embeddings"]
        if arr.shape[0] != len(urls):
            return None
        logger.info("Loaded embedding cache: %s (%s docs)", npz_path.name, arr.shape[0])
        return arr
    except Exception as e:
        logger.warning("Embedding cache load failed: %s", e)
        return None


def save_embedding_matrix(
    *,
    cache_dir: Path,
    urls: list[str],
    embedding_model: str,
    matrix: np.ndarray,
) -> None:
    fp = _fingerprint_docs(urls, embedding_model)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = cache_paths(cache_dir, fp)
    try:
        np.savez_compressed(npz_path, embeddings=matrix.astype(np.float32))
        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": fp,
                    "doc_count": len(urls),
                    "embedding_model": embedding_model,
                    "shape": list(matrix.shape),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Saved embedding cache: %s", npz_path)
    except Exception as e:
        logger.warning("Embedding cache save failed: %s", e)

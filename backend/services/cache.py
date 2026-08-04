"""Filesystem-backed embedding cache.

Embeddings are expensive to (re)compute, so we cache them on disk keyed
by the SHA256 hash of the chunk text. If a document is re-uploaded
unchanged, or two documents share identical chunk text, the cached
vector is reused instead of calling the embedding model again.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from utils.hashing import sha256_text
from utils.logger import get_logger, log_event

logger = get_logger("guidely.cache")

CACHE_DIR = Path(__file__).resolve().parent.parent / "storage" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_INDEX_FILE = CACHE_DIR / "cache_index.json"


class EmbeddingCache:
    """Maps text-hash -> embedding vector, persisted as .npy files."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _vector_path(self, text_hash: str) -> Path:
        return self.cache_dir / f"{text_hash}.npy"

    def get(self, text: str) -> Optional[np.ndarray]:
        text_hash = sha256_text(text)
        path = self._vector_path(text_hash)
        if path.exists():
            self.hits += 1
            log_event(logger, "cache_hit", text_hash=text_hash)
            return np.load(path)
        self.misses += 1
        log_event(logger, "cache_miss", text_hash=text_hash)
        return None

    def put(self, text: str, vector: np.ndarray) -> str:
        text_hash = sha256_text(text)
        np.save(self._vector_path(text_hash), vector)
        return text_hash

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
        }


_cache_singleton: Optional[EmbeddingCache] = None


def get_cache() -> EmbeddingCache:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = EmbeddingCache()
    return _cache_singleton

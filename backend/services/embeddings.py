"""Embedding generation service.

The sentence-transformers model is expensive to load (it pulls weights
from disk/network), so it is loaded exactly once per process and reused
for every request via a module-level singleton. Embeddings for
individual chunk texts are cached on disk (see cache.py) keyed by
content hash so unchanged documents are never re-embedded.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from services.cache import get_cache
from services.metrics import get_metrics_tracker
from utils.logger import get_logger, log_event

logger = get_logger("guidely.embeddings")

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class EmbeddingModelError(Exception):
    """Raised when the embedding model fails to load or run."""


class EmbeddingService:
    """Wraps a sentence-transformers model with lazy, single-load semantics."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = None  # loaded lazily on first use
        self.dimension: Optional[int] = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers` or configure an "
                "alternate EMBEDDING_PROVIDER."
            ) from exc

        try:
            log_event(logger, "loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
            log_event(
                logger,
                "embedding_model_loaded",
                model=self.model_name,
                dimension=self.dimension,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingModelError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc
        return self._model

    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Embed a single string, using the on-disk cache when possible."""
        cache = get_cache()
        metrics = get_metrics_tracker()

        if use_cache:
            cached = cache.get(text)
            if cached is not None:
                metrics.record_cache_hit()
                return cached

        model = self._load_model()
        try:
            vector = model.encode(text, normalize_embeddings=True)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingModelError(f"Embedding generation failed: {exc}") from exc

        vector = np.asarray(vector, dtype="float32")
        if use_cache:
            cache.put(text, vector)
        metrics.record_cache_miss()
        metrics.record_embedding_generated()
        return vector

    def embed_batch(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Embed multiple texts, reusing the cache per-text where possible."""
        if not texts:
            return np.zeros((0, self.dimension or 384), dtype="float32")

        cache = get_cache()
        metrics = get_metrics_tracker()

        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        to_embed_idx: List[int] = []
        to_embed_text: List[str] = []

        if use_cache:
            for i, t in enumerate(texts):
                cached = cache.get(t)
                if cached is not None:
                    vectors[i] = cached
                    metrics.record_cache_hit()
                else:
                    to_embed_idx.append(i)
                    to_embed_text.append(t)
        else:
            to_embed_idx = list(range(len(texts)))
            to_embed_text = list(texts)

        if to_embed_text:
            model = self._load_model()
            try:
                new_vectors = model.encode(
                    to_embed_text, normalize_embeddings=True, batch_size=32
                )
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingModelError(
                    f"Batch embedding generation failed: {exc}"
                ) from exc
            new_vectors = np.asarray(new_vectors, dtype="float32")
            for idx, vec, t in zip(to_embed_idx, new_vectors, to_embed_text):
                vectors[idx] = vec
                if use_cache:
                    cache.put(t, vec)
                metrics.record_cache_miss()
                metrics.record_embedding_generated()

        return np.vstack(vectors)  # type: ignore[arg-type]


_service_singleton: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = EmbeddingService()
    return _service_singleton

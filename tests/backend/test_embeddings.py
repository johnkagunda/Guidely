"""Tests for services.embeddings and services.cache.

NOTE: these require `sentence-transformers`, `torch`, and `numpy` to be
installed (see requirements.txt). The embedding *model* itself is
mocked out via monkeypatch so these tests run fast and offline; only
the caching/plumbing logic is under test, not model quality.
"""
import numpy as np
import pytest

from services.cache import EmbeddingCache
from services.embeddings import EmbeddingService


@pytest.fixture
def tmp_cache(tmp_path):
    return EmbeddingCache(cache_dir=tmp_path)


def test_cache_miss_then_hit(tmp_cache):
    text = "Employees receive 21 days of annual leave."
    assert tmp_cache.get(text) is None
    assert tmp_cache.misses == 1

    vec = np.array([0.1, 0.2, 0.3], dtype="float32")
    tmp_cache.put(text, vec)

    cached = tmp_cache.get(text)
    assert cached is not None
    np.testing.assert_array_equal(cached, vec)
    assert tmp_cache.hits == 1


def test_cache_key_is_content_based(tmp_cache):
    v1 = np.array([1.0, 2.0], dtype="float32")
    tmp_cache.put("document A content", v1)
    # Different text -> different key -> no accidental collision
    assert tmp_cache.get("document B content") is None


def test_cache_hit_rate_stats(tmp_cache):
    tmp_cache.put("a", np.array([1.0], dtype="float32"))
    tmp_cache.get("a")  # hit
    tmp_cache.get("a")  # hit
    tmp_cache.get("b")  # miss
    stats = tmp_cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(2 / 3, rel=1e-3)


class _FakeModel:
    """Stand-in for a SentenceTransformer so tests don't need real weights."""

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        if isinstance(texts, str):
            return np.array([len(texts), 0, 0, 0], dtype="float32")
        return np.array([[len(t), 0, 0, 0] for t in texts], dtype="float32")


def test_embed_text_uses_cache_on_second_call(monkeypatch, tmp_path):
    service = EmbeddingService(model_name="fake-model")
    service._model = _FakeModel()  # skip real model loading

    monkeypatch.setattr("services.embeddings.get_cache", lambda: EmbeddingCache(tmp_path))

    v1 = service.embed_text("hello", use_cache=True)
    v2 = service.embed_text("hello", use_cache=True)
    np.testing.assert_array_equal(v1, v2)


def test_embed_batch_only_calls_model_for_uncached_texts(monkeypatch, tmp_path):
    service = EmbeddingService(model_name="fake-model")
    calls = []

    class CountingModel(_FakeModel):
        def encode(self, texts, normalize_embeddings=True, batch_size=32):
            calls.append(list(texts))
            return super().encode(texts, normalize_embeddings, batch_size)

    service._model = CountingModel()
    cache = EmbeddingCache(tmp_path)
    monkeypatch.setattr("services.embeddings.get_cache", lambda: cache)

    service.embed_batch(["one", "two"], use_cache=True)
    service.embed_batch(["one", "three"], use_cache=True)

    # "one" should only ever be sent to the model once (first batch call)
    all_embedded = [t for call in calls for t in call]
    assert all_embedded.count("one") == 1
    assert "two" in all_embedded
    assert "three" in all_embedded

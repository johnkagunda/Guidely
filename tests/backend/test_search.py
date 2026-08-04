"""Tests for the /search API route.

NOTE: requires fastapi and httpx. Both the embedding model and the LLM
provider are monkeypatched so tests run fast and fully offline.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import services.vector_store as vs_module
    import services.rag as rag_module
    from services.vector_store import VectorStore
    from main import app

    monkeypatch.setattr(vs_module, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(vs_module, "CHUNKS_PATH", tmp_path / "chunks.json")

    store = VectorStore()
    vs_module._store_singleton = store

    # Seed the store with one known chunk so search has something to find.
    vectors = np.array([[1.0, 0.0]], dtype="float32")
    chunks = [{
        "chunk_id": "policy-001",
        "document": "company-policy.txt",
        "section": "Annual Leave",
        "text": "Employees are entitled to 21 working days of paid annual leave per year.",
    }]
    store.add_vectors(vectors, chunks)

    class FakeEmbeddingService:
        def embed_text(self, text, use_cache=True):
            return np.array([1.0, 0.0], dtype="float32")

    monkeypatch.setattr(rag_module, "get_embedding_service", lambda: FakeEmbeddingService())
    monkeypatch.setattr(
        rag_module, "generate_answer",
        lambda question, context: "Employees receive 21 working days of annual leave per year."
    )

    return TestClient(app)


def test_search_returns_answer_and_sources(client):
    resp = client.post("/search", json={"query": "How many annual leave days?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "21" in body["answer"]
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["document"] == "company-policy.txt"
    assert body["retrieved_chunks"] >= 1
    assert body["latency_ms"] >= 0


def test_search_rejects_empty_query(client):
    resp = client.post("/search", json={"query": ""})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_search_rejects_whitespace_only_query(client):
    resp = client.post("/search", json={"query": "   "})
    assert resp.status_code == 400


def test_search_no_results_returns_friendly_response(client, monkeypatch):
    import services.vector_store as vs_module
    from services.vector_store import VectorStore

    empty_store = VectorStore()
    vs_module._store_singleton = empty_store

    resp = client.post("/search", json={"query": "anything at all"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert body["retrieved_chunks"] == 0
    assert "couldn't find" in body["answer"].lower()


def test_search_handles_llm_failure_gracefully(client, monkeypatch):
    import services.rag as rag_module
    from services.llm import LLMError

    def raise_error(question, context):
        raise LLMError("simulated provider outage")

    monkeypatch.setattr(rag_module, "generate_answer", raise_error)

    resp = client.post("/search", json={"query": "How many annual leave days?"})
    assert resp.status_code == 502
    assert "detail" in resp.json()


def test_search_handles_llm_timeout(client, monkeypatch):
    import services.rag as rag_module
    from services.llm import LLMTimeoutError

    def raise_timeout(question, context):
        raise LLMTimeoutError("Ollama request timed out after 30s")

    monkeypatch.setattr(rag_module, "generate_answer", raise_timeout)

    resp = client.post("/search", json={"query": "How many annual leave days?"})
    assert resp.status_code == 504

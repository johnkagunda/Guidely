"""Tests for the /documents API routes.

NOTE: requires fastapi, httpx (for the TestClient), numpy, and faiss-cpu.
The embedding model is monkeypatched so no real model weights are needed.
"""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import routes.documents as docs_module
    import services.vector_store as vs_module
    from main import app

    # Redirect all filesystem state into a temp dir for test isolation.
    monkeypatch.setattr(docs_module, "UPLOAD_DIR", tmp_path / "uploads")
    docs_module.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(docs_module, "REGISTRY_PATH", tmp_path / "documents.json")
    monkeypatch.setattr(vs_module, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(vs_module, "CHUNKS_PATH", tmp_path / "chunks.json")

    # Fresh vector store per test, and a fake embedding service so no
    # real model download/inference is required.
    from services.vector_store import VectorStore
    vs_module._store_singleton = VectorStore()

    class FakeEmbeddingService:
        def embed_batch(self, texts, use_cache=True):
            return np.array([[len(t), 0.0] for t in texts], dtype="float32")

        def embed_text(self, text, use_cache=True):
            return np.array([len(text), 0.0], dtype="float32")

    monkeypatch.setattr(docs_module, "get_embedding_service", lambda: FakeEmbeddingService())

    return TestClient(app)


def _txt_file(name="policy.txt", content="Annual Leave\nEmployees get 21 days off per year."):
    return {"file": (name, io.BytesIO(content.encode("utf-8")), "text/plain")}


def test_upload_new_document_indexes_it(client):
    resp = client.post("/documents/upload", files=_txt_file())
    assert resp.status_code == 200
    body = resp.json()
    assert body["reindexed"] is True
    assert body["skipped_unchanged"] is False
    assert body["document"]["status"] == "indexed"
    assert body["document"]["chunk_count"] >= 1


def test_reupload_unchanged_document_is_skipped(client):
    client.post("/documents/upload", files=_txt_file())
    resp = client.post("/documents/upload", files=_txt_file())
    body = resp.json()
    assert body["skipped_unchanged"] is True
    assert body["reindexed"] is False


def test_reupload_changed_document_reindexes(client):
    client.post("/documents/upload", files=_txt_file(content="Version one text about leave."))
    resp = client.post(
        "/documents/upload",
        files=_txt_file(content="Version two text about leave, now with more detail added."),
    )
    body = resp.json()
    assert body["reindexed"] is True
    assert body["skipped_unchanged"] is False


def test_upload_rejects_unsupported_file_type(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("malware.exe", io.BytesIO(b"binary junk"), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


def test_upload_rejects_empty_file(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert resp.status_code == 400


def test_list_documents_after_upload(client):
    client.post("/documents/upload", files=_txt_file())
    resp = client.get("/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["documents"][0]["filename"] == "policy.txt"


def test_get_single_document(client):
    upload = client.post("/documents/upload", files=_txt_file()).json()
    doc_id = upload["document"]["id"]
    resp = client.get(f"/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["filename"] == "policy.txt"


def test_get_missing_document_returns_404(client):
    resp = client.get("/documents/does-not-exist")
    assert resp.status_code == 404


def test_delete_document_removes_it(client):
    upload = client.post("/documents/upload", files=_txt_file()).json()
    doc_id = upload["document"]["id"]
    resp = client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert client.get(f"/documents/{doc_id}").status_code == 404


def test_reindex_endpoint(client):
    upload = client.post("/documents/upload", files=_txt_file()).json()
    doc_id = upload["document"]["id"]
    resp = client.post(f"/documents/{doc_id}/reindex")
    assert resp.status_code == 200
    # unchanged content -> should report skipped_unchanged
    assert resp.json()["skipped_unchanged"] is True

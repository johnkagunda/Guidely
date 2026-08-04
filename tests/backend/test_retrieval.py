"""Tests for services.vector_store (FAISS-backed retrieval).

NOTE: requires `faiss-cpu` to be installed (see requirements.txt).
Uses small hand-built vectors so results are deterministic and do not
depend on the embedding model.
"""
import numpy as np
import pytest

from services.vector_store import VectorStore


def _chunk(chunk_id, document, text, section="General"):
    return {"chunk_id": chunk_id, "document": document, "section": section, "text": text}


@pytest.fixture
def store(tmp_path, monkeypatch):
    # Redirect persistence paths into a temp dir so tests never touch
    # the real backend/storage directory.
    import services.vector_store as vs_module
    monkeypatch.setattr(vs_module, "INDEX_PATH", tmp_path / "test.faiss")
    monkeypatch.setattr(vs_module, "CHUNKS_PATH", tmp_path / "chunks.json")
    return VectorStore()


def test_add_and_search_returns_closest_match(store):
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    chunks = [
        _chunk("c1", "doc1.txt", "about apples"),
        _chunk("c2", "doc1.txt", "about oranges"),
        _chunk("c3", "doc2.txt", "about cars"),
    ]
    store.add_vectors(vectors, chunks)

    query = np.array([0.9, 0.1, 0.0], dtype="float32")
    results = store.search(query, top_k=1)

    assert len(results) == 1
    top_meta, score = results[0]
    assert top_meta["chunk_id"] == "c1"


def test_search_respects_top_k(store):
    vectors = np.eye(5, dtype="float32")
    chunks = [_chunk(f"c{i}", "doc.txt", f"chunk {i}") for i in range(5)]
    store.add_vectors(vectors, chunks)

    results = store.search(np.array([1, 0, 0, 0, 0], dtype="float32"), top_k=3)
    assert len(results) == 3


def test_search_on_empty_store_returns_empty_list(store):
    results = store.search(np.array([1.0, 0.0], dtype="float32"), top_k=3)
    assert results == []


def test_remove_document_drops_only_its_chunks(store):
    vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype="float32")
    chunks = [
        _chunk("a1", "docA.txt", "alpha"),
        _chunk("a2", "docA.txt", "alpha two"),
        _chunk("b1", "docB.txt", "beta"),
    ]
    store.add_vectors(vectors, chunks)

    removed = store.remove_document("docA.txt")
    assert removed == 2
    assert store.chunk_count() == 1
    remaining_docs = {m["document"] for m in store.chunk_metadata.values()}
    assert remaining_docs == {"docB.txt"}


def test_save_and_load_round_trip(store, tmp_path):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    chunks = [_chunk("x1", "doc.txt", "x one"), _chunk("x2", "doc.txt", "x two")]
    store.add_vectors(vectors, chunks)
    store.save()

    reloaded = VectorStore()
    import services.vector_store as vs_module
    # reloaded instance shares the same monkeypatched paths via module state
    loaded = reloaded.load()
    assert loaded is True
    assert reloaded.chunk_count() == 2

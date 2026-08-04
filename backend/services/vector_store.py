"""FAISS-backed vector store service.

Responsibilities:
    - create/load a FAISS index
    - add vectors with associated chunk metadata
    - similarity search
    - persist the index (and chunk metadata) to disk
    - rebuild the index from scratch (e.g. after deleting a document)

The FAISS index only stores vectors + integer ids. Chunk metadata
(document, section, text) is kept in a parallel JSON file
(`storage/metadata/chunks.json`) keyed by the same integer id, since
FAISS itself is not a metadata store.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger, log_event

logger = get_logger("guidely.vector_store")

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
INDEX_DIR = STORAGE_DIR / "indexes"
METADATA_DIR = STORAGE_DIR / "metadata"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = INDEX_DIR / "guidely.faiss"
CHUNKS_PATH = METADATA_DIR / "chunks.json"


class VectorStoreError(Exception):
    pass


class VectorStore:
    """Thin, thread-safe wrapper around a FAISS flat index."""

    def __init__(self, dimension: Optional[int] = None):
        self._lock = threading.RLock()
        self.dimension = dimension
        self._index = None  # faiss.Index, created lazily once dimension known
        # id (int) -> chunk metadata dict {chunk_id, document, section, text}
        self.chunk_metadata: Dict[int, dict] = {}
        self._next_id = 0

    # -- index lifecycle ---------------------------------------------------

    def _ensure_index(self, dimension: int):
        if self._index is not None:
            return
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError(
                "faiss-cpu is not installed. Run `pip install faiss-cpu`."
            ) from exc
        self.dimension = dimension
        # IndexFlatIP over normalized vectors == cosine similarity search
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        log_event(logger, "faiss_index_created", dimension=dimension)

    def create_index(self, dimension: int) -> None:
        with self._lock:
            self._index = None
            self._ensure_index(dimension)
            self.chunk_metadata = {}
            self._next_id = 0

    def add_vectors(self, vectors: np.ndarray, chunks: List[dict]) -> List[int]:
        """Add vectors + their metadata dicts. Returns the assigned ids."""
        if len(vectors) != len(chunks):
            raise VectorStoreError("vectors and chunks length mismatch")
        if len(vectors) == 0:
            return []

        with self._lock:
            self._ensure_index(vectors.shape[1])
            ids = np.arange(self._next_id, self._next_id + len(vectors)).astype("int64")
            self._index.add_with_ids(vectors.astype("float32"), ids)
            for _id, chunk in zip(ids, chunks):
                self.chunk_metadata[int(_id)] = chunk
            self._next_id += len(vectors)
            log_event(logger, "vectors_added", count=len(vectors))
            return ids.tolist()

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Return up to top_k (chunk_metadata, score) pairs, best first."""
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
            k = min(top_k, self._index.ntotal)
            scores, ids = self._index.search(q, k)
            results = []
            for score, _id in zip(scores[0], ids[0]):
                if _id == -1:
                    continue
                meta = self.chunk_metadata.get(int(_id))
                if meta is not None:
                    results.append((meta, float(score)))
            return results

    def remove_document(self, document_name: str) -> int:
        """Remove all vectors belonging to a given document, then rebuild."""
        with self._lock:
            remaining = [
                (meta) for meta in self.chunk_metadata.values()
                if meta["document"] != document_name
            ]
            removed_count = len(self.chunk_metadata) - len(remaining)
            self.rebuild_from_chunks(remaining)
            return removed_count

    def rebuild_from_chunks(self, chunks: List[dict]) -> None:
        """Rebuild the entire index from a fresh list of chunk dicts.

        Each chunk dict must contain an 'embedding' key (list[float]) in
        addition to chunk_id/document/section/text. Used after deletions
        or bulk re-indexing to keep FAISS ids contiguous and clean.
        """
        with self._lock:
            if not chunks:
                self._index = None
                self.chunk_metadata = {}
                self._next_id = 0
                self.save()
                return

            dim = len(chunks[0]["embedding"])
            self._index = None
            self._ensure_index(dim)
            self.chunk_metadata = {}
            self._next_id = 0

            vectors = np.array([c["embedding"] for c in chunks], dtype="float32")
            meta_only = [
                {k: v for k, v in c.items() if k != "embedding"} for c in chunks
            ]
            self.add_vectors(vectors, meta_only)
            self.save()
            log_event(logger, "index_rebuilt", chunk_count=len(chunks))

    # -- persistence ---------------------------------------------------------

    def save(self) -> None:
        with self._lock:
            try:
                import faiss
            except ImportError as exc:
                raise VectorStoreError("faiss-cpu is not installed.") from exc

            if self._index is not None:
                faiss.write_index(self._index, str(INDEX_PATH))
            else:
                if INDEX_PATH.exists():
                    INDEX_PATH.unlink()

            serializable = {str(k): v for k, v in self.chunk_metadata.items()}
            CHUNKS_PATH.write_text(
                json.dumps(
                    {
                        "dimension": self.dimension,
                        "next_id": self._next_id,
                        "chunks": serializable,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            log_event(logger, "index_saved", chunk_count=len(self.chunk_metadata))

    def load(self) -> bool:
        """Load a previously persisted index + metadata from disk.

        Returns True if an existing index was found and loaded.
        """
        with self._lock:
            if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
                return False
            try:
                import faiss
            except ImportError as exc:
                raise VectorStoreError("faiss-cpu is not installed.") from exc

            self._index = faiss.read_index(str(INDEX_PATH))
            data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
            self.dimension = data.get("dimension")
            self._next_id = data.get("next_id", 0)
            self.chunk_metadata = {int(k): v for k, v in data.get("chunks", {}).items()}
            log_event(
                logger, "index_loaded", chunk_count=len(self.chunk_metadata)
            )
            return True

    # -- introspection ------------------------------------------------------

    def chunk_count(self) -> int:
        with self._lock:
            return len(self.chunk_metadata)

    def all_chunks_for_document(self, document_name: str) -> List[dict]:
        with self._lock:
            return [
                m for m in self.chunk_metadata.values()
                if m["document"] == document_name
            ]


_store_singleton: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = VectorStore()
        try:
            _store_singleton.load()
        except VectorStoreError as exc:
            log_event(logger, "vector_store_load_failed", error=str(exc))
    return _store_singleton

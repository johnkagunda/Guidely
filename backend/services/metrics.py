"""In-process metrics tracker, persisted to disk so counts survive restarts.

Tracks query latency, embedding cache performance, indexing stats, and
categorized error counts as described in the project spec.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage" / "logs"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = STORAGE_DIR / "metrics.json"

ERROR_CATEGORIES = [
    "empty_query",
    "missing_model",
    "corrupted_document",
    "unsupported_file",
    "no_results",
    "embedding_failure",
    "llm_failure",
    "timeout",
]


class MetricsTracker:
    def __init__(self):
        self._lock = threading.RLock()
        self.query_latencies_ms: List[float] = []
        self.queries_successful = 0
        self.queries_failed = 0

        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0
        self.embeddings_generated = 0

        self.documents_indexed = 0
        self.chunks_created = 0
        self.total_indexing_duration_ms = 0.0
        self.skipped_unchanged = 0

        self.errors: Dict[str, int] = {k: 0 for k in ERROR_CATEGORIES}

        self._load()

    # -- query metrics -------------------------------------------------

    def record_query(self, latency_ms: float, success: bool) -> None:
        with self._lock:
            self.query_latencies_ms.append(latency_ms)
            if success:
                self.queries_successful += 1
            else:
                self.queries_failed += 1
            self._save()

    def record_error(self, category: str) -> None:
        with self._lock:
            if category not in self.errors:
                self.errors[category] = 0
            self.errors[category] += 1
            self._save()

    # -- embedding metrics -----------------------------------------------

    def record_cache_hit(self) -> None:
        with self._lock:
            self.embedding_cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self.embedding_cache_misses += 1

    def record_embedding_generated(self) -> None:
        with self._lock:
            self.embeddings_generated += 1
            self._save()

    # -- indexing metrics -------------------------------------------------

    def record_indexing(self, chunks: int, duration_ms: float, skipped: bool) -> None:
        with self._lock:
            if skipped:
                self.skipped_unchanged += 1
            else:
                self.documents_indexed += 1
                self.chunks_created += chunks
                self.total_indexing_duration_ms += duration_ms
            self._save()

    # -- readout ------------------------------------------------------------

    def latency_stats(self) -> dict:
        with self._lock:
            if not self.query_latencies_ms:
                return {"median_ms": 0.0, "p95_ms": 0.0, "average_ms": 0.0}
            sorted_lat = sorted(self.query_latencies_ms)
            p95_idx = max(0, int(round(0.95 * (len(sorted_lat) - 1))))
            return {
                "median_ms": round(median(sorted_lat), 2),
                "p95_ms": round(sorted_lat[p95_idx], 2),
                "average_ms": round(sum(sorted_lat) / len(sorted_lat), 2),
            }

    def embedding_cache_stats(self) -> dict:
        with self._lock:
            total = self.embedding_cache_hits + self.embedding_cache_misses
            hit_rate = (self.embedding_cache_hits / total) if total else 0.0
            return {
                "hits": self.embedding_cache_hits,
                "misses": self.embedding_cache_misses,
                "hit_rate": round(hit_rate, 4),
                "embeddings_generated": self.embeddings_generated,
            }

    def indexing_stats(self) -> dict:
        with self._lock:
            return {
                "documents_indexed": self.documents_indexed,
                "chunks_created": self.chunks_created,
                "total_indexing_duration_ms": round(self.total_indexing_duration_ms, 2),
                "skipped_unchanged": self.skipped_unchanged,
            }

    def snapshot(self, documents: int, chunks: int) -> dict:
        with self._lock:
            return {
                "documents": documents,
                "chunks": chunks,
                "queries_served": self.queries_successful + self.queries_failed,
                "queries_successful": self.queries_successful,
                "queries_failed": self.queries_failed,
                "latency": self.latency_stats(),
                "embedding_cache": self.embedding_cache_stats(),
                "indexing": self.indexing_stats(),
                "errors": dict(self.errors),
            }

    # -- persistence ----------------------------------------------------------

    def _save(self) -> None:
        data = {
            "query_latencies_ms": self.query_latencies_ms[-1000:],  # cap growth
            "queries_successful": self.queries_successful,
            "queries_failed": self.queries_failed,
            "embedding_cache_hits": self.embedding_cache_hits,
            "embedding_cache_misses": self.embedding_cache_misses,
            "embeddings_generated": self.embeddings_generated,
            "documents_indexed": self.documents_indexed,
            "chunks_created": self.chunks_created,
            "total_indexing_duration_ms": self.total_indexing_duration_ms,
            "skipped_unchanged": self.skipped_unchanged,
            "errors": self.errors,
        }
        try:
            METRICS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass  # metrics persistence is best-effort, never fatal

    def _load(self) -> None:
        if not METRICS_PATH.exists():
            return
        try:
            data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self.query_latencies_ms = data.get("query_latencies_ms", [])
        self.queries_successful = data.get("queries_successful", 0)
        self.queries_failed = data.get("queries_failed", 0)
        self.embedding_cache_hits = data.get("embedding_cache_hits", 0)
        self.embedding_cache_misses = data.get("embedding_cache_misses", 0)
        self.embeddings_generated = data.get("embeddings_generated", 0)
        self.documents_indexed = data.get("documents_indexed", 0)
        self.chunks_created = data.get("chunks_created", 0)
        self.total_indexing_duration_ms = data.get("total_indexing_duration_ms", 0.0)
        self.skipped_unchanged = data.get("skipped_unchanged", 0)
        loaded_errors = data.get("errors", {})
        for k in ERROR_CATEGORIES:
            self.errors[k] = loaded_errors.get(k, 0)


_tracker_singleton: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    global _tracker_singleton
    if _tracker_singleton is None:
        _tracker_singleton = MetricsTracker()
    return _tracker_singleton

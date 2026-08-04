"""RAG pipeline orchestration.

Question -> validate -> embed question -> FAISS search -> build context
-> LLM -> answer + sources -> JSON response.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List

from services.embeddings import EmbeddingModelError, get_embedding_service
from services.llm import LLMError, generate_answer
from services.metrics import get_metrics_tracker
from services.vector_store import get_vector_store
from utils.logger import get_logger, log_event

logger = get_logger("guidely.rag")

DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
SNIPPET_LENGTH = 240


class EmptyQueryError(Exception):
    pass


class NoResultsError(Exception):
    """Raised when the vector store has nothing relevant (or is empty)."""


@dataclass
class RagSource:
    document: str
    section: str | None
    snippet: str
    score: float


@dataclass
class RagResult:
    answer: str
    sources: List[RagSource]
    retrieved_chunks: int
    latency_ms: float


def _snippet(text: str, length: int = SNIPPET_LENGTH) -> str:
    text = text.strip()
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + "..."


def _build_context(chunks_with_scores) -> str:
    parts = []
    total_len = 0
    for meta, _score in chunks_with_scores:
        block = f"[{meta['document']} - {meta.get('section') or 'General'}]\n{meta['text']}"
        if total_len + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total_len += len(block)
    return "\n\n---\n\n".join(parts)


def answer_question(query: str, top_k: int | None = None) -> RagResult:
    """Run the full RAG pipeline for a user question.

    Raises EmptyQueryError, NoResultsError, EmbeddingModelError, or
    LLMError so the route layer can translate them into the right HTTP
    status codes.
    """
    metrics = get_metrics_tracker()
    start = time.perf_counter()

    if query is None or query.strip() == "":
        metrics.record_error("empty_query")
        raise EmptyQueryError("Query cannot be empty")

    k = top_k or DEFAULT_TOP_K
    store = get_vector_store()

    if store.chunk_count() == 0:
        metrics.record_error("no_results")
        raise NoResultsError(
            "No documents have been indexed yet. Upload documents before searching."
        )

    embedding_service = get_embedding_service()
    try:
        query_vector = embedding_service.embed_text(query, use_cache=False)
    except EmbeddingModelError as exc:
        metrics.record_error("embedding_failure")
        raise

    results = store.search(query_vector, top_k=k)
    if not results:
        metrics.record_error("no_results")
        raise NoResultsError("No relevant information was found for this question.")

    context = _build_context(results)

    try:
        answer_text = generate_answer(query, context)
    except LLMError as exc:
        category = "timeout" if "timed out" in str(exc).lower() else "llm_failure"
        metrics.record_error(category)
        raise

    sources = [
        RagSource(
            document=meta["document"],
            section=meta.get("section"),
            snippet=_snippet(meta["text"]),
            score=round(score, 4),
        )
        for meta, score in results
    ]

    latency_ms = (time.perf_counter() - start) * 1000
    metrics.record_query(latency_ms, success=True)
    log_event(
        logger,
        "query_answered",
        query=query,
        retrieved_chunks=len(results),
        latency_ms=round(latency_ms, 2),
    )

    return RagResult(
        answer=answer_text,
        sources=sources,
        retrieved_chunks=len(results),
        latency_ms=round(latency_ms, 2),
    )

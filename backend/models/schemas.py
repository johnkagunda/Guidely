"""Pydantic request/response schemas for the Guidely API."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentInfo(BaseModel):
    """Metadata describing a single indexed (or pending) document."""

    id: str
    filename: str
    file_type: str
    size_bytes: int
    status: str  # "indexed" | "pending" | "failed"
    content_hash: Optional[str] = None
    chunk_count: int = 0
    last_indexed_at: Optional[str] = None
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class UploadResponse(BaseModel):
    document: DocumentInfo
    reindexed: bool
    skipped_unchanged: bool
    indexing_duration_ms: Optional[float] = None


class ReindexResponse(BaseModel):
    document: DocumentInfo
    skipped_unchanged: bool
    indexing_duration_ms: Optional[float] = None


class DeleteResponse(BaseModel):
    id: str
    deleted: bool


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if v is None or v.strip() == "":
            raise ValueError("Query cannot be empty")
        return v


class SourceRef(BaseModel):
    document: str
    section: Optional[str] = None
    snippet: str
    score: float


class SearchResponse(BaseModel):
    answer: str
    sources: List[SourceRef]
    retrieved_chunks: int
    latency_ms: float


class ErrorResponse(BaseModel):
    error: str


# ---------------------------------------------------------------------------
# Health / Metrics
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    llm_provider: str
    vector_store: str
    documents: int
    chunks: int


class LatencyStats(BaseModel):
    median_ms: float
    p95_ms: float
    average_ms: float


class EmbeddingCacheStats(BaseModel):
    hits: int
    misses: int
    hit_rate: float
    embeddings_generated: int


class IndexingStats(BaseModel):
    documents_indexed: int
    chunks_created: int
    total_indexing_duration_ms: float
    skipped_unchanged: int


class MetricsResponse(BaseModel):
    documents: int
    chunks: int
    queries_served: int
    queries_successful: int
    queries_failed: int
    latency: LatencyStats
    embedding_cache: EmbeddingCacheStats
    indexing: IndexingStats
    errors: dict

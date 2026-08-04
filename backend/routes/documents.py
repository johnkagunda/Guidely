"""Document upload, listing, retrieval, deletion, and re-indexing routes."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from models.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    ReindexResponse,
    UploadResponse,
)
from services import parser
from services.chunker import chunk_document
from services.embeddings import EmbeddingModelError, get_embedding_service
from services.metrics import get_metrics_tracker
from services.vector_store import VectorStoreError, get_vector_store
from utils.hashing import sha256_text
from utils.logger import get_logger, log_event

logger = get_logger("guidely.documents")
router = APIRouter(prefix="/documents", tags=["documents"])

BACKEND_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BACKEND_DIR / "data" / "uploaded-docs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = BACKEND_DIR / "storage" / "metadata" / "documents.json"
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Document registry (simple JSON file, filesystem storage per project spec)
# ---------------------------------------------------------------------------

def _load_registry() -> Dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_registry(registry: Dict[str, dict]) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _to_info(doc_id: str, record: dict) -> DocumentInfo:
    return DocumentInfo(
        id=doc_id,
        filename=record["filename"],
        file_type=record["file_type"],
        size_bytes=record["size_bytes"],
        status=record["status"],
        content_hash=record.get("content_hash"),
        chunk_count=record.get("chunk_count", 0),
        last_indexed_at=record.get("last_indexed_at"),
        uploaded_at=record["uploaded_at"],
    )


# ---------------------------------------------------------------------------
# Indexing helper (shared by upload + reindex)
# ---------------------------------------------------------------------------

def _index_document(doc_id: str, record: dict, text: str, content_hash: str) -> dict:
    """Chunk, embed, and add a document's content to the vector store.

    Returns the updated record. Assumes any previous chunks for this
    document have already been removed from the vector store.
    """
    metrics = get_metrics_tracker()
    start = time.perf_counter()

    chunks = chunk_document(text, record["filename"])
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no usable text chunks")

    embedding_service = get_embedding_service()
    texts = [c.text for c in chunks]
    try:
        vectors = embedding_service.embed_batch(texts)
    except EmbeddingModelError as exc:
        metrics.record_error("embedding_failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    store = get_vector_store()
    chunk_dicts = [c.to_dict() for c in chunks]
    store.add_vectors(vectors, chunk_dicts)
    store.save()

    duration_ms = (time.perf_counter() - start) * 1000
    metrics.record_indexing(chunks=len(chunks), duration_ms=duration_ms, skipped=False)

    record["status"] = "indexed"
    record["content_hash"] = content_hash
    record["chunk_count"] = len(chunks)
    record["last_indexed_at"] = datetime.now(timezone.utc).isoformat()
    log_event(
        logger,
        "document_indexed",
        document=record["filename"],
        chunks=len(chunks),
        duration_ms=round(duration_ms, 2),
    )
    return record, duration_ms


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("", response_model=DocumentListResponse)
def list_documents():
    registry = _load_registry()
    docs = [_to_info(doc_id, rec) for doc_id, rec in registry.items()]
    docs.sort(key=lambda d: d.uploaded_at, reverse=True)
    return DocumentListResponse(documents=docs, total=len(docs))


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not parser.is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type. Supported types: "
                f"{', '.join(sorted(parser.SUPPORTED_EXTENSIONS))}"
            ),
        )

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds the 20MB upload limit")

    registry = _load_registry()

    # Reuse an existing document id if this filename was already uploaded
    existing_id = next(
        (did for did, rec in registry.items() if rec["filename"] == file.filename),
        None,
    )
    doc_id = existing_id or str(uuid.uuid4())

    dest_path = UPLOAD_DIR / f"{doc_id}__{file.filename}"
    dest_path.write_bytes(raw_bytes)

    try:
        text = parser.extract_text(dest_path, file.filename)
    except parser.UnsupportedFileTypeError as exc:
        get_metrics_tracker().record_error("unsupported_file")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except parser.CorruptedDocumentError as exc:
        get_metrics_tracker().record_error("corrupted_document")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    content_hash = sha256_text(text)
    previous_hash = registry.get(doc_id, {}).get("content_hash")

    record = registry.get(doc_id, {
        "filename": file.filename,
        "file_type": Path(file.filename).suffix.lower(),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "chunk_count": 0,
    })
    record["size_bytes"] = len(raw_bytes)
    record["file_type"] = Path(file.filename).suffix.lower()

    if previous_hash == content_hash:
        get_metrics_tracker().record_indexing(chunks=0, duration_ms=0, skipped=True)
        registry[doc_id] = record
        _save_registry(registry)
        return UploadResponse(
            document=_to_info(doc_id, record),
            reindexed=False,
            skipped_unchanged=True,
            indexing_duration_ms=None,
        )

    # remove old vectors for this document before re-adding, if any
    get_vector_store().remove_document(record["filename"])

    record, duration_ms = _index_document(doc_id, record, text, content_hash)
    registry[doc_id] = record
    _save_registry(registry)

    return UploadResponse(
        document=_to_info(doc_id, record),
        reindexed=True,
        skipped_unchanged=False,
        indexing_duration_ms=round(duration_ms, 2),
    )


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str):
    registry = _load_registry()
    record = registry.get(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _to_info(doc_id, record)


@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    registry = _load_registry()
    record = registry.get(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    get_vector_store().remove_document(record["filename"])

    for path in UPLOAD_DIR.glob(f"{doc_id}__*"):
        path.unlink(missing_ok=True)

    del registry[doc_id]
    _save_registry(registry)

    return DeleteResponse(id=doc_id, deleted=True)


@router.post("/{doc_id}/reindex", response_model=ReindexResponse)
def reindex_document(doc_id: str):
    registry = _load_registry()
    record = registry.get(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    matches = list(UPLOAD_DIR.glob(f"{doc_id}__*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Stored file for this document is missing")
    dest_path = matches[0]

    try:
        text = parser.extract_text(dest_path, record["filename"])
    except parser.CorruptedDocumentError as exc:
        get_metrics_tracker().record_error("corrupted_document")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    content_hash = sha256_text(text)
    if record.get("content_hash") == content_hash:
        get_metrics_tracker().record_indexing(chunks=0, duration_ms=0, skipped=True)
        return ReindexResponse(
            document=_to_info(doc_id, record),
            skipped_unchanged=True,
            indexing_duration_ms=None,
        )

    get_vector_store().remove_document(record["filename"])
    record, duration_ms = _index_document(doc_id, record, text, content_hash)
    registry[doc_id] = record
    _save_registry(registry)

    return ReindexResponse(
        document=_to_info(doc_id, record),
        skipped_unchanged=False,
        indexing_duration_ms=round(duration_ms, 2),
    )

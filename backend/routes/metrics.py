"""Metrics reporting route."""
from __future__ import annotations

from fastapi import APIRouter

from models.schemas import MetricsResponse
from routes.documents import _load_registry
from services.metrics import get_metrics_tracker
from services.vector_store import get_vector_store

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    registry = _load_registry()
    store = get_vector_store()
    tracker = get_metrics_tracker()
    snapshot = tracker.snapshot(documents=len(registry), chunks=store.chunk_count())
    return MetricsResponse(**snapshot)

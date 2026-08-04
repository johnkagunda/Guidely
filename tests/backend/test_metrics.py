"""Tests for /metrics and /health endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import services.metrics as metrics_module
    import routes.documents as docs_module
    from main import app

    monkeypatch.setattr(metrics_module, "METRICS_PATH", tmp_path / "metrics.json")
    metrics_module._tracker_singleton = metrics_module.MetricsTracker()

    monkeypatch.setattr(docs_module, "REGISTRY_PATH", tmp_path / "documents.json")

    return TestClient(app)


def test_health_endpoint_shape(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "embedding_model" in body
    assert "documents" in body
    assert "chunks" in body


def test_metrics_endpoint_shape(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "documents" in body
    assert "chunks" in body
    assert "queries_served" in body
    assert "latency" in body
    assert set(body["latency"].keys()) == {"median_ms", "p95_ms", "average_ms"}
    assert "embedding_cache" in body
    assert "errors" in body


def test_metrics_reflect_recorded_query(client):
    from services.metrics import get_metrics_tracker

    tracker = get_metrics_tracker()
    tracker.record_query(latency_ms=1200, success=True)
    tracker.record_query(latency_ms=1800, success=True)

    resp = client.get("/metrics")
    body = resp.json()
    assert body["queries_served"] == 2
    assert body["latency"]["median_ms"] == 1500.0


def test_metrics_track_errors_by_category(client):
    from services.metrics import get_metrics_tracker

    tracker = get_metrics_tracker()
    tracker.record_error("empty_query")
    tracker.record_error("empty_query")
    tracker.record_error("llm_failure")

    resp = client.get("/metrics")
    body = resp.json()
    assert body["errors"]["empty_query"] == 2
    assert body["errors"]["llm_failure"] == 1

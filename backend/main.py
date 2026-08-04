"""Guidely backend entrypoint.

Run with:  uvicorn main:app --reload --port 8000
(from inside the backend/ directory, so relative imports resolve)
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.schemas import HealthResponse
from routes import documents, metrics, search
from routes.documents import _load_registry
from services.vector_store import get_vector_store
from utils.logger import get_logger, log_event

logger = get_logger("guidely.main")

app = FastAPI(
    title="Guidely",
    description="Internal knowledge assistant powered by retrieval-augmented generation.",
    version="1.0.0",
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak raw stack traces to the client."""
    log_event(logger, "unhandled_exception", path=str(request.url), error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected server error occurred. Please try again."},
    )


@app.on_event("startup")
def on_startup():
    # Loading the FAISS index (if it exists) at startup avoids ever
    # re-embedding documents just because the process restarted.
    store = get_vector_store()
    log_event(logger, "startup_complete", chunks_loaded=store.chunk_count())


app.include_router(documents.router)
app.include_router(search.router)
app.include_router(metrics.router)


@app.get("/health", response_model=HealthResponse)
def health():
    store = get_vector_store()
    registry = _load_registry()
    return HealthResponse(
        status="healthy",
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        vector_store="ready" if store.chunk_count() > 0 else "empty",
        documents=len(registry),
        chunks=store.chunk_count(),
    )


@app.get("/")
def root():
    return {"service": "Guidely API", "docs": "/docs", "health": "/health"}

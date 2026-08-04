# Guidely

Guidely is an internal knowledge assistant. Upload company documents, and
Guidely indexes them with vector embeddings so your team can ask
questions in plain language and get concise answers with clear source
citations — no answer is invented; if the documents don't contain it,
Guidely says so.

> **Status note (read first):** this project was built in a sandboxed
> environment with no network access, so dependencies (FastAPI, FAISS,
> sentence-transformers, npm packages) could not be installed and the
> app could not actually be run end-to-end there. All code was written,
> and everything listed under "What has actually been verified" below
> really was checked. Everything else is untested and the evaluation
> table stays `TBD` until you run it yourself locally, per the project
> requirements — no numbers here are fabricated.

## What has actually been verified

- All 28 backend `.py` files parse with Python's `ast` module (no syntax errors).
- All frontend `.jsx` files compile successfully through esbuild (no syntax errors, imports resolve).
- `services/chunker.py` was executed directly against all 5 sample documents and produces sensible, section-labeled chunks (see output shape below).
- `tests/backend/test_chunking.py` (8 tests) and `tests/backend/test_hashing.py` (5 tests) were actually run with plain Python assertions and **pass** — these two modules have zero external dependencies.
- Everything else — the FastAPI routes, FAISS retrieval, embedding model, LLM calls, the React app actually rendering in a browser, and the evaluation script — is written but **not yet executed**, because this environment can't install `fastapi`, `faiss-cpu`, `sentence-transformers`, `torch`, or run `npm install` / `ollama serve`. Run it locally following the steps below, then fill in the results table.

## 1. What Guidely is

An internal RAG (retrieval-augmented generation) assistant: upload
`.txt`/`.md`/`.pdf`/`.docx` documents, Guidely chunks and embeds them,
stores the vectors in FAISS, and answers natural-language questions by
retrieving the most relevant chunks and asking an LLM to answer using
only that retrieved context.

## 2. Features

- Document upload with automatic parsing, chunking, and indexing
- Content-hash based change detection — re-uploading an unchanged file skips re-indexing
- Embedding cache keyed by content hash — never re-embeds unchanged text
- FAISS vector search with persisted index (survives backend restarts)
- Configurable LLM backend: local Ollama (default, free) or any OpenAI-compatible API
- Answers grounded strictly in retrieved context, with cited sources
- Document management (list, delete, re-index) via an admin dashboard
- Automatic metrics: query latency (median/p95), embedding cache hit rate, indexing stats, categorized error counts
- Friendly error handling for empty queries, no results, timeouts, unsupported files, and corrupted documents

## 3. Architecture

```
React (Vite) frontend  --HTTP-->  FastAPI backend
                                       |
                          +------------+------------+
                          |            |             |
                     Parser/Chunker  Embeddings   LLM (Ollama/OpenAI)
                          |            |
                          +--> FAISS vector store <--+
                          |
                    Filesystem storage (index, metadata, cache, logs)
```

Request flow for a question:

```
Question -> validate -> embed question -> FAISS search (top_k)
-> build context from retrieved chunks -> LLM generates answer
-> return {answer, sources, retrieved_chunks, latency_ms}
```

## 4. Project structure

```
guidely/
├── frontend/        React + Vite dashboard (Search, Admin, Metrics pages)
├── backend/         FastAPI app, RAG services, sample docs, filesystem storage
├── tests/backend/   pytest suite
├── tests/evaluation/ evaluation_queries.json (+ evaluation_results.json after running)
├── scripts/         evaluate_retrieval.py (manual Retrieval@3 harness)
├── requirements.txt
├── .env.example
└── README.md
```

(Full tree matches the original spec — see the repo for exact file layout.)

## 5. Technology stack

- **Frontend:** React, Vite, React Router, vanilla CSS
- **Backend:** Python, FastAPI, Uvicorn, Pydantic, python-dotenv
- **RAG:** FAISS (`faiss-cpu`), `sentence-transformers` (`all-MiniLM-L6-v2` by default), Ollama or any OpenAI-compatible API
- **Storage:** local filesystem only (no database)

## 6. Installation

```bash
git clone <this-repo>
cd guidely

# Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

## 7. Environment setup

```bash
cp .env.example .env
# edit .env: set LLM_PROVIDER, OLLAMA_MODEL or OPENAI_API_KEY, etc.
```

Never commit your real `.env` file — it's already in `.gitignore`.

## 8. Running Ollama (default, free local LLM)

```bash
# install from https://ollama.com, then:
ollama serve
ollama pull llama3.2
```

Leave `ollama serve` running in a terminal. Guidely's default `.env`
points at `http://localhost:11434` with model `llama3.2`.

To use an OpenAI-compatible API instead, set `LLM_PROVIDER=openai` and
`OPENAI_API_KEY` in `.env`.

## 9. Running FastAPI

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API docs, and
`http://localhost:8000/health` for a health check.

## 10. Running React

```bash
cd frontend
npm run dev
```

Visit `http://localhost:5173`. The dev server proxies `/api/*` to the
backend on port 8000 (see `vite.config.js`).

## 11. Uploading documents

Five realistic sample documents ship in `backend/data/sample-docs/`
(company policy, employee FAQ, employee guide, security policy,
onboarding guide). Upload them either:

- Through the **Documents** page in the UI, or
- Directly via the API:

```bash
for f in backend/data/sample-docs/*.txt; do
  curl -F "file=@$f" http://localhost:8000/documents/upload
done
```

## 12. Indexing

Indexing happens automatically on upload: parse -> hash -> chunk ->
embed -> add to FAISS -> persist. Re-uploading an unchanged file is
detected via SHA256 content hash and skipped. Use the **Re-index**
button (or `POST /documents/{id}/reindex`) to force re-processing of a
changed file already on disk.

## 13. Asking questions

Use the **Search** page, or:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How many annual leave days do employees receive?"}'
```

## 14. RAG pipeline

See "Architecture" above. Context is capped at `MAX_CONTEXT_CHARS`
(default 6000) to keep LLM calls fast and cheap; `top_k` (default 3,
configurable via `TOP_K` or per-request) controls how many chunks are
retrieved.

## 15. Embedding cache

`backend/services/cache.py` stores one `.npy` vector per unique chunk
text, keyed by SHA256 hash, under `backend/storage/cache/`. Re-indexing
unchanged text is a cache hit, not a new model call.

## 16. FAISS storage

The index lives at `backend/storage/indexes/guidely.faiss`; chunk
metadata (document, section, text) lives alongside it in
`backend/storage/metadata/chunks.json`. Both are loaded on backend
startup so restarting the server never forces re-embedding.

## 17. API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/documents` | List documents |
| POST | `/documents/upload` | Upload + index a document |
| GET | `/documents/{id}` | Get one document's metadata |
| DELETE | `/documents/{id}` | Delete a document |
| POST | `/documents/{id}/reindex` | Force re-index |
| POST | `/search` | Ask a question |
| GET | `/metrics` | Metrics snapshot |

## 18. Testing

```bash
pip install -r requirements.txt   # includes pytest
pytest tests/backend -v
```

`test_chunking.py` and `test_hashing.py` have no external dependencies
and were confirmed passing during development (see "What has actually
been verified"). The remaining suites (`test_embeddings.py`,
`test_retrieval.py`, `test_documents.py`, `test_search.py`,
`test_metrics.py`) mock the embedding model and LLM provider so they
run fast and offline, but do require `fastapi`, `httpx`, `numpy`, and
`faiss-cpu` to be installed — run them locally and update this section
with the real pass/fail count.

## 19. Evaluation methodology

`tests/evaluation/evaluation_queries.json` has 20 questions (19
answerable from the sample docs + 1 deliberately unanswerable control
question, to check Guidely correctly declines instead of hallucinating).

With the backend running and the sample docs indexed:

```bash
python scripts/evaluate_retrieval.py --base-url http://localhost:8000
```

This makes real HTTP calls to `/search` for every question and computes:

- **Retrieval@3** — expected document appears in the top-3 retrieved sources
- **Source precision** — top-1 retrieved source matches the expected document
- **Answer reference coverage** — answerable queries that returned at least one source

Results are written to `tests/evaluation/evaluation_results.json`.

## 20. Metrics

`GET /metrics` reports query counts, median/p95 latency, embedding
cache hit rate, indexing stats, and categorized error counts — all
tracked automatically as the app is used (see the **Metrics** page in
the UI).

## 21. Known limitations

- No database — filesystem-only storage; not designed for concurrent multi-instance deployment.
- PDF parsing extracts text only; scanned/image-only PDFs will fail with a clear error (no OCR).
- Section detection in chunking uses a heuristic (markdown headers or short standalone lines), not a layout-aware parser — documents without clear headings fall back to one section per file.
- No authentication/authorization layer — intended for trusted internal use behind existing network/VPN controls.
- FAISS `IndexFlatIP` does exact search, which is simple and accurate but doesn't scale to very large corpora (approximate indexes like HNSW would be a future upgrade).
- This build has not been run end-to-end (see status note at the top) — treat it as a complete but freshly-written codebase that needs its first real test pass, not a battle-tested one.

## 22. Future improvements

- Swap `IndexFlatIP` for an approximate FAISS index (HNSW/IVF) at scale
- Add OCR fallback for scanned PDFs
- Add streaming responses for the LLM answer
- Add user authentication and per-document access control
- Add a re-ranking step after initial retrieval
- Add citation highlighting that maps directly back into the source document

## Evaluation results

| Metric                    |                Target | Actual |
| ------------------------- | --------------------: | -----: |
| Retrieval@3               |                 ≥ 80% |    TBD |
| Answer reference coverage |                 ≥ 90% |    TBD |
| Median latency            |                  < 3s |    TBD |
| P95 latency               |                  < 5s |    TBD |
| Embedding cache hit rate  | 100% repeated queries |    TBD |
| Source precision          |                 ≥ 80% |    TBD |
| Failure handling          |                  Pass |    TBD |

Run `pytest tests/backend -v` and `python scripts/evaluate_retrieval.py`
locally, then replace the `TBD` values above with the real results.

"""Manual evaluation harness for Guidely.

Runs every question in tests/evaluation/evaluation_queries.json against
a LIVE Guidely backend (started separately, e.g. `uvicorn main:app`)
and computes:

    Retrieval@3          = queries where the expected document appears
                            among the top-3 retrieved sources / total queries
    Answer reference coverage = successful answers that cite at least
                            one source / total answerable queries
    Source precision      = top-1 retrieved source's document matches
                            the expected document / total answerable queries

Usage:
    python scripts/evaluate_retrieval.py [--base-url http://localhost:8000]

Requires the backend to be running with the sample documents already
uploaded and indexed (see README "Uploading documents").

This script performs REAL HTTP calls; it does not fabricate results.
If the backend is unreachable, it exits with an error rather than
printing placeholder numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_FILE = Path(__file__).resolve().parent.parent / "tests" / "evaluation" / "evaluation_queries.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Guidely retrieval and answer quality.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    try:
        import httpx
    except ImportError:
        print("This script requires httpx: pip install httpx", file=sys.stderr)
        return 1

    if not EVAL_FILE.exists():
        print(f"Evaluation file not found: {EVAL_FILE}", file=sys.stderr)
        return 1

    queries = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    try:
        health = httpx.get(f"{args.base_url}/health", timeout=5)
        health.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"Could not reach Guidely backend at {args.base_url}: {exc}", file=sys.stderr)
        print("Start it first with: uvicorn main:app --reload (from backend/)", file=sys.stderr)
        return 1

    answerable = [q for q in queries if q.get("expected_document")]
    unanswerable = [q for q in queries if not q.get("expected_document")]

    retrieval_hits = 0
    top1_matches = 0
    answers_with_sources = 0
    correctly_declined = 0
    results = []

    for q in queries:
        try:
            resp = httpx.post(
                f"{args.base_url}/search",
                json={"query": q["question"], "top_k": args.top_k},
                timeout=60,
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            print(f"Request failed for: {q['question']!r} -> {exc}", file=sys.stderr)
            results.append({"question": q["question"], "error": str(exc)})
            continue

        sources = body.get("sources", [])
        retrieved_docs = [s["document"] for s in sources]

        row = {
            "question": q["question"],
            "expected_document": q.get("expected_document"),
            "retrieved_documents": retrieved_docs,
            "retrieved_chunks": body.get("retrieved_chunks"),
            "latency_ms": body.get("latency_ms"),
            "has_sources": len(sources) > 0,
        }

        if q.get("expected_document"):
            hit = q["expected_document"] in retrieved_docs
            top1 = bool(retrieved_docs) and retrieved_docs[0] == q["expected_document"]
            row["retrieval_hit"] = hit
            row["top1_match"] = top1
            if hit:
                retrieval_hits += 1
            if top1:
                top1_matches += 1
            if sources:
                answers_with_sources += 1
        else:
            declined = len(sources) == 0
            row["correctly_declined"] = declined
            if declined:
                correctly_declined += 1

        results.append(row)

    n_answerable = len(answerable) or 1
    n_total = len(queries) or 1

    retrieval_at_k = retrieval_hits / n_answerable
    source_precision = top1_matches / n_answerable
    answer_reference_coverage = answers_with_sources / n_answerable
    decline_rate = (correctly_declined / len(unanswerable)) if unanswerable else None

    print("\n=== Guidely Evaluation Results ===\n")
    print(f"Total queries:                {len(queries)}")
    print(f"Answerable queries:           {len(answerable)}")
    print(f"Unanswerable (control) queries: {len(unanswerable)}")
    print(f"\nRetrieval@{args.top_k}:                 {retrieval_at_k:.1%}  (target >= 80%)")
    print(f"Source precision (top-1):     {source_precision:.1%}  (target >= 80%)")
    print(f"Answer reference coverage:    {answer_reference_coverage:.1%}  (target >= 90%)")
    if decline_rate is not None:
        print(f"Correctly declined (no hallucination): {decline_rate:.1%}")

    out_path = Path(__file__).resolve().parent.parent / "tests" / "evaluation" / "evaluation_results.json"
    out_path.write_text(
        json.dumps(
            {
                "retrieval_at_k": retrieval_at_k,
                "source_precision": source_precision,
                "answer_reference_coverage": answer_reference_coverage,
                "decline_rate": decline_rate,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nFull results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

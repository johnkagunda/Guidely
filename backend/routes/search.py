"""Search / question-answering route."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models.schemas import SearchRequest, SearchResponse, SourceRef
from services.embeddings import EmbeddingModelError
from services.llm import LLMConfigError, LLMError, LLMTimeoutError
from services.metrics import get_metrics_tracker
from services.rag import EmptyQueryError, NoResultsError, answer_question
from utils.logger import get_logger, log_event

logger = get_logger("guidely.search")
router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    metrics = get_metrics_tracker()

    if request.query.strip() == "":
        metrics.record_error("empty_query")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = answer_question(request.query, top_k=request.top_k)
    except EmptyQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoResultsError as exc:
        # Friendly "no results" response rather than a hard error, per spec.
        return SearchResponse(
            answer=(
                "I couldn't find anything in the indexed documents that "
                "answers this question. Try rephrasing it, or check that "
                "the relevant document has been uploaded."
            ),
            sources=[],
            retrieved_chunks=0,
            latency_ms=0.0,
        )
    except LLMTimeoutError as exc:
        log_event(logger, "llm_timeout", query=request.query, error=str(exc))
        raise HTTPException(status_code=504, detail="The language model timed out. Please try again.") from exc
    except LLMConfigError as exc:
        log_event(logger, "llm_config_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMError as exc:
        metrics.record_error("llm_failure")
        log_event(logger, "llm_failure", query=request.query, error=str(exc))
        raise HTTPException(status_code=502, detail=f"The language model failed to generate an answer: {exc}") from exc
    except EmbeddingModelError as exc:
        metrics.record_error("embedding_failure")
        log_event(logger, "embedding_failure", query=request.query, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {exc}") from exc

    return SearchResponse(
        answer=result.answer,
        sources=[
            SourceRef(document=s.document, section=s.section, snippet=s.snippet, score=s.score)
            for s in result.sources
        ],
        retrieved_chunks=result.retrieved_chunks,
        latency_ms=result.latency_ms,
    )

import logging
import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.query import HealthResponse, QueryRequest, QueryResponse
from app.services.generation import (
    GenerationTimeoutError,
    GenerationUnavailableError,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rag"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    retrieval = getattr(request.app.state, "retrieval_service", None)
    generation = getattr(request.app.state, "generation_service", None)
    settings = request.app.state.settings
    ollama_ready = bool(getattr(request.app.state, "ollama_ready", False))
    rag_ready = retrieval is not None and generation is not None and ollama_ready

    status_value = "ok" if rag_ready else "degraded"
    detail = getattr(request.app.state, "startup_error", None)
    return HealthResponse(
        status=status_value,
        rag_ready=rag_ready,
        collection_name=getattr(retrieval, "collection_name", None),
        indexed_chunks=int(getattr(retrieval, "indexed_chunks", 0)),
        ollama_reachable=ollama_ready,
        model=settings.ollama_model,
        detail=detail,
    )


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    request_id = uuid4().hex[:12]
    started = time.perf_counter()
    retrieval = getattr(request.app.state, "retrieval_service", None)
    generation = getattr(request.app.state, "generation_service", None)
    settings = request.app.state.settings

    if generation is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The generation service is not ready.",
        )

    # Deterministic refusals do not need retrieval or a running LLM.
    risk = generation.classify_risk(payload.question)
    if risk.risk_level == "refuse_redirect":
        response = generation.answer(payload.question, [])
        logger.info(
            "request_id=%s route=/query status=200 risk=%s latency_ms=%.1f",
            request_id,
            risk.reason,
            (time.perf_counter() - started) * 1000,
        )
        return response

    if retrieval is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The persisted RAG index is not ready.",
        )

    if len(payload.question) > settings.max_question_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question exceeds the configured maximum length.",
        )

    retrieval_started = time.perf_counter()
    try:
        chunks = retrieval.retrieve(
            payload.question,
            top_k=settings.retrieval_top_k,
        )
    except Exception as exc:
        logger.exception(
            "request_id=%s retrieval_failed error=%s",
            request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The retrieval service could not answer the request.",
        ) from exc
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    config = retrieval.config
    try:
        response = generation.answer(
            payload.question,
            chunks,
            min_top_score=config.min_top_score,
            max_context_chars=config.max_context_chars,
        )
    except GenerationTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The local language model timed out.",
        ) from exc
    except GenerationUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local language model is unavailable.",
        ) from exc

    logger.info(
        "request_id=%s route=/query status=200 chunks=%d generation_allowed=%s "
        "retrieval_ms=%.1f total_ms=%.1f",
        request_id,
        len(chunks),
        response.confidence.generation_allowed,
        retrieval_ms,
        (time.perf_counter() - started) * 1000,
    )
    return response

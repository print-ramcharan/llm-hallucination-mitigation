"""FastAPI REST routes for Module 3: Hybrid Retrieval & Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, status

from src.retrieval.hybrid import default_hybrid_retriever
from src.retrieval.models import RetrievalRequest, RetrievalResponse

router = APIRouter(prefix="/api/retrieval", tags=["Hybrid Retrieval"])


@router.post(
    "/hybrid",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute hybrid retrieval with pre-flight query processing and RRF fusion",
)
def retrieve_hybrid(payload: RetrievalRequest) -> RetrievalResponse:
    """Run parallel dense semantic (FAISS) and sparse lexical (BM25) search.

    The query undergoes contextual pre-flight understanding, generates both rank lists,
    and fuses them using Reciprocal Rank Fusion (RRF with k=60).
    """
    return default_hybrid_retriever.retrieve_request(payload)


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Get status and telemetry of hybrid retrieval indexes",
)
def get_retrieval_status() -> Dict[str, Any]:
    """Retrieve index sizes and status across FAISS, BM25, and Metadata Store."""
    stats = default_hybrid_retriever.index_manager.get_stats()
    return {
        "status": "ready",
        "module": "Hybrid Retrieval & RRF",
        "rrf_smoothing_constant_default": 60,
        "indexes": stats,
    }

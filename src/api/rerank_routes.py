"""FastAPI REST routes for Module 4: Deep Cross-Encoder Reranking & Pruning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.query_processing.models import ConversationTurn
from src.reranking.models import RankedContext, RerankingRequest
from src.reranking.reranker import default_reranker
from src.retrieval.hybrid import default_hybrid_retriever

router = APIRouter(prefix="/api/rerank", tags=["Cross-Encoder Reranking"])


class HybridAndRerankRequest(BaseModel):
    """Payload for executing end-to-end hybrid retrieval followed by cross-encoder reranking."""

    query: str = Field(..., description="User search query string.")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        default=None,
        description="Optional interactive conversation history for coreference and ellipsis resolution.",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured metadata filters.",
    )
    top_k_fused: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Candidate pool size to retrieve from Module 3 hybrid fusion before reranking.",
    )
    threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Hard cutoff relevance threshold tau for candidate pruning.",
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of top-precision chunks to retain for the prompt context.",
    )


@router.post(
    "",
    response_model=RankedContext,
    status_code=status.HTTP_200_OK,
    summary="Rerank candidate chunks using pairwise deep cross-encoder cross-attention",
)
def rerank_candidates(payload: RerankingRequest) -> RankedContext:
    """Score candidate chunks against the query using cross-attention, calibrate scores,

    and prune low-confidence candidates below the threshold.
    """
    return default_reranker.rerank_request(payload)


@router.post(
    "/pipeline",
    response_model=RankedContext,
    status_code=status.HTTP_200_OK,
    summary="End-to-end pipeline: Hybrid Retrieval (Module 3) -> Deep Reranking & Pruning (Module 4)",
)
def hybrid_retrieve_and_rerank(payload: HybridAndRerankRequest) -> RankedContext:
    """Execute hybrid retrieval (dense + sparse + RRF) and immediately rerank and prune candidates."""
    retrieval_response = default_hybrid_retriever.retrieve(
        query=payload.query,
        conversation_history=payload.conversation_history,
        filters=payload.filters,
        top_k_fused=payload.top_k_fused,
    )
    return default_reranker.rerank_retrieval_response(
        response=retrieval_response,
        threshold=payload.threshold,
        top_n=payload.top_n,
    )


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Get cross-encoder model status and pruning parameters",
)
def get_reranker_status() -> Dict[str, Any]:
    """Retrieve status, model identifier, and default threshold configurations."""
    return {
        "status": "ready",
        "module": "Deep Cross-Encoder Reranking & Pruning",
        "model_name": default_reranker.engine.model_name,
        "default_threshold": default_reranker.default_threshold,
        "default_top_n": default_reranker.default_top_n,
    }

"""FastAPI REST routes for Module 5: Extractive Context Compaction & Optimization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.context.compactor import default_context_compactor
from src.context.models import CompactionRequest, OptimizedContext
from src.query_processing.models import ConversationTurn
from src.reranking.reranker import default_reranker
from src.retrieval.hybrid import default_hybrid_retriever

router = APIRouter(prefix="/api/context", tags=["Context Optimization"])


class FullContextPipelineRequest(BaseModel):
    """Payload executing the full chain: Hybrid Retrieval -> Reranking -> Context Compaction."""

    query: str = Field(..., description="User search query string.")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        default=None,
        description="Optional interactive chat history turns.",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured metadata filters.",
    )
    top_k_fused: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Candidate pool size to retrieve from Module 3 hybrid retrieval.",
    )
    rerank_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Cutoff threshold tau for Module 4 cross-encoder reranking.",
    )
    rerank_top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum candidates retained after Module 4 reranking.",
    )
    max_token_budget: int = Field(
        default=1500,
        ge=100,
        le=8000,
        description="Target maximum token allowance for final prompt context.",
    )
    enable_sentence_pruning: bool = Field(
        default=True,
        description="Whether to perform sentence-level salience extraction.",
    )
    enable_deduplication: bool = Field(
        default=True,
        description="Whether to deduplicate overlapping chunk seams.",
    )
    enable_lost_in_middle_reordering: bool = Field(
        default=True,
        description="Whether to apply U-shaped Lost-in-the-Middle reordering.",
    )


@router.post(
    "/compact",
    response_model=OptimizedContext,
    status_code=status.HTTP_200_OK,
    summary="Compact and optimize reranked chunks into prompt-ready context",
)
def compact_context(payload: CompactionRequest) -> OptimizedContext:
    """Extract salient sentences, remove chunk seam overlap, apply U-shaped reordering,

    and package within token budget with explicit [Doc X, Chunk Y] citation tags.
    """
    return default_context_compactor.compact_request(payload)


@router.post(
    "/pipeline",
    response_model=OptimizedContext,
    status_code=status.HTTP_200_OK,
    summary="End-to-End: Hybrid Retrieval (M3) -> Deep Reranking (M4) -> Context Compaction (M5)",
)
def run_full_context_pipeline(payload: FullContextPipelineRequest) -> OptimizedContext:
    """Execute end-to-end pipeline: retrieve, rerank, and compact prompt context in one call."""
    # 1. Module 3: Hybrid Retrieval
    retrieval_res = default_hybrid_retriever.retrieve(
        query=payload.query,
        conversation_history=payload.conversation_history,
        filters=payload.filters,
        top_k_fused=payload.top_k_fused,
    )

    # 2. Module 4: Cross-Encoder Reranking & Pruning
    ranked_context = default_reranker.rerank_retrieval_response(
        response=retrieval_res,
        threshold=payload.rerank_threshold,
        top_n=payload.rerank_top_n,
    )

    # 3. Module 5: Extractive Context Compaction & Optimization
    return default_context_compactor.compact(
        query=payload.query,
        chunks=ranked_context.chunks,
        max_token_budget=payload.max_token_budget,
        enable_sentence_pruning=payload.enable_sentence_pruning,
        enable_deduplication=payload.enable_deduplication,
        enable_lost_in_middle_reordering=payload.enable_lost_in_middle_reordering,
    )


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Get status and configurations of the context optimization engine",
)
def get_context_status() -> Dict[str, Any]:
    """Retrieve default token allowances, salience cutoffs, and component statuses."""
    return {
        "status": "ready",
        "module": "Extractive Context Compaction & Optimization",
        "default_token_budget": default_context_compactor.budget_manager.default_budget,
        "default_min_salience": default_context_compactor.sentence_pruner.default_min_salience,
        "default_dedup_threshold": default_context_compactor.deduplicator.default_similarity_threshold,
    }

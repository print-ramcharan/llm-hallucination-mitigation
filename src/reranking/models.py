"""Data models and schemas for Module 4: Deep Cross-Encoder Reranking & Pruning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.retrieval.models import RetrievedCandidate


class RerankedChunk(BaseModel):
    """High-precision chunk retained after deep cross-encoder cross-attention and pruning."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    content: str = Field(..., description="Text content of the chunk.")
    document_id: str = Field(..., description="Parent document identifier.")
    chunk_index: int = Field(default=0, description="0-indexed position within parent document.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Standardized chunk metadata tags (e.g. source, page, section, doc_type).",
    )
    initial_rank: Optional[int] = Field(
        default=None,
        description="1-based candidate rank prior to cross-encoder reranking.",
    )
    initial_score: Optional[float] = Field(
        default=None,
        description="Candidate retrieval score prior to reranking (e.g. RRF score).",
    )
    rerank_score: float = Field(
        ...,
        description="Calibrated cross-attention relevance probability s in [0, 1].",
    )
    raw_score: float = Field(
        ...,
        description="Uncalibrated raw logit produced by the cross-encoder model.",
    )
    rerank_position: int = Field(
        ...,
        description="1-based final rank order after cross-encoder scoring and pruning.",
    )


class RerankingRequest(BaseModel):
    """Payload for submitting candidates to the cross-encoder reranker."""

    query: str = Field(..., description="Original user search query.")
    candidates: List[RetrievedCandidate] = Field(
        ...,
        description="Candidate chunks from Module 3 (e.g. Top-20 RRF pool).",
    )
    threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Hard cutoff relevance threshold tau; chunks with s < tau are dropped.",
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of top-scoring candidates to retain for the context window.",
    )
    normalization: str = Field(
        default="sigmoid",
        description="Calibration function to map logits into [0, 1] ('sigmoid' or 'min_max').",
    )


class RankedContext(BaseModel):
    """Final high-precision prompt context payload after cross-encoder reranking and pruning."""

    query: str = Field(..., description="Search query evaluated.")
    chunks: List[RerankedChunk] = Field(
        default_factory=list,
        description="Top-N retained high-precision chunks ordered descending by rerank score.",
    )
    total_input_candidates: int = Field(
        ...,
        description="Total number of candidates submitted to the reranker.",
    )
    retained_count: int = Field(
        ...,
        description="Number of candidates passing the cutoff threshold and retained.",
    )
    pruned_count: int = Field(
        ...,
        description="Number of candidate chunks dropped due to low confidence or top-N cutoff.",
    )
    threshold_applied: float = Field(
        ...,
        description="Hard cutoff threshold tau used during pruning.",
    )
    model_name: str = Field(
        ...,
        description="Pretrained cross-encoder model used for cross-attention.",
    )
    execution_time_ms: float = Field(
        ...,
        description="Execution time of the reranking and pruning pipeline in milliseconds.",
    )

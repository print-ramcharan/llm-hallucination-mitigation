"""Data models and schemas for Module 3: Hybrid Retrieval & Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.query_processing.models import ConversationTurn, ProcessedQuery


class RetrievedCandidate(BaseModel):
    """Unified retrieved candidate chunk with rank and provenance telemetry."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    content: str = Field(..., description="Text content of the retrieved chunk.")
    document_id: str = Field(..., description="Parent document identifier.")
    chunk_index: int = Field(default=0, description="0-indexed position within document.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Standardized chunk metadata tags (e.g. source, page, section, doc_type).",
    )
    dense_rank: Optional[int] = Field(
        default=None,
        description="1-based rank in dense vector retrieval (None if not retrieved by dense search).",
    )
    sparse_rank: Optional[int] = Field(
        default=None,
        description="1-based rank in sparse BM25 retrieval (None if not retrieved by sparse search).",
    )
    dense_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity score from FAISS Inner Product search.",
    )
    sparse_score: Optional[float] = Field(
        default=None,
        description="Lexical relevance score from BM25+ index.",
    )
    rrf_score: float = Field(
        default=0.0,
        description="Fused Reciprocal Rank Fusion (RRF) score: sum(1 / (k + rank)).",
    )


class RetrievalRequest(BaseModel):
    """Input payload requesting hybrid retrieval."""

    query: str = Field(..., description="Raw user query string.")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        default=None,
        description="Preceding interactive conversation turns for contextual rewriting.",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Explicit structured metadata filters (e.g. {'document_type': 'HR_POLICY'}).",
    )
    top_k_dense: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Number of candidate chunks to fetch from dense FAISS search.",
    )
    top_k_sparse: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Number of candidate chunks to fetch from sparse BM25 search.",
    )
    top_k_fused: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of fused candidate chunks to output after RRF merging.",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF smoothing constant preventing high-rank dominance.",
    )


class RetrievalResponse(BaseModel):
    """Output payload containing RRF-fused candidates and pre-flight telemetry."""

    query: str = Field(..., description="Original user query.")
    processed_query: ProcessedQuery = Field(
        ...,
        description="Contextually rewritten, classified, and filtered query representation.",
    )
    candidates: List[RetrievedCandidate] = Field(
        default_factory=list,
        description="Unified candidates ordered descending by reciprocal rank fusion score.",
    )
    total_candidates: int = Field(..., description="Number of candidates returned.")
    dense_count: int = Field(..., description="Number of candidates retrieved by dense search.")
    sparse_count: int = Field(..., description="Number of candidates retrieved by sparse BM25 search.")
    execution_time_ms: float = Field(
        ...,
        description="Total end-to-end retrieval and fusion execution latency in milliseconds.",
    )

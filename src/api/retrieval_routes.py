"""FastAPI REST routes for Module 3: Hybrid Retrieval & Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status

from src.auth import current_user_id, owner_filters
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
    return default_hybrid_retriever.retrieve(
        query=payload.query,
        conversation_history=payload.conversation_history,
        filters=owner_filters(payload.filters),
        top_k_dense=payload.top_k_dense,
        top_k_sparse=payload.top_k_sparse,
        top_k_fused=payload.top_k_fused,
        rrf_k=payload.rrf_k,
    )


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Get status and telemetry of hybrid retrieval indexes",
)
def get_retrieval_status() -> dict[str, Any]:
    """Retrieve retrieval status without disclosing other users' index counts."""
    owner_id = current_user_id()
    owner_chunks = [
        record
        for record in default_hybrid_retriever.index_manager.metadata_store._store.values()
        if record.get("metadata", {}).get("owner_id") == owner_id
    ]
    return {
        "status": "ready",
        "module": "Hybrid Retrieval & RRF",
        "rrf_smoothing_constant_default": 60,
        "indexes": {
            "owned_chunks": len(owner_chunks),
            "embedding_model": default_hybrid_retriever.index_manager.embedding_engine.model_name,
        },
    }

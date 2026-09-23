"""Module 3: Hybrid Retrieval & Reciprocal Rank Fusion (RRF) Package."""

from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever, default_hybrid_retriever
from src.retrieval.models import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedCandidate,
)
from src.retrieval.rrf import RRFEngine
from src.retrieval.sparse import SparseRetriever

__all__ = [
    "DenseRetriever",
    "HybridRetriever",
    "RRFEngine",
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievedCandidate",
    "SparseRetriever",
    "default_hybrid_retriever",
]

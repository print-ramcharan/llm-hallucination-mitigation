"""Hybrid Retrieval Coordinator integrating Query Processing, Dense FAISS, Sparse BM25, and RRF."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.indexing.manager import IndexManager, default_index_manager
from src.query_processing.models import ConversationTurn
from src.query_processing.pipeline import QueryProcessor, default_query_processor
from src.retrieval.dense import DenseRetriever
from src.retrieval.models import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedCandidate,
)
from src.retrieval.rrf import RRFEngine
from src.retrieval.sparse import SparseRetriever


class HybridRetriever:
    """End-to-end coordinator for Module 3 Hybrid Retrieval & Reciprocal Rank Fusion.

    Execution Pipeline:
      1. Pre-Flight Query Understanding (Conversational rewriting, Intent, Filter extraction)
      2. Parallel / Synchronous Dense Retrieval (FAISS Inner Product + Cosine)
      3. Parallel / Synchronous Sparse Retrieval (BM25+ keyword matching)
      4. Reciprocal Rank Fusion (RRF with smoothing constant k)
      5. Enriched candidate ranking with full metadata hydration
    """

    def __init__(
        self,
        index_manager: Optional[IndexManager] = None,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        rrf_engine: Optional[RRFEngine] = None,
    ) -> None:
        self.index_manager = index_manager or default_index_manager
        self.query_processor = query_processor or default_query_processor

        self.dense_retriever = dense_retriever or DenseRetriever(
            vector_index=self.index_manager.vector_index,
            metadata_store=self.index_manager.metadata_store,
            embedding_engine=self.index_manager.embedding_engine,
        )

        self.sparse_retriever = sparse_retriever or SparseRetriever(
            bm25_index=self.index_manager.bm25_index,
            metadata_store=self.index_manager.metadata_store,
        )

        self.rrf_engine = rrf_engine or RRFEngine(
            metadata_store=self.index_manager.metadata_store
        )

    def retrieve(
        self,
        query: str,
        conversation_history: Optional[List[ConversationTurn]] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k_dense: int = 25,
        top_k_sparse: int = 25,
        top_k_fused: int = 20,
        rrf_k: int = 60,
    ) -> RetrievalResponse:
        """Execute complete hybrid retrieval with pre-flight query processing and RRF fusion.

        Args:
            query: Raw user query string.
            conversation_history: Optional list of previous chat turns for pronoun resolution.
            filters: Optional structured metadata filters.
            top_k_dense: Candidate count from dense FAISS search.
            top_k_sparse: Candidate count from sparse BM25 search.
            top_k_fused: Top candidate count to return after fusion.
            rrf_k: Smoothing constant k for RRF (default 60).

        Returns:
            RetrievalResponse with fused candidates and execution telemetry.
        """
        start_time = time.perf_counter()

        # Stage 1: Query Processing Pre-Flight
        processed_query = self.query_processor.process(
            query=query,
            conversation_history=conversation_history,
        )

        # Merge extracted metadata filters with explicit user filters (explicit overrides)
        combined_filters: Dict[str, Any] = {}
        if processed_query.filters:
            combined_filters.update(processed_query.filters)
        if filters:
            combined_filters.update(filters)

        search_query = processed_query.query or query

        # Stage 2: Dense Retrieval (Semantic)
        dense_results = self.dense_retriever.retrieve(
            query=search_query,
            top_k=top_k_dense,
            filters=combined_filters or None,
        )

        # Stage 3: Sparse Retrieval (Lexical BM25)
        sparse_results = self.sparse_retriever.retrieve(
            query=search_query,
            top_k=top_k_sparse,
            filters=combined_filters or None,
        )

        # Stage 4: Reciprocal Rank Fusion (RRF)
        candidates: List[RetrievedCandidate] = self.rrf_engine.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k_fused,
            rrf_k=rrf_k,
            metadata_store=self.index_manager.metadata_store,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResponse(
            query=query,
            processed_query=processed_query,
            candidates=candidates,
            total_candidates=len(candidates),
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            execution_time_ms=round(elapsed_ms, 2),
        )

    def retrieve_request(self, request: RetrievalRequest) -> RetrievalResponse:
        """Helper to invoke retrieve using a structured RetrievalRequest model."""
        return self.retrieve(
            query=request.query,
            conversation_history=request.conversation_history,
            filters=request.filters,
            top_k_dense=request.top_k_dense,
            top_k_sparse=request.top_k_sparse,
            top_k_fused=request.top_k_fused,
            rrf_k=request.rrf_k,
        )


default_hybrid_retriever = HybridRetriever()

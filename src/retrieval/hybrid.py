"""Hybrid Retrieval Coordinator integrating Query Processing, Dense FAISS, Sparse BM25, and RRF."""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
from typing import Any

import numpy as np

from src.cache import LruTtlCache
from src.indexing.manager import IndexManager, default_index_manager
from src.query_processing.models import ConversationTurn, QueryIntent
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
        index_manager: IndexManager | None = None,
        query_processor: QueryProcessor | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        rrf_engine: RRFEngine | None = None,
    ) -> None:
        self.index_manager = index_manager or default_index_manager
        self.query_processor = query_processor or default_query_processor

        self._custom_dense_retriever = dense_retriever
        self._custom_sparse_retriever = sparse_retriever
        self._custom_rrf_engine = rrf_engine

        self.result_cache: LruTtlCache[str, RetrievalResponse] = LruTtlCache(
            max_size=2048, ttl_seconds=90
        )
        # Bounded semantic query cache for near-instant paraphrased query hits (cos sim >= 0.96)
        self._semantic_cache: list[dict[str, Any]] = []
        # Reusable thread pool for parallel dense and sparse retrieval
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    @property
    def dense_retriever(self) -> DenseRetriever:
        """Dynamically resolved DenseRetriever bound to current index_manager."""
        if self._custom_dense_retriever is not None:
            return self._custom_dense_retriever
        return DenseRetriever(
            vector_index=self.index_manager.vector_index,
            metadata_store=self.index_manager.metadata_store,
            embedding_engine=self.index_manager.embedding_engine,
        )

    @dense_retriever.setter
    def dense_retriever(self, val: DenseRetriever | None) -> None:
        self._custom_dense_retriever = val

    @property
    def sparse_retriever(self) -> SparseRetriever:
        """Dynamically resolved SparseRetriever bound to current index_manager."""
        if self._custom_sparse_retriever is not None:
            return self._custom_sparse_retriever
        return SparseRetriever(
            bm25_index=self.index_manager.bm25_index,
            metadata_store=self.index_manager.metadata_store,
        )

    @sparse_retriever.setter
    def sparse_retriever(self, val: SparseRetriever | None) -> None:
        self._custom_sparse_retriever = val

    @property
    def rrf_engine(self) -> RRFEngine:
        """Dynamically resolved RRFEngine bound to current index_manager."""
        if self._custom_rrf_engine is not None:
            return self._custom_rrf_engine
        return RRFEngine(
            metadata_store=self.index_manager.metadata_store,
        )

    @rrf_engine.setter
    def rrf_engine(self, val: RRFEngine | None) -> None:
        self._custom_rrf_engine = val

    def retrieve(
        self,
        query: str,
        conversation_history: list[ConversationTurn] | None = None,
        filters: dict[str, Any] | None = None,
        top_k_dense: int = 50,
        top_k_sparse: int = 50,
        top_k_fused: int = 50,
        rrf_k: int = 60,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        enable_hyde: bool = False,
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
            dense_weight: Optional explicit multiplier weight for dense FAISS rankings.
            sparse_weight: Optional explicit multiplier weight for sparse BM25 rankings.

        Returns:
            RetrievalResponse with fused candidates and execution telemetry.
        """
        start_time = time.perf_counter()

        # Clean empty query handling
        if not query or not query.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RetrievalResponse(
                query=query,
                processed_query=self.query_processor.process(""),
                candidates=[],
                total_candidates=0,
                dense_count=0,
                sparse_count=0,
                execution_time_ms=round(elapsed_ms, 2),
            )

        history_key = [
            turn.model_dump(mode="json") for turn in (conversation_history or [])
        ]
        cache_payload = {
            "query": " ".join(query.split()).casefold(),
            "history": history_key,
            "filters": filters or {},
            "top_k_dense": top_k_dense,
            "top_k_sparse": top_k_sparse,
            "top_k_fused": top_k_fused,
            "rrf_k": rrf_k,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "index_revision": self.index_manager.revision,
        }
        cache_key = json.dumps(cache_payload, sort_keys=True, default=str)
        cached_response = self.result_cache.get(cache_key)
        if cached_response is not None:
            cached_response = cached_response.model_copy(deep=True)
            cached_response.execution_time_ms = round(
                (time.perf_counter() - start_time) * 1000, 2
            )
            return cached_response

        # Fast Semantic Query Cache Check for single-turn standalone queries (similarity >= 0.96)
        query_vec: np.ndarray | None = None
        if not conversation_history and self._semantic_cache:
            try:
                query_vec = self.dense_retriever.embedding_engine.embed_query(query)
                norm_q = np.linalg.norm(query_vec)
                if norm_q > 0:
                    for entry in reversed(self._semantic_cache):
                        if (
                            entry["revision"] == self.index_manager.revision
                            and entry["filters"] == (filters or {})
                            and entry["top_k_fused"] >= top_k_fused
                            and entry.get("dense_weight") == dense_weight
                            and entry.get("sparse_weight") == sparse_weight
                        ):
                            sim = float(np.dot(query_vec, entry["vec"]))
                            if sim >= 0.96:
                                hit = entry["response"].model_copy(deep=True)
                                if len(hit.candidates) > top_k_fused:
                                    hit.candidates = hit.candidates[:top_k_fused]
                                    hit.total_candidates = len(hit.candidates)
                                hit.execution_time_ms = round(
                                    (time.perf_counter() - start_time) * 1000, 2
                                )
                                return hit
            except Exception:
                pass

        # Stage 1: Query Processing Pre-Flight
        processed_query = self.query_processor.process(
            query=query,
            conversation_history=conversation_history,
        )

        # Merge extracted metadata filters with explicit user filters (explicit overrides)
        combined_filters: dict[str, Any] = {}
        if processed_query.filters:
            combined_filters.update(processed_query.filters)
        if filters:
            combined_filters.update(filters)

        dense_query = processed_query.rewritten_query or query
        sparse_query = processed_query.query or query

        # Helper runners with robust exception suppression for single-retriever failure isolation
        def _exec_dense(
            q: str,
            k: int,
            f: dict[str, Any] | None,
            vec: np.ndarray | None = None,
        ) -> list[tuple[str, float]]:
            try:
                return self.dense_retriever.retrieve(
                    query=q, top_k=k, filters=f, query_vector=vec
                )
            except Exception:
                return []

        def _exec_sparse(
            q: str,
            k: int,
            f: dict[str, Any] | None,
        ) -> list[tuple[str, float]]:
            try:
                return self.sparse_retriever.retrieve(query=q, top_k=k, filters=f)
            except Exception:
                return []

        # Stage 2 & 3: Parallel Execution of Dense (Semantic) and Sparse (BM25) Retrieval
        q_vec_for_dense = (
            query_vec
            if (query_vec is not None and dense_query.strip().lower() == query.strip().lower())
            else None
        )
        if enable_hyde and processed_query.hyde_passage:
            try:
                base_vec = (
                    q_vec_for_dense
                    if q_vec_for_dense is not None
                    else self.dense_retriever.embedding_engine.embed_query(dense_query)
                )
                hyde_vec = self.dense_retriever.embedding_engine.embed_query(
                    processed_query.hyde_passage
                )
                blended = 0.65 * base_vec + 0.35 * hyde_vec
                q_vec_for_dense = self.dense_retriever.embedding_engine.normalize_l2(blended)
            except Exception:
                pass

        future_dense = self._executor.submit(
            _exec_dense,
            dense_query,
            top_k_dense,
            combined_filters or None,
            q_vec_for_dense,
        )
        future_sparse = self._executor.submit(
            _exec_sparse,
            sparse_query,
            top_k_sparse,
            combined_filters or None,
        )
        dense_results = future_dense.result()
        sparse_results = future_sparse.result()

        # Adaptive filter relaxation:
        # If auto-extracted heuristic filters caused sparse or dense search to return 0 or
        # severely depleted candidates, fall back to the user's explicit filters
        if (
            (
                not dense_results
                or not sparse_results
                or (len(dense_results) + len(sparse_results) < 2)
            )
            and processed_query.filters
            and combined_filters != (filters or {})
        ):
            fallback_dense_query = processed_query.rewritten_query or query
            fallback_sparse_query = processed_query.query or query
            f_dense = self._executor.submit(
                _exec_dense,
                fallback_dense_query,
                top_k_dense,
                filters,
                None,
            )
            f_sparse = self._executor.submit(
                _exec_sparse,
                fallback_sparse_query,
                top_k_sparse,
                filters,
            )
            rel_dense = f_dense.result()
            rel_sparse = f_sparse.result()

            if not dense_results or len(rel_dense) > len(dense_results):
                dense_results = rel_dense
            if not sparse_results or len(rel_sparse) > len(sparse_results):
                sparse_results = rel_sparse

        # Dynamic intent-aware RRF weights if not explicitly overridden
        if dense_weight is None or sparse_weight is None:
            eff_dense_weight = 1.0
            eff_sparse_weight = 1.0
            if (
                processed_query.intent in (QueryIntent.FACTUAL, QueryIntent.KEYWORD_LOOKUP)
                or re.search(r"\b\d{4}\b|[A-Z0-9_-]{4,}", sparse_query)
            ):
                # Keyword / code / date factoids benefit from BM25 lexical precision
                eff_sparse_weight = 1.2
                eff_dense_weight = 0.8
            elif processed_query.intent in (QueryIntent.SUMMARY, QueryIntent.EXPLORATORY):
                # Broad thematic questions benefit from dense semantic vector matching
                eff_dense_weight = 1.2
                eff_sparse_weight = 0.8
            elif processed_query.intent == QueryIntent.COMPARISON:
                eff_dense_weight = 1.05
                eff_sparse_weight = 1.05

            dense_weight = dense_weight if dense_weight is not None else eff_dense_weight
            sparse_weight = sparse_weight if sparse_weight is not None else eff_sparse_weight

        # Stage 4: Reciprocal Rank Fusion (RRF) with Intent Weighting
        candidates: list[RetrievedCandidate] = self.rrf_engine.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k_fused,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            metadata_store=self.index_manager.metadata_store,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        response = RetrievalResponse(
            query=query,
            processed_query=processed_query,
            candidates=candidates,
            total_candidates=len(candidates),
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            execution_time_ms=round(elapsed_ms, 2),
        )
        self.result_cache.set(cache_key, response.model_copy(deep=True))

        # Store in Semantic Query Cache (max 256 entries)
        if not conversation_history:
            try:
                if query_vec is None:
                    query_vec = self.dense_retriever.embedding_engine.embed_query(query)
                if len(self._semantic_cache) >= 256:
                    self._semantic_cache.pop(0)
                self._semantic_cache.append(
                    {
                        "vec": query_vec,
                        "query": query,
                        "response": response.model_copy(deep=True),
                        "revision": self.index_manager.revision,
                        "filters": filters or {},
                        "top_k_fused": top_k_fused,
                        "dense_weight": dense_weight,
                        "sparse_weight": sparse_weight,
                    }
                )
            except Exception:
                pass

        return response

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
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight,
        )


default_hybrid_retriever = HybridRetriever()

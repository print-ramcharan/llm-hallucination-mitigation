"""Reciprocal Rank Fusion (RRF) Engine for unbiased multi-modal candidate combination."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.indexing.metadata_store import MetadataStore
from src.retrieval.models import RetrievedCandidate


class RRFEngine:
    """Combines ranked lists from dense and sparse retrieval using Reciprocal Rank Fusion.

    Formula:
        RRF_Score(d) = sum_{m in {dense, sparse}} 1 / (k + rank_m(d))

    Where k (default 60) is the Cormack et al. smoothing constant ensuring
    consistent, un-skewed rank aggregation across distinct retrieval mechanisms.
    """

    def __init__(self, metadata_store: Optional[MetadataStore] = None) -> None:
        self.metadata_store = metadata_store

    def fuse(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        top_k: int = 20,
        rrf_k: int = 60,
        metadata_store: Optional[MetadataStore] = None,
    ) -> List[RetrievedCandidate]:
        """Merge dense and sparse search rankings into a single unified candidate pool.

        Args:
            dense_results: Ordered list of (chunk_id, cosine_sim) from dense retrieval.
            sparse_results: Ordered list of (chunk_id, bm25_score) from sparse retrieval.
            top_k: Maximum number of fused candidates to return.
            rrf_k: Smoothing constant k (default 60).
            metadata_store: Optional metadata store to override instance store.

        Returns:
            List of RetrievedCandidate objects sorted descending by RRF score.
        """
        store = metadata_store or self.metadata_store

        # Map: chunk_id -> dict with ranks, raw scores, and accumulated RRF score
        candidates_map: Dict[str, Dict[str, Any]] = {}

        # 1. Process dense rankings (1-indexed)
        for rank_idx, (chunk_id, score) in enumerate(dense_results, start=1):
            if chunk_id not in candidates_map:
                candidates_map[chunk_id] = {
                    "dense_rank": rank_idx,
                    "dense_score": score,
                    "sparse_rank": None,
                    "sparse_score": None,
                    "rrf_score": 1.0 / (rrf_k + rank_idx),
                }
            else:
                candidates_map[chunk_id]["dense_rank"] = rank_idx
                candidates_map[chunk_id]["dense_score"] = score
                candidates_map[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank_idx)

        # 2. Process sparse rankings (1-indexed)
        for rank_idx, (chunk_id, score) in enumerate(sparse_results, start=1):
            if chunk_id not in candidates_map:
                candidates_map[chunk_id] = {
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": rank_idx,
                    "sparse_score": score,
                    "rrf_score": 1.0 / (rrf_k + rank_idx),
                }
            else:
                candidates_map[chunk_id]["sparse_rank"] = rank_idx
                candidates_map[chunk_id]["sparse_score"] = score
                candidates_map[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank_idx)

        # 3. Sort candidates descending by RRF score (secondary sort by raw scores for stability)
        sorted_chunk_items = sorted(
            candidates_map.items(),
            key=lambda item: (
                item[1]["rrf_score"],
                (item[1]["dense_score"] or 0.0) + (item[1]["sparse_score"] or 0.0),
            ),
            reverse=True,
        )

        # 4. Truncate to top_k fused candidates
        top_items = sorted_chunk_items[:top_k]

        # 5. Hydrate candidates with content and metadata from MetadataStore
        fused_candidates: List[RetrievedCandidate] = []
        for chunk_id, stats in top_items:
            record = store.get(chunk_id) if store else None

            content = record.get("content", "") if record else ""
            doc_id = record.get("document_id", "") if record else ""
            chunk_idx = record.get("chunk_index", 0) if record else 0
            metadata = record.get("metadata", {}) if record else {}

            candidate = RetrievedCandidate(
                chunk_id=chunk_id,
                content=content,
                document_id=doc_id,
                chunk_index=chunk_idx,
                metadata=metadata,
                dense_rank=stats["dense_rank"],
                sparse_rank=stats["sparse_rank"],
                dense_score=stats["dense_score"],
                sparse_score=stats["sparse_score"],
                rrf_score=stats["rrf_score"],
            )
            fused_candidates.append(candidate)

        return fused_candidates

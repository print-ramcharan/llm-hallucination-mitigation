"""Reciprocal Rank Fusion (RRF) Engine for unbiased multi-modal candidate combination."""

from __future__ import annotations

from typing import Any

from src.indexing.metadata_store import MetadataStore
from src.retrieval.models import RetrievedCandidate


class RRFEngine:
    """Combines ranked lists from dense and sparse retrieval using Reciprocal Rank Fusion.

    Formula:
        RRF_Score(d) = sum_{m in {dense, sparse}} 1 / (k + rank_m(d))

    Where k (default 60) is the Cormack et al. smoothing constant ensuring
    consistent, un-skewed rank aggregation across distinct retrieval mechanisms.
    """

    def __init__(self, metadata_store: MetadataStore | None = None) -> None:
        self.metadata_store = metadata_store

    def fuse(
        self,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        top_k: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        metadata_store: MetadataStore | None = None,
    ) -> list[RetrievedCandidate]:
        """Merge dense and sparse search rankings into a single unified candidate pool.

        Args:
            dense_results: Ordered list of (chunk_id, cosine_sim) from dense retrieval.
            sparse_results: Ordered list of (chunk_id, bm25_score) from sparse retrieval.
            top_k: Maximum number of fused candidates to return.
            rrf_k: Smoothing constant k (default 60).
            dense_weight: Multiplier weight for dense rankings (default 1.0).
            sparse_weight: Multiplier weight for sparse rankings (default 1.0).
            metadata_store: Optional metadata store to override instance store.

        Returns:
            List of RetrievedCandidate objects sorted descending by RRF score.
        """
        store = metadata_store or self.metadata_store

        # Map: chunk_id -> dict with ranks, raw scores, and accumulated RRF score
        candidates_map: dict[str, dict[str, Any]] = {}

        def _record_candidate(
            chunk_id: str, rank: int, score: float, prefix: str, weight: float = 1.0
        ) -> None:
            if chunk_id not in candidates_map:
                candidates_map[chunk_id] = {
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": None,
                    "sparse_score": None,
                    "rrf_score": 0.0,
                }
            entry = candidates_map[chunk_id]
            entry[f"{prefix}_rank"] = rank
            entry[f"{prefix}_score"] = score
            entry["rrf_score"] += weight * (1.0 / (rrf_k + rank))

        # 1. Process dense rankings (1-indexed)
        for rank_idx, (chunk_id, score) in enumerate(dense_results, start=1):
            _record_candidate(chunk_id, rank_idx, score, "dense", weight=dense_weight)

        # 2. Process sparse rankings (1-indexed)
        for rank_idx, (chunk_id, score) in enumerate(sparse_results, start=1):
            _record_candidate(chunk_id, rank_idx, score, "sparse", weight=sparse_weight)

        # 3. Sort candidates descending by RRF score (secondary sort by normalized scores for stability)
        max_sparse = max(
            (
                entry["sparse_score"]
                for entry in candidates_map.values()
                if entry["sparse_score"] is not None
            ),
            default=1.0,
        )
        max_sparse = max(max_sparse, 1.0)

        sorted_chunk_items = sorted(
            candidates_map.items(),
            key=lambda item: (
                item[1]["rrf_score"],
                (item[1]["dense_score"] or 0.0)
                + ((item[1]["sparse_score"] or 0.0) / max_sparse),
            ),
            reverse=True,
        )

        # 4 & 5. Hydrate candidates with content and metadata from MetadataStore up to top_k
        fused_candidates: list[RetrievedCandidate] = []
        for chunk_id, stats in sorted_chunk_items:
            record = store.get(chunk_id) if store else None

            # When an active store is available, skip ghost IDs not found in the store
            if store is not None and store.total_chunks > 0 and record is None:
                continue

            content = record.get("content", "") if record else ""
            doc_id = record.get("document_id", "") if record else ""
            chunk_idx = record.get("chunk_index", 0) if record else 0
            metadata = record.get("metadata", {}) if record else {}
            parent_id = record.get("parent_id") if record else metadata.get("parent_id")
            parent_content = (
                record.get("parent_content")
                if record
                else metadata.get("parent_content")
            )

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
                parent_id=parent_id,
                parent_content=parent_content,
            )
            fused_candidates.append(candidate)
            if len(fused_candidates) >= top_k:
                break

        return fused_candidates

"""Candidate Pruning Layer and Deep Cross-Encoder Reranker Coordinator."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from src.reranking.engine import CrossEncoderEngine, default_cross_encoder_engine
from src.reranking.models import (
    RankedContext,
    RerankedChunk,
    RerankingRequest,
)
from src.retrieval.models import RetrievalResponse, RetrievedCandidate


class CrossEncoderReranker:
    """Reranks and prunes candidates using joint cross-attention and score calibration.

    Key Sub-Components:
      1. Pairwise Cross-Attention (cross-encoder/ms-marco-MiniLM-L-6-v2)
      2. Score Normalization & Calibration (Sigmoid / Min-Max into [0, 1])
      3. Hard Cutoff Pruning (drops candidates with relevance s < tau, default 0.35)
      4. Top-N Retention (preserves top 3 to 5 highest-confidence chunks for LLM prompt context)
    """

    def __init__(
        self,
        engine: Optional[CrossEncoderEngine] = None,
        default_threshold: float = 0.35,
        default_top_n: int = 5,
    ) -> None:
        self.engine = engine or default_cross_encoder_engine
        self.default_threshold = default_threshold
        self.default_top_n = default_top_n

    @staticmethod
    def sigmoid(logits: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid activation mapping logits to probabilities in [0, 1]."""
        clipped = np.clip(logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def min_max_scale(logits: np.ndarray) -> np.ndarray:
        """Min-max normalization mapping logits to [0, 1]."""
        if logits.size <= 1:
            return np.ones_like(logits, dtype=np.float32)
        min_v = float(np.min(logits))
        max_v = float(np.max(logits))
        if np.isclose(max_v, min_v):
            return np.ones_like(logits, dtype=np.float32)
        return (logits - min_v) / (max_v - min_v)

    def calibrate(self, logits: np.ndarray, method: str = "sigmoid") -> np.ndarray:
        """Calibrate raw model logits into uniform relevance scores in [0, 1]."""
        if logits.size == 0:
            return logits

        if method == "min_max":
            return self.min_max_scale(logits)
        return self.sigmoid(logits)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        normalization: str = "sigmoid",
    ) -> RankedContext:
        """Score, calibrate, and aggressively prune candidates into a high-precision RankedContext.

        Args:
            query: Search query string.
            candidates: Retrieved candidates from Module 3 (e.g. Top-20 RRF pool).
            threshold: Hard cutoff threshold tau (default 0.35).
            top_n: Maximum number of retained chunks (default 5).
            normalization: Score calibration function ('sigmoid' or 'min_max').

        Returns:
            RankedContext containing top-N high-precision chunks and pruning telemetry.
        """
        start_time = time.perf_counter()
        tau = self.default_threshold if threshold is None else threshold
        n = self.default_top_n if top_n is None else top_n

        total_input = len(candidates)
        if total_input == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RankedContext(
                query=query,
                chunks=[],
                total_input_candidates=0,
                retained_count=0,
                pruned_count=0,
                threshold_applied=tau,
                model_name=self.engine.model_name,
                execution_time_ms=round(elapsed_ms, 2),
            )

        # 1. Construct (query, passage) pairs for joint cross-attention
        pairs = [(query, c.content) for c in candidates]

        # 2. Compute raw all-to-all cross-attention logits
        raw_logits = self.engine.predict(pairs)

        # 3. Calibrate scores into probabilities s in [0, 1]
        calibrated_scores = self.calibrate(raw_logits, method=normalization)

        # 4. Associate scores with initial ranks
        scored_candidates = []
        for idx, candidate in enumerate(candidates):
            scored_candidates.append(
                {
                    "candidate": candidate,
                    "initial_rank": idx + 1,
                    "initial_score": candidate.rrf_score,
                    "raw_score": float(raw_logits[idx]),
                    "rerank_score": float(calibrated_scores[idx]),
                }
            )

        # 5. Sort candidates descending by calibrated cross-encoder score
        scored_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)

        # 6. Candidate Pruning Layer: Hard Cutoff (s >= tau) + Top-N Retention
        retained_chunks: List[RerankedChunk] = []
        for rank_pos, item in enumerate(scored_candidates, start=1):
            if item["rerank_score"] < tau:
                # Discard candidates falling below the relevance threshold
                continue

            c: RetrievedCandidate = item["candidate"]
            retained_chunks.append(
                RerankedChunk(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    document_id=c.document_id,
                    chunk_index=c.chunk_index,
                    metadata=c.metadata,
                    initial_rank=item["initial_rank"],
                    initial_score=item["initial_score"],
                    rerank_score=round(item["rerank_score"], 4),
                    raw_score=round(item["raw_score"], 4),
                    rerank_position=len(retained_chunks) + 1,
                )
            )

            # Cap retained candidates to top-N budget
            if len(retained_chunks) >= n:
                break

        pruned_count = total_input - len(retained_chunks)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RankedContext(
            query=query,
            chunks=retained_chunks,
            total_input_candidates=total_input,
            retained_count=len(retained_chunks),
            pruned_count=pruned_count,
            threshold_applied=tau,
            model_name=self.engine.model_name,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def rerank_request(self, request: RerankingRequest) -> RankedContext:
        """Helper to invoke rerank from a structured RerankingRequest."""
        return self.rerank(
            query=request.query,
            candidates=request.candidates,
            threshold=request.threshold,
            top_n=request.top_n,
            normalization=request.normalization,
        )

    def rerank_retrieval_response(
        self,
        response: RetrievalResponse,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        normalization: str = "sigmoid",
    ) -> RankedContext:
        """Convenience chaining Module 3 RetrievalResponse into Module 4 reranking."""
        return self.rerank(
            query=response.query,
            candidates=response.candidates,
            threshold=threshold,
            top_n=top_n,
            normalization=normalization,
        )


default_reranker = CrossEncoderReranker()

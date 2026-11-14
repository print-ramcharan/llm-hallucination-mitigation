"""Candidate Pruning Layer and Deep Cross-Encoder Reranker Coordinator."""

from __future__ import annotations

import re
import time

import numpy as np

from src.reranking.engine import CrossEncoderEngine, default_cross_encoder_engine
from src.reranking.models import (
    RankedContext,
    RerankedChunk,
    RerankingRequest,
)
from src.retrieval.models import RetrievalResponse, RetrievedCandidate


def clean_query_for_reranking(query: str) -> str:
    """Strip document-framing meta phrases that bias cross-attention away from substantive content."""
    cleaned = query.strip()
    patterns = [
        r"\b(?:what\s+does\s+(?:the|this)\s+(?:document|doc|file|text|passage|context)\s+say\s+about)\b",
        r"\b(?:what\s+is\s+(?:the|this)\s+(?:document|doc|file|text|passage|context)\s+about)\b",
        r"\b(?:summarize|summary\s+of|overview\s+of|details\s+of|key\s+points\s+of)\s+(?:the|this)\s+(?:document|doc|file|text|passage|context)\b",
        r"\b(?:mentioned|stated|listed|given|described|provided|found)\s+in\s+(?:the|this)\s+(?:document|doc|file|text|passage|context)\b",
        r"\baccording\s+to\s+(?:the|this)\s+(?:document|doc|file|text|passage|context)\b",
        r"\b(?:in|of|from|about)\s+(?:the|this)\s+(?:document|doc|file|text|passage|context)\b",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) >= 3 else query.strip()


class CrossEncoderReranker:
    """Reranks and prunes candidates using joint cross-attention and score calibration.

    Key Sub-Components:
      1. Cascaded Reranking (prunes candidate pool before deep cross-encoder evaluation)
      2. Pairwise Cross-Attention (cross-encoder/ms-marco-MiniLM-L-6-v2)
      3. Score Normalization & Calibration (Sigmoid / Min-Max into [0, 1])
      4. Hard Cutoff & Dynamic Score-Margin Pruning (drops candidates with s < tau or far below top score)
      5. Top-N Retention (preserves top 3 to 5 highest-confidence chunks for LLM prompt context)
    """

    def __init__(
        self,
        engine: CrossEncoderEngine | None = None,
        default_threshold: float = 0.15,
        default_top_n: int = 20,
        max_cross_encoder_candidates: int = 20,
        default_score_margin: float | None = None,
    ) -> None:
        self.engine = engine or default_cross_encoder_engine
        self.default_threshold = default_threshold
        self.default_top_n = default_top_n
        self.max_cross_encoder_candidates = max_cross_encoder_candidates
        self.default_score_margin = default_score_margin

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
        candidates: list[RetrievedCandidate],
        threshold: float | None = None,
        top_n: int | None = None,
        normalization: str = "sigmoid",
        score_margin: float | None = None,
        max_cross_encoder_candidates: int | None = None,
        adaptive_pruning: bool = False,
    ) -> RankedContext:
        """Score, calibrate, and aggressively prune candidates into a high-precision RankedContext.

        Args:
            query: Search query string.
            candidates: Retrieved candidates from Module 3 (e.g. Top-20 RRF pool).
            threshold: Hard cutoff threshold tau (default 0.15).
            top_n: Maximum number of retained chunks (default 20).
            normalization: Score calibration function ('sigmoid' or 'min_max').
            score_margin: Relative cutoff factor (keep candidates with score >= top_score * margin).
            max_cross_encoder_candidates: Max top RRF candidates to evaluate through cross-encoder.
            adaptive_pruning: Dynamically restrict evaluated pool to top 8-10 candidates if score separation is high.

        Returns:
            RankedContext containing top-N high-precision chunks and pruning telemetry.
        """
        start_time = time.perf_counter()
        tau = self.default_threshold if threshold is None else threshold
        n = self.default_top_n if top_n is None else top_n
        margin = (
            self.default_score_margin if score_margin is None else score_margin
        )

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

        # 1. Cascaded Reranking: evaluate only top candidate slice to cut cross-attention latency
        max_eval = (
            max_cross_encoder_candidates
            if max_cross_encoder_candidates is not None
            else self.max_cross_encoder_candidates
        )
        if adaptive_pruning and len(candidates) > 8:
            top_rrf = candidates[0].rrf_score
            eighth_rrf = candidates[min(7, len(candidates) - 1)].rrf_score
            if top_rrf > 0 and (eighth_rrf / top_rrf) < 0.70:
                max_eval = min(max_eval, max(8, min(n, 12)))

        eval_candidates = (
            candidates[:max_eval]
            if max_eval is not None and len(candidates) > max_eval
            else candidates
        )

        # 2. Clean query of framing noise for cross-attention scoring
        attention_query = clean_query_for_reranking(query)
        pairs = [(attention_query, c.content) for c in eval_candidates]

        # 3. Compute raw cross-attention logits
        raw_logits = self.engine.predict(pairs)

        # 4. Calibrate scores into probabilities s in [0, 1]
        calibrated_scores = self.calibrate(raw_logits, method=normalization)

        # 5. Associate scores with initial ranks
        scored_candidates = []
        for idx, candidate in enumerate(eval_candidates):
            scored_candidates.append(
                {
                    "candidate": candidate,
                    "initial_rank": idx + 1,
                    "initial_score": candidate.rrf_score,
                    "raw_score": float(raw_logits[idx]),
                    "rerank_score": float(calibrated_scores[idx]),
                }
            )

        # 6. Sort candidates descending by calibrated cross-encoder score
        scored_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)

        # 7. Candidate Pruning Layer: Retain up to top-N chunks with s >= tau
        qualifying = [
            item for item in scored_candidates if item["rerank_score"] >= tau
        ]

        # Dynamic Score-Margin Pruning: Drop candidates that drop sharply below the top leader.
        # cutoff = top_score - margin ensures candidates more than `margin` points below the
        # top score are pruned (e.g. margin=0.40 → drop anything >0.40 below the top).
        if qualifying and margin is not None and margin > 0.0:
            top_score = qualifying[0]["rerank_score"]
            cutoff = top_score - margin
            qualifying = [
                item for item in qualifying if item["rerank_score"] >= cutoff
            ]

        # Select top-N from qualifying (or fallback to top candidates if default threshold and nothing qualified)
        if qualifying:
            selected = qualifying[:n]
        elif threshold is not None and threshold != self.default_threshold:
            # Explicit non-default threshold was provided and nothing qualified
            selected = []
        else:
            fallback_k = min(n, 3) if len(scored_candidates) >= 3 else len(scored_candidates)
            selected = scored_candidates[:fallback_k]

        retained_chunks: list[RerankedChunk] = []
        for rank_idx, item in enumerate(selected):
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
                    rerank_position=rank_idx + 1,
                )
            )

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
            adaptive_pruning=getattr(request, "adaptive_pruning", False),
        )

    def rerank_retrieval_response(
        self,
        response: RetrievalResponse,
        threshold: float | None = None,
        top_n: int | None = None,
        normalization: str = "sigmoid",
        adaptive_pruning: bool = False,
    ) -> RankedContext:
        """Convenience chaining Module 3 RetrievalResponse into Module 4 reranking."""
        return self.rerank(
            query=response.query,
            candidates=response.candidates,
            threshold=threshold,
            top_n=top_n,
            normalization=normalization,
            adaptive_pruning=adaptive_pruning,
        )


default_reranker = CrossEncoderReranker()

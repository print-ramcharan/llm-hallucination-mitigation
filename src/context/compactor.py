"""Extractive Context Compactor and Prompt Optimizer Coordinator."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

from src.chunking.tokenizer import count_tokens
from src.context.budget_manager import TokenBudgetManager
from src.context.deduplicator import ContextDeduplicator
from src.context.models import (
    CompactedEvidence,
    CompactionRequest,
    OptimizedContext,
)
from src.context.reorderer import LostInTheMiddleReorderer
from src.context.sentence_pruner import SentencePruner
from src.reranking.models import RankedContext, RerankedChunk


class ContextCompactor:
    """Coordinates extractive pruning, seam deduplication, U-shaped reordering, and budget enforcement.

    Pipeline Stages:
      1. Sentence-level Salience Extraction (extracts key answers, drops filler)
      2. Chunk Seam Deduplication (filters 50-token overlap seam duplicates via Jaccard)
      3. Lost-in-the-Middle Reordering (U-shaped distribution: Top-1 at start, Top-2 at end)
      4. Token Budget Packaging & Citation Formatting ([Doc X, Chunk Y] headers)
    """

    def __init__(
        self,
        sentence_pruner: Optional[SentencePruner] = None,
        deduplicator: Optional[ContextDeduplicator] = None,
        reorderer: Optional[LostInTheMiddleReorderer] = None,
        budget_manager: Optional[TokenBudgetManager] = None,
    ) -> None:
        self.sentence_pruner = sentence_pruner or SentencePruner()
        self.deduplicator = deduplicator or ContextDeduplicator()
        self.reorderer = reorderer or LostInTheMiddleReorderer()
        self.budget_manager = budget_manager or TokenBudgetManager()

    def compact(
        self,
        query: str,
        chunks: List[RerankedChunk],
        max_token_budget: int = 1500,
        enable_sentence_pruning: bool = True,
        enable_deduplication: bool = True,
        enable_lost_in_middle_reordering: bool = True,
        dedup_similarity_threshold: float = 0.75,
        min_sentence_salience: float = 0.15,
    ) -> OptimizedContext:
        """Execute complete extractive compaction, deduplication, and U-shaped prompt packaging.

        Args:
            query: User search query.
            chunks: High-precision reranked chunks from Module 4.
            max_token_budget: Hard token limit for the prompt context.
            enable_sentence_pruning: Whether to prune low-salience filler sentences.
            enable_deduplication: Whether to remove duplicate sentences across chunk seams.
            enable_lost_in_middle_reordering: Whether to apply U-shaped reordering.
            dedup_similarity_threshold: Jaccard overlap cutoff for duplicate sentences.
            min_sentence_salience: Minimum salience required to retain an extracted sentence.

        Returns:
            OptimizedContext ready for LLM prompt injection with explicit citation tags.
        """
        start_time = time.perf_counter()

        if not chunks:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return OptimizedContext(
                query=query,
                formatted_prompt_context="",
                evidence_items=[],
                total_tokens=0,
                original_tokens=0,
                saved_tokens=0,
                token_budget=max_token_budget,
                dedup_pruned_count=0,
                reordered=False,
                execution_time_ms=round(elapsed_ms, 2),
            )

        seen_sentence_terms: List[Set[str]] = []
        dedup_pruned_total = 0

        compacted_candidates: List[Dict[str, Any]] = []

        # Stage 1 & 2: Sentence-Level Pruning and Overlap Deduplication per chunk
        for chunk in chunks:
            orig_text = chunk.content

            # 1. Sentence Pruning
            if enable_sentence_pruning:
                _, sentences = self.sentence_pruner.prune_chunk(
                    query=query,
                    text=orig_text,
                    min_salience=min_sentence_salience,
                )
            else:
                sentences = self.sentence_pruner.split_sentences(orig_text)

            # 2. Seam Deduplication
            if enable_deduplication:
                unique_sentences, pruned_cnt = self.deduplicator.deduplicate_sentences(
                    sentences=sentences,
                    seen_sentence_terms=seen_sentence_terms,
                    similarity_threshold=dedup_similarity_threshold,
                )
                dedup_pruned_total += pruned_cnt
            else:
                unique_sentences = sentences

            # Reconstruct compacted text
            extracted_text = " ".join(unique_sentences).strip()
            if not extracted_text:
                continue

            orig_chunk_tokens = count_tokens(orig_text)
            comp_chunk_tokens = count_tokens(extracted_text)

            compacted_candidates.append(
                {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "original_text": orig_text,
                    "extracted_text": extracted_text,
                    "token_count": comp_chunk_tokens,
                    "original_token_count": orig_chunk_tokens,
                    "metadata": chunk.metadata,
                }
            )

        # Stage 3: Lost-in-the-Middle Context Reordering
        if enable_lost_in_middle_reordering and len(compacted_candidates) > 2:
            reordered_candidates = self.reorderer.reorder(compacted_candidates)
            was_reordered = True
        else:
            reordered_candidates = compacted_candidates
            was_reordered = False

        # Stage 4: Token Budget Enforcement & Citation Packaging
        formatted_context, accepted_items, total_tokens = self.budget_manager.fit_within_budget(
            evidence_candidates=reordered_candidates,
            max_budget=max_token_budget,
        )

        # Build CompactedEvidence objects
        evidence_items: List[CompactedEvidence] = []
        for pos, item in enumerate(accepted_items, start=1):
            orig_t = item["original_token_count"]
            curr_t = item["token_count"]
            ratio = round(curr_t / orig_t, 3) if orig_t > 0 else 1.0

            evidence_items.append(
                CompactedEvidence(
                    citation_tag=item.get("citation_tag", f"[Doc 1, Chunk {pos}]"),
                    document_id=item["document_id"],
                    chunk_id=item["chunk_id"],
                    original_text=item["original_text"],
                    extracted_text=item["extracted_text"],
                    token_count=curr_t,
                    original_token_count=orig_t,
                    compression_ratio=ratio,
                    reorder_position=pos,
                    metadata=item.get("metadata", {}),
                )
            )

        # Compute baseline prompt context token count without compaction for true savings metric
        baseline_blocks = [
            self.budget_manager.format_evidence_block(
                citation_tag=item.get("citation_tag", f"[Doc 1, Chunk {pos}]"),
                text=item["original_text"],
                metadata=item.get("metadata", {}),
            )
            for pos, item in enumerate(accepted_items, start=1)
        ]
        baseline_context = "\n\n".join(baseline_blocks)
        original_context_tokens = count_tokens(baseline_context) if baseline_blocks else 0
        saved_tokens = max(0, original_context_tokens - total_tokens)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return OptimizedContext(
            query=query,
            formatted_prompt_context=formatted_context,
            evidence_items=evidence_items,
            total_tokens=total_tokens,
            original_tokens=original_context_tokens,
            saved_tokens=saved_tokens,
            token_budget=max_token_budget,
            dedup_pruned_count=dedup_pruned_total,
            reordered=was_reordered,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def compact_request(self, request: CompactionRequest) -> OptimizedContext:
        """Helper to invoke compaction from a structured CompactionRequest."""
        return self.compact(
            query=request.query,
            chunks=request.chunks,
            max_token_budget=request.max_token_budget,
            enable_sentence_pruning=request.enable_sentence_pruning,
            enable_deduplication=request.enable_deduplication,
            enable_lost_in_middle_reordering=request.enable_lost_in_middle_reordering,
            dedup_similarity_threshold=request.dedup_similarity_threshold,
            min_sentence_salience=request.min_sentence_salience,
        )

    def compact_ranked_context(
        self,
        ranked_context: RankedContext,
        max_token_budget: int = 1500,
        enable_sentence_pruning: bool = True,
        enable_deduplication: bool = True,
        enable_lost_in_middle_reordering: bool = True,
    ) -> OptimizedContext:
        """Convenience method chaining Module 4 RankedContext directly into Module 5 compaction."""
        return self.compact(
            query=ranked_context.query,
            chunks=ranked_context.chunks,
            max_token_budget=max_token_budget,
            enable_sentence_pruning=enable_sentence_pruning,
            enable_deduplication=enable_deduplication,
            enable_lost_in_middle_reordering=enable_lost_in_middle_reordering,
        )


default_context_compactor = ContextCompactor()

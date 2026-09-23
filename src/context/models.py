"""Data models and schemas for Module 5: Extractive Context Compaction & Optimization."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.reranking.models import RerankedChunk


class CompactedEvidence(BaseModel):
    """An individual compacted evidence segment with citation provenance and token metrics."""

    citation_tag: str = Field(
        ...,
        description="Explicit evidence citation identifier (e.g. '[Doc 1, Chunk 2]').",
    )
    document_id: str = Field(..., description="Parent document identifier.")
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    original_text: str = Field(..., description="Original unpruned chunk text.")
    extracted_text: str = Field(
        ...,
        description="Compacted, salient sentence text after pruning and seam deduplication.",
    )
    token_count: int = Field(
        ...,
        description="Token count of the extracted evidence segment.",
    )
    original_token_count: int = Field(
        ...,
        description="Token count of the chunk prior to extractive compaction.",
    )
    compression_ratio: float = Field(
        ...,
        description="Ratio of compacted tokens over original tokens (e.g. 0.65 = 35% saved).",
    )
    reorder_position: int = Field(
        ...,
        description="1-based position in the final U-shaped context sequence.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved chunk metadata tags (e.g. source, page, section).",
    )


class CompactionRequest(BaseModel):
    """Payload for submitting reranked candidates to the extractive context optimizer."""

    query: str = Field(..., description="Original user search query.")
    chunks: List[RerankedChunk] = Field(
        ...,
        description="Top-ranked high-confidence chunks from Module 4 cross-encoder reranking.",
    )
    max_token_budget: int = Field(
        default=1500,
        ge=100,
        le=8000,
        description="Target maximum token allowance for the compacted prompt context.",
    )
    enable_sentence_pruning: bool = Field(
        default=True,
        description="Whether to perform sentence-level salience extraction.",
    )
    enable_deduplication: bool = Field(
        default=True,
        description="Whether to filter overlapping boundary sentences across chunk seams.",
    )
    enable_lost_in_middle_reordering: bool = Field(
        default=True,
        description="Whether to reorder candidates along a U-shaped attention distribution.",
    )
    dedup_similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Jaccard word-overlap threshold above which sentences are pruned as duplicates.",
    )
    min_sentence_salience: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum relevance salience required to retain an extracted sentence.",
    )


class OptimizedContext(BaseModel):
    """Structured, compacted prompt context payload ready for LLM inference injection."""

    query: str = Field(..., description="Search query evaluated.")
    formatted_prompt_context: str = Field(
        ...,
        description="Fully formatted prompt text block with explicit [Doc X, Chunk Y] citation tags.",
    )
    evidence_items: List[CompactedEvidence] = Field(
        default_factory=list,
        description="Ordered list of compacted evidence segments.",
    )
    total_tokens: int = Field(
        ...,
        description="Total token consumption of the formatted prompt context.",
    )
    original_tokens: int = Field(
        ...,
        description="Sum of original chunk token counts prior to compaction.",
    )
    saved_tokens: int = Field(
        ...,
        description="Net tokens saved by extractive pruning and seam deduplication.",
    )
    token_budget: int = Field(
        ...,
        description="Maximum token allowance enforced.",
    )
    dedup_pruned_count: int = Field(
        ...,
        description="Number of duplicate sentences filtered out across chunk seams.",
    )
    reordered: bool = Field(
        ...,
        description="Whether Lost-in-the-Middle U-shaped reordering was applied.",
    )
    execution_time_ms: float = Field(
        ...,
        description="Total context compaction and packaging execution latency in milliseconds.",
    )

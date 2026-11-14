"""Deduplication Engine for eliminating chunk seam overlaps and redundant sentences."""

from __future__ import annotations

import re

_WORD_REGEX = re.compile(r"\b[a-zA-Z0-9_\-]+\b")


class ContextDeduplicator:
    """Detects and prunes overlapping boundary sentences across adjacent chunks.

    Specifically targets duplicate sentences introduced by the 50-token sliding
    window chunk seam during document ingestion.
    """

    def __init__(self, default_similarity_threshold: float = 0.75) -> None:
        self.default_similarity_threshold = default_similarity_threshold

    @staticmethod
    def _to_term_set(sentence: str) -> set[str]:
        """Convert a sentence into a set of normalized lowercase words."""
        return set(_WORD_REGEX.findall(sentence.lower()))

    @classmethod
    def jaccard_similarity(cls, set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard token overlap similarity between two word sets."""
        if not set_a or not set_b:
            return 0.0
        intersection_size = len(set_a.intersection(set_b))
        union_size = len(set_a.union(set_b))
        if union_size == 0:
            return 0.0
        return intersection_size / union_size

    def deduplicate_sentences(
        self,
        sentences: list[str],
        seen_sentence_terms: list[set[str]],
        similarity_threshold: float | None = None,
    ) -> tuple[list[str], int]:
        """Filter out sentences that duplicate previously seen sentences in the context pool.

        Args:
            sentences: Ordered sentences from the current candidate chunk.
            seen_sentence_terms: History of term-sets for sentences already admitted into context.
            similarity_threshold: Jaccard cutoff threshold (default 0.75).

        Returns:
            Tuple of (unique_sentences, pruned_duplicates_count).
        """
        threshold = (
            self.default_similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        )
        unique_sentences: list[str] = []
        pruned_count = 0

        for sentence in sentences:
            s_clean = sentence.strip()
            if not s_clean:
                continue

            current_terms = self._to_term_set(s_clean)
            if not current_terms:
                unique_sentences.append(s_clean)
                continue

            # Check for Jaccard overlap against all previously admitted sentences
            is_duplicate = False
            for prev_terms in seen_sentence_terms:
                sim = self.jaccard_similarity(current_terms, prev_terms)
                if sim >= threshold:
                    is_duplicate = True
                    pruned_count += 1
                    break

            if not is_duplicate:
                unique_sentences.append(s_clean)
                seen_sentence_terms.append(current_terms)

        return unique_sentences, pruned_count

    def strip_seam_overlap(
        self,
        prev_chunk_text: str,
        curr_chunk_text: str,
        similarity_threshold: float = 0.70,
    ) -> str:
        """Strip redundant trailing-to-leading seam overlap between two consecutive chunks.

        When consecutive chunks (chunk N and chunk N+1) are both admitted into the prompt,
        this eliminates duplicate overlapping sentences at the beginning of chunk N+1.
        """
        if not prev_chunk_text or not curr_chunk_text:
            return curr_chunk_text

        prev_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", prev_chunk_text) if s.strip()
        ]
        curr_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", curr_chunk_text) if s.strip()
        ]
        if not prev_sentences or not curr_sentences:
            return curr_chunk_text

        # Compare trailing sentences of prev against leading sentences of curr
        prev_tail_terms = [self._to_term_set(s) for s in prev_sentences[-3:]]

        prune_idx = 0
        for s in curr_sentences[:3]:
            curr_terms = self._to_term_set(s)
            if not curr_terms:
                continue
            if any(
                self.jaccard_similarity(curr_terms, p_terms) >= similarity_threshold
                for p_terms in prev_tail_terms
            ):
                prune_idx += 1
            else:
                break

        if prune_idx > 0 and prune_idx < len(curr_sentences):
            return " ".join(curr_sentences[prune_idx:]).strip()
        return curr_chunk_text

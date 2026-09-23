"""Deduplication Engine for eliminating chunk seam overlaps and redundant sentences."""

from __future__ import annotations

import re
from typing import List, Set, Tuple

_WORD_REGEX = re.compile(r"\b[a-zA-Z0-9_\-]+\b")


class ContextDeduplicator:
    """Detects and prunes overlapping boundary sentences across adjacent chunks.

    Specifically targets duplicate sentences introduced by the 50-token sliding
    window chunk seam during document ingestion.
    """

    def __init__(self, default_similarity_threshold: float = 0.75) -> None:
        self.default_similarity_threshold = default_similarity_threshold

    @staticmethod
    def _to_term_set(sentence: str) -> Set[str]:
        """Convert a sentence into a set of normalized lowercase words."""
        return set(_WORD_REGEX.findall(sentence.lower()))

    @classmethod
    def jaccard_similarity(cls, set_a: Set[str], set_b: Set[str]) -> float:
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
        sentences: List[str],
        seen_sentence_terms: List[Set[str]],
        similarity_threshold: float | None = None,
    ) -> Tuple[List[str], int]:
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
        unique_sentences: List[str] = []
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

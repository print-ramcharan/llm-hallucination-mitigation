"""Lost-in-the-Middle Context Reorderer for U-shaped attention distribution."""

from __future__ import annotations

from typing import List, TypeVar

T = TypeVar("T")


class LostInTheMiddleReorderer:
    """Re-arranges evidence chunks in a U-shaped attention curve.

    Mitigates LLM attention degradation (the 'Lost-in-the-Middle' phenomenon documented
    by Liu et al., where models recall facts best at prompt extremities and degrade centrally).

    Ordering Curve:
      - Position 1 (Beginning): Highest-ranked primary evidence chunk.
      - Position Last (End, immediately before query): Second-highest ranked chunk.
      - Middle Positions: Supporting background evidence.
    """

    @staticmethod
    def reorder(items: List[T]) -> List[T]:
        """Reorder ranked items into a U-shaped attention distribution.

        Args:
            items: Elements sorted descending by relevance score (Rank 1, 2, 3, 4, 5...).

        Returns:
            Re-arranged list with top-1 at start, top-2 at end, and remaining items distributed inward.
        """
        n = len(items)
        if n <= 2:
            return list(items)

        reordered: List[T] = [None] * n  # type: ignore
        left = 0
        right = n - 1

        for i, item in enumerate(items):
            if i % 2 == 0:
                reordered[left] = item
                left += 1
            else:
                reordered[right] = item
                right -= 1

        return reordered

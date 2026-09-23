"""Cross-Encoder Model Engine wrapping sentence-transformers for pairwise cross-attention."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


class CrossEncoderEngine:
    """Wraps pretrained CrossEncoder models for pairwise deep cross-attention scoring.

    Performs joint all-to-all attention between (query, passage) token sequences
    to generate uncalibrated relevance logits.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Pretrained model identifier."""
        return self._model_name

    def _get_model(self):
        """Lazy loader for CrossEncoder model to avoid blocking on module import."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def predict(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Compute all-to-all cross-attention scores for a batch of (query, passage) pairs.

        Args:
            pairs: List of (query_text, passage_text) tuples.

        Returns:
            1D numpy array of raw logits.
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        model = self._get_model()
        scores = model.predict(pairs)
        return np.asarray(scores, dtype=np.float32)


default_cross_encoder_engine = CrossEncoderEngine()

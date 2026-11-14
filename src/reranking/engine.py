"""Cross-Encoder Model Engine wrapping sentence-transformers for pairwise cross-attention with inference acceleration."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import numpy as np


def _get_inference_context():
    """Return torch.inference_mode() if torch is available, else nullcontext."""
    try:
        import torch

        return torch.inference_mode()
    except Exception:
        return nullcontext()


class CrossEncoderEngine:
    """Wraps pretrained CrossEncoder models for pairwise deep cross-attention scoring.

    Performs joint all-to-all attention between (query, passage) token sequences
    to generate uncalibrated relevance logits with accelerated inference.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        quantize: bool | None = None,
    ) -> None:
        self._model_name = model_name
        self.quantize = (
            quantize
            if quantize is not None
            else os.environ.get("QUANTIZE_RERANKER", "0").lower() in ("1", "true")
        )
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        """Pretrained model identifier."""
        return self._model_name

    def _get_model(self):
        """Lazy loader for CrossEncoder model to avoid blocking on module import."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            try:
                model = CrossEncoder(self._model_name, local_files_only=True)
            except Exception:
                model = CrossEncoder(self._model_name)

            if self.quantize:
                try:
                    import torch

                    model.model = torch.ao.quantization.quantize_dynamic(
                        model.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                except Exception:
                    pass

            self._model = model
        return self._model

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Compute all-to-all cross-attention scores for a batch of (query, passage) pairs.

        Args:
            pairs: List of (query_text, passage_text) tuples.

        Returns:
            1D numpy array of raw logits.
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        model = self._get_model()
        with _get_inference_context():
            scores = model.predict(pairs)
        return np.asarray(scores, dtype=np.float32)


default_cross_encoder_engine = CrossEncoderEngine()

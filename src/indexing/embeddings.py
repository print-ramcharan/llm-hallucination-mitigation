"""Embedding generation engine using sentence-transformers with L2 normalization and accelerated inference."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import numpy as np

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

from src.cache import LruTtlCache

# pyrefly: ignore [missing-import]
from src.ingestion.models import DocumentChunk

# Default high-performance sentence-transformers dense retrieval model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMENSION = 384


def _get_inference_context():
    """Return torch.inference_mode() if torch is available, else nullcontext."""
    try:
        import torch

        return torch.inference_mode()
    except Exception:
        return nullcontext()


class EmbeddingEngine:
    """Computes dense embeddings for document chunks specifically using sentence-transformers.

    Applies L2 normalization to ensure that inner product matches cosine similarity.
    Supports accelerated execution via inference_mode and optional dynamic INT8 quantization.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        quantize: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.quantize = (
            quantize
            if quantize is not None
            else os.environ.get("QUANTIZE_EMBEDDINGS", "0").lower() in ("1", "true")
        )
        self._model: Any | None = None
        self.query_cache: LruTtlCache[str, np.ndarray] = LruTtlCache(
            max_size=4096, ttl_seconds=600
        )

    @property
    def model(self) -> Any:
        """Lazy-load the sentence-transformers model with optional quantization."""
        if self._model is None:
            try:
                model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                model = SentenceTransformer(self.model_name)

            if self.quantize:
                try:
                    import torch

                    model = torch.ao.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                except Exception:
                    pass

            self._model = model
        return self._model

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality (384 for sentence-transformers/all-MiniLM-L6-v2)."""
        if self._model is None and self.model_name == DEFAULT_EMBEDDING_MODEL:
            return DEFAULT_DIMENSION
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        return dim if dim is not None else DEFAULT_DIMENSION

    @staticmethod
    def normalize_l2(vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length using L2 norm: v / ||v||_2.

        With unit-length vectors, dot product is equivalent to cosine similarity:
        cos_sim(u, v) = u . v
        """
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / max(norm, 1e-12)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of strings and return L2-normalized float32 vectors."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Accelerated inference with torch.inference_mode()
        with _get_inference_context():
            raw_embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        vectors = np.asarray(raw_embeddings, dtype=np.float32)
        # Guarantee strict L2 normalization
        return self.normalize_l2(vectors)

    def embed_chunks(
        self, chunks: list[DocumentChunk], batch_size: int = 64
    ) -> np.ndarray:
        """Extract text from chunks and generate L2-normalized embeddings."""
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)

        texts = [chunk.content for chunk in chunks]
        return self.embed_texts(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query into an L2-normalized 1D vector."""
        if not query.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        normalized_query = " ".join(query.split())
        cache_key = f"{self.model_name}:{normalized_query.casefold()}"
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        with _get_inference_context():
            vec = self.model.encode(
                [normalized_query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
        vector = self.normalize_l2(np.asarray(vec, dtype=np.float32))
        self.query_cache.set(cache_key, vector.copy())
        return vector


default_embedding_engine = EmbeddingEngine()

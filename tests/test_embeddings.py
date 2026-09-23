"""Unit tests for EmbeddingEngine and L2 normalization."""

import numpy as np
import pytest

from src.indexing.embeddings import EmbeddingEngine
from src.ingestion.models import DocumentChunk


class TestEmbeddingEngine:
    @pytest.fixture(scope="module")
    def engine(self):
        return EmbeddingEngine()

    def test_dimension_is_384(self, engine):
        assert engine.dimension == 384

    def test_l2_normalization_property(self, engine):
        """Verify all generated vectors have unit norm (||v||_2 == 1.0)."""
        texts = [
            "Context degradation affects long document attention.",
            "FAISS computes fast inner product search.",
            "BM25 handles sparse lexical term matching.",
        ]
        embeddings = engine.embed_texts(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

        # Compute L2 norm for each row
        norms = np.linalg.norm(embeddings, axis=1)
        for i, norm in enumerate(norms):
            assert np.isclose(norm, 1.0, atol=1e-5), f"Vector {i} norm is {norm}, expected 1.0"

    def test_cosine_similarity_via_dot_product(self, engine):
        """Verify that dot product between L2-normalized vectors equals cosine similarity."""
        text1 = "Leave policy allows 20 days of paid vacation per year."
        text2 = "Employees receive 20 days of paid annual time off."
        unrelated = "Quantum electrodynamics describes photon interactions."

        v1 = engine.embed_query(text1)
        v2 = engine.embed_query(text2)
        v3 = engine.embed_query(unrelated)

        # Dot products
        sim_related = float(np.dot(v1, v2))
        sim_unrelated = float(np.dot(v1, v3))

        # Self-similarity should be 1.0
        assert np.isclose(float(np.dot(v1, v1)), 1.0, atol=1e-5)
        # Related texts should have significantly higher similarity than unrelated text
        assert sim_related > sim_unrelated
        assert sim_related > 0.65

    def test_embed_chunks(self, engine):
        chunks = [
            DocumentChunk(id="c1", document_id="d1", content="Chunk 1 content here."),
            DocumentChunk(id="c2", document_id="d1", content="Chunk 2 content here."),
        ]
        embeddings = engine.embed_chunks(chunks)
        assert embeddings.shape == (2, 384)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)

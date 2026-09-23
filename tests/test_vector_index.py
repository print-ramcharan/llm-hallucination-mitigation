"""Unit tests for FaissVectorIndex."""

from pathlib import Path

import pytest

from src.indexing.embeddings import EmbeddingEngine
from src.indexing.vector_index import FaissVectorIndex
from src.ingestion.models import DocumentChunk


class TestFaissVectorIndex:
    @pytest.fixture
    def engine(self):
        return EmbeddingEngine()

    def test_add_and_search(self, tmp_path: Path, engine: EmbeddingEngine):
        index = FaissVectorIndex(dimension=384, storage_dir=tmp_path)

        chunks = [
            DocumentChunk(id="c1", document_id="doc1", content="Employee health insurance and medical benefits."),
            DocumentChunk(id="c2", document_id="doc1", content="Paid time off and annual vacation leave."),
            DocumentChunk(id="c3", document_id="doc2", content="Python software development guidelines."),
        ]
        embeddings = engine.embed_chunks(chunks)
        index.add_chunks(chunks, embeddings)

        assert index.total_vectors == 3

        # Search for medical benefits
        query_vec = engine.embed_query("healthcare and insurance")
        results = index.search(query_vec, top_k=2)

        assert len(results) == 2
        top_chunk_id, top_score = results[0]
        assert top_chunk_id == "c1"
        assert top_score > 0.5

    def test_persistence_and_reload(self, tmp_path: Path, engine: EmbeddingEngine):
        # 1. Create and populate index
        index1 = FaissVectorIndex(dimension=384, storage_dir=tmp_path)
        chunks = [
            DocumentChunk(id="chunk_a", document_id="doc_a", content="Machine learning models."),
            DocumentChunk(id="chunk_b", document_id="doc_b", content="Database indexing techniques."),
        ]
        embeddings = engine.embed_chunks(chunks)
        index1.add_chunks(chunks, embeddings)
        assert index1.total_vectors == 2

        # 2. Reload into a new FaissVectorIndex instance pointing to same storage
        index2 = FaissVectorIndex(dimension=384, storage_dir=tmp_path)
        assert index2.total_vectors == 2

        query_vec = engine.embed_query("deep learning AI")
        results = index2.search(query_vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "chunk_a"

    def test_remove_document(self, tmp_path: Path, engine: EmbeddingEngine):
        index = FaissVectorIndex(dimension=384, storage_dir=tmp_path)
        chunks = [
            DocumentChunk(id="c1", document_id="doc1", content="Doc 1 chunk 1"),
            DocumentChunk(id="c2", document_id="doc1", content="Doc 1 chunk 2"),
            DocumentChunk(id="c3", document_id="doc2", content="Doc 2 chunk 1"),
        ]
        index.add_chunks(chunks, engine.embed_chunks(chunks))
        assert index.total_vectors == 3

        # Remove doc1
        index.remove_document("doc1")
        assert index.total_vectors == 1

        results = index.search(engine.embed_query("Doc"), top_k=5)
        assert len(results) == 1
        assert results[0][0] == "c3"

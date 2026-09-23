"""Unit tests for BM25Index."""

from pathlib import Path

from src.indexing.bm25_index import BM25Index, bm25_tokenize
from src.ingestion.models import DocumentChunk


class TestBM25Index:
    def test_tokenize(self):
        tokens = bm25_tokenize("Context-Degradation in LLMs (Module 1)")
        assert "context" in tokens
        assert "degradation" in tokens
        assert "llms" in tokens
        assert "module" in tokens
        assert "1" in tokens

    def test_add_and_search(self, tmp_path: Path):
        index = BM25Index(storage_dir=tmp_path)

        chunks = [
            DocumentChunk(id="c1", document_id="doc1", content="Comprehensive parental leave policy for employees."),
            DocumentChunk(id="c2", document_id="doc1", content="Salary review and quarterly performance bonuses."),
            DocumentChunk(id="c3", document_id="doc2", content="API authentication using Bearer tokens."),
        ]
        index.add_chunks(chunks)

        assert index.total_documents == 3

        # Exact keyword match
        results = index.search("parental leave", top_k=2)
        assert len(results) >= 1
        top_id, top_score = results[0]
        assert top_id == "c1"
        assert top_score > 0.0

    def test_persistence_and_reload(self, tmp_path: Path):
        index1 = BM25Index(storage_dir=tmp_path)
        chunks = [
            DocumentChunk(id="c1", document_id="doc1", content="TensorFlow neural network training."),
            DocumentChunk(id="c2", document_id="doc2", content="Kubernetes cluster orchestration."),
        ]
        index1.add_chunks(chunks)
        assert index1.total_documents == 2

        # Reload into new instance
        index2 = BM25Index(storage_dir=tmp_path)
        assert index2.total_documents == 2

        results = index2.search("Kubernetes", top_k=1)
        assert len(results) == 1
        assert results[0][0] == "c2"

    def test_remove_document(self, tmp_path: Path):
        index = BM25Index(storage_dir=tmp_path)
        chunks = [
            DocumentChunk(id="c1", document_id="doc1", content="Doc1 unique keyword alpha"),
            DocumentChunk(id="c2", document_id="doc2", content="Doc2 unique keyword beta"),
        ]
        index.add_chunks(chunks)
        assert index.total_documents == 2

        index.remove_document("doc1")
        assert index.total_documents == 1

        results = index.search("alpha", top_k=5)
        assert len(results) == 0

        results_beta = index.search("beta", top_k=5)
        assert len(results_beta) == 1
        assert results_beta[0][0] == "c2"

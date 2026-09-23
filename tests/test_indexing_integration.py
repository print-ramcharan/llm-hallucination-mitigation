"""Integration tests for IndexManager (Vector + BM25 + Metadata Store)."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app
from src.indexing.manager import IndexManager
from src.ingestion.models import Document, DocumentElement, DocumentType

client = TestClient(app)


class TestIndexManagerIntegration:
    def test_end_to_end_indexing(self, tmp_path: Path):
        manager = IndexManager(storage_dir=tmp_path)

        elements = [
            DocumentElement(id="el_0", element_type="heading", content="Leave Policy", section_title="Leave Policy", page_number=12),
            DocumentElement(
                id="el_1",
                element_type="paragraph",
                content="Full-time employees receive 20 days of paid vacation leave annually.",
                section_title="Leave Policy",
                page_number=12,
            ),
        ]

        doc = Document.create(
            source_name="employee_handbook.pdf",
            file_type=DocumentType.PDF,
            elements=elements,
            extra_metadata={"document_type": "HR_POLICY"},
        )

        res = manager.index_document(doc)
        assert res["status"] == "indexed_successfully"
        assert res["indexed_chunks"] == len(doc.chunks)

        # 1. Check Vector Index
        assert manager.vector_index.total_vectors == len(doc.chunks)
        query_vec = manager.embedding_engine.embed_query("vacation leave days")
        v_results = manager.vector_index.search(query_vec, top_k=1)
        assert len(v_results) == 1
        top_chunk_id, sim = v_results[0]
        assert "handbook_12_" in top_chunk_id
        assert sim > 0.6

        # 2. Check BM25 Index
        assert manager.bm25_index.total_documents == len(doc.chunks)
        bm_results = manager.bm25_index.search("vacation", top_k=1)
        assert len(bm_results) == 1
        assert bm_results[0][0] == top_chunk_id

        # 3. Check Metadata Store
        record = manager.metadata_store.get(top_chunk_id)
        assert record is not None
        assert record["metadata"]["document_type"] == "HR_POLICY"
        assert record["metadata"]["page"] == 12
        assert record["metadata"]["section"] == "Leave Policy"

        # 4. Check Stats
        stats = manager.get_stats()
        assert stats["vector_chunks"] == len(doc.chunks)
        assert stats["bm25_documents"] == len(doc.chunks)
        assert stats["metadata_chunks"] == len(doc.chunks)
        assert stats["embedding_dimension"] == 384


def test_api_indexes_stats_endpoint():
    res = client.get("/api/ingest/indexes/stats")
    assert res.status_code == 200
    data = res.json()
    assert "vector_chunks" in data
    assert "bm25_documents" in data
    assert "metadata_chunks" in data
    assert data["embedding_dimension"] == 384

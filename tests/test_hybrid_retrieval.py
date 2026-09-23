"""Unit and integration tests for Module 3: Hybrid Retrieval & Reciprocal Rank Fusion (RRF)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.indexing.manager import IndexManager
from src.ingestion.models import Document, DocumentElement, DocumentType
from src.query_processing.models import ConversationTurn, QueryIntent
from src.retrieval import (
    DenseRetriever,
    HybridRetriever,
    RetrievalResponse,
    RetrievedCandidate,
    RRFEngine,
    SparseRetriever,
)

client = TestClient(app)


@pytest.fixture
def test_manager(tmp_path: Path) -> IndexManager:
    """Creates a temporary IndexManager with sample multi-domain indexed documents."""
    manager = IndexManager(storage_dir=tmp_path)

    # Document 1: HR Policy (Vacation & Sick Leave)
    hr_elements = [
        DocumentElement(
            id="hr_el_0",
            element_type="heading",
            content="Annual Leave and Vacation Policy",
            section_title="Vacation Policy",
            page_number=1,
        ),
        DocumentElement(
            id="hr_el_1",
            element_type="paragraph",
            content="Employees are granted 20 days of paid annual vacation leave per calendar year. Requests must be submitted 2 weeks in advance.",
            section_title="Vacation Policy",
            page_number=1,
        ),
        DocumentElement(
            id="hr_el_2",
            element_type="paragraph",
            content="Sick leave provides up to 10 days of paid absence for medical appointments and illness without requiring prior notice.",
            section_title="Sick Leave",
            page_number=2,
        ),
    ]
    doc_hr = Document.create(
        source_name="employee_handbook_2024.pdf",
        file_type=DocumentType.PDF,
        elements=hr_elements,
        extra_metadata={"document_type": "HR_POLICY", "year": "2024", "department": "HR"},
    )
    manager.index_document(doc_hr)

    # Document 2: Technical Architecture Spec
    tech_elements = [
        DocumentElement(
            id="tech_el_0",
            element_type="heading",
            content="Vector Database Scaling and FAISS Architecture",
            section_title="Architecture",
            page_number=1,
        ),
        DocumentElement(
            id="tech_el_1",
            element_type="paragraph",
            content="The FAISS vector database utilizes IndexFlatIP for L2-normalized inner product cosine similarity at high throughput.",
            section_title="Architecture",
            page_number=1,
        ),
    ]
    doc_tech = Document.create(
        source_name="system_architecture_2024.md",
        file_type=DocumentType.MD,
        elements=tech_elements,
        extra_metadata={"document_type": "TECHNICAL_SPEC", "year": "2024", "department": "Engineering"},
    )
    manager.index_document(doc_tech)

    return manager


class TestRRFEngine:
    """Tests Reciprocal Rank Fusion algorithmic logic, score calculations, and candidate hydration."""

    def test_rrf_scoring_math(self):
        engine = RRFEngine()

        # Document A: Dense rank 1, Sparse rank 2
        # Document B: Dense rank 2, Sparse absent
        # Document C: Dense absent, Sparse rank 1
        dense_results = [("doc_A", 0.95), ("doc_B", 0.85)]
        sparse_results = [("doc_C", 12.4), ("doc_A", 9.1)]

        k = 60
        candidates = engine.fuse(dense_results, sparse_results, top_k=10, rrf_k=k)

        assert len(candidates) == 3

        # Candidate A appeared in both: 1/(60+1) + 1/(60+2)
        expected_score_a = (1.0 / 61.0) + (1.0 / 62.0)
        assert candidates[0].chunk_id == "doc_A"
        assert candidates[0].rrf_score == pytest.approx(expected_score_a, rel=1e-5)
        assert candidates[0].dense_rank == 1
        assert candidates[0].sparse_rank == 2

        # Candidate C was rank 1 in sparse: 1/(60+1)
        expected_score_c = 1.0 / 61.0
        assert candidates[1].chunk_id == "doc_C"
        assert candidates[1].rrf_score == pytest.approx(expected_score_c, rel=1e-5)
        assert candidates[1].dense_rank is None
        assert candidates[1].sparse_rank == 1

        # Candidate B was rank 2 in dense: 1/(60+2)
        expected_score_b = 1.0 / 62.0
        assert candidates[2].chunk_id == "doc_B"
        assert candidates[2].rrf_score == pytest.approx(expected_score_b, rel=1e-5)
        assert candidates[2].dense_rank == 2
        assert candidates[2].sparse_rank is None

    def test_rrf_custom_k_constant(self):
        engine = RRFEngine()
        dense_results = [("chunk_1", 0.9)]
        sparse_results = [("chunk_1", 8.0)]

        # With k=10: 1/(10+1) + 1/(10+1) = 2/11
        candidates = engine.fuse(dense_results, sparse_results, rrf_k=10)
        assert len(candidates) == 1
        assert candidates[0].rrf_score == pytest.approx(2.0 / 11.0, rel=1e-5)

    def test_rrf_top_k_truncation(self):
        engine = RRFEngine()
        dense = [(f"chunk_{i}", 1.0 / (i + 1)) for i in range(10)]
        sparse = [(f"chunk_{i}", 10.0 - i) for i in range(10)]

        candidates = engine.fuse(dense, sparse, top_k=3)
        assert len(candidates) == 3


class TestDenseRetriever:
    """Tests DenseRetriever semantic search and metadata filtering."""

    def test_dense_semantic_search(self, test_manager: IndexManager):
        retriever = DenseRetriever(
            vector_index=test_manager.vector_index,
            metadata_store=test_manager.metadata_store,
            embedding_engine=test_manager.embedding_engine,
        )

        results = retriever.retrieve("taking time off for holiday", top_k=2)
        assert len(results) >= 1
        top_chunk_id, sim_score = results[0]
        assert "handbook" in top_chunk_id.lower()
        assert sim_score > 0.4

    def test_dense_metadata_filtering(self, test_manager: IndexManager):
        retriever = DenseRetriever(
            vector_index=test_manager.vector_index,
            metadata_store=test_manager.metadata_store,
            embedding_engine=test_manager.embedding_engine,
        )

        # Query matches both tech and HR semantically, but filter restricts to Engineering
        results = retriever.retrieve(
            query="database architecture and scaling guidelines",
            top_k=5,
            filters={"department": "Engineering"},
        )
        assert len(results) >= 1
        for chunk_id, _ in results:
            record = test_manager.metadata_store.get(chunk_id)
            assert record["metadata"]["department"] == "Engineering"


class TestSparseRetriever:
    """Tests SparseRetriever BM25 lexical keyword matching and metadata filtering."""

    def test_sparse_keyword_search(self, test_manager: IndexManager):
        retriever = SparseRetriever(
            bm25_index=test_manager.bm25_index,
            metadata_store=test_manager.metadata_store,
        )

        results = retriever.retrieve("FAISS IndexFlatIP throughput", top_k=2)
        assert len(results) >= 1
        top_chunk_id, score = results[0]
        assert "architect" in top_chunk_id.lower()
        assert score > 0.0

    def test_sparse_empty_query(self, test_manager: IndexManager):
        retriever = SparseRetriever(
            bm25_index=test_manager.bm25_index,
            metadata_store=test_manager.metadata_store,
        )
        assert retriever.retrieve("", top_k=5) == []


class TestHybridRetrieverIntegration:
    """Tests end-to-end HybridRetriever with QueryProcessor conversational pre-flight and RRF."""

    def test_hybrid_retrieval_end_to_end(self, test_manager: IndexManager):
        retriever = HybridRetriever(index_manager=test_manager)

        response = retriever.retrieve("How many vacation leave days do employees receive?")
        assert isinstance(response, RetrievalResponse)
        assert response.total_candidates > 0
        assert response.dense_count > 0
        assert response.sparse_count > 0

        top_candidate = response.candidates[0]
        assert isinstance(top_candidate, RetrievedCandidate)
        assert "20 days" in top_candidate.content
        assert top_candidate.rrf_score > 0.0

    def test_conversational_pronoun_resolution_in_hybrid(self, test_manager: IndexManager):
        retriever = HybridRetriever(index_manager=test_manager)

        # History establishes context as annual vacation leave
        history = [
            ConversationTurn(role="user", content="Tell me about the vacation leave policy."),
            ConversationTurn(role="assistant", content="Employees get 20 days of vacation leave."),
        ]

        # Follow-up query uses anaphora "it"
        response = retriever.retrieve(
            query="Does it require advance notice to take it?",
            conversation_history=history,
        )

        assert response.processed_query.is_conversational is True
        assert "vacation" in response.processed_query.rewritten_query.lower()
        assert len(response.candidates) > 0
        assert "2 weeks in advance" in response.candidates[0].content

    def test_intent_and_filter_extraction_in_hybrid(self, test_manager: IndexManager):
        retriever = HybridRetriever(index_manager=test_manager)

        response = retriever.retrieve("Compare vacation leave vs sick leave in the 2024 policy")
        assert response.processed_query.intent == QueryIntent.COMPARISON
        assert response.processed_query.filters.get("year") == 2024


class TestHybridRetrievalAPI:
    """Tests FastAPI REST endpoints for hybrid retrieval."""

    def test_api_hybrid_retrieval(self, test_manager: IndexManager, monkeypatch):
        # Point default_hybrid_retriever to test_manager
        from src.api.retrieval_routes import default_hybrid_retriever
        monkeypatch.setattr(default_hybrid_retriever, "index_manager", test_manager)
        monkeypatch.setattr(
            default_hybrid_retriever,
            "dense_retriever",
            DenseRetriever(
                vector_index=test_manager.vector_index,
                metadata_store=test_manager.metadata_store,
                embedding_engine=test_manager.embedding_engine,
            ),
        )
        monkeypatch.setattr(
            default_hybrid_retriever,
            "sparse_retriever",
            SparseRetriever(
                bm25_index=test_manager.bm25_index,
                metadata_store=test_manager.metadata_store,
            ),
        )
        monkeypatch.setattr(
            default_hybrid_retriever,
            "rrf_engine",
            RRFEngine(metadata_store=test_manager.metadata_store),
        )

        payload = {
            "query": "What are the rules for sick leave?",
            "top_k_dense": 10,
            "top_k_sparse": 10,
            "top_k_fused": 5,
            "rrf_k": 60,
        }

        resp = client.post("/api/retrieval/hybrid", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["query"] == "What are the rules for sick leave?"
        assert data["total_candidates"] > 0
        assert len(data["candidates"]) <= 5
        assert "execution_time_ms" in data
        assert data["candidates"][0]["rrf_score"] > 0

    def test_api_retrieval_status(self):
        resp = client.get("/api/retrieval/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["module"] == "Hybrid Retrieval & RRF"
        assert "indexes" in data

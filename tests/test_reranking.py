"""Unit and integration tests for Module 4: Deep Cross-Encoder Reranking & Pruning."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.indexing.manager import IndexManager
from src.ingestion.models import Document, DocumentElement, DocumentType
from src.reranking import (
    CrossEncoderReranker,
    RankedContext,
    RerankedChunk,
    default_reranker,
)
from src.retrieval import HybridRetriever, RetrievedCandidate

client = TestClient(app)


class TestScoreCalibration:
    """Tests numerical properties of Sigmoid and Min-Max calibration functions."""

    def test_sigmoid_properties(self):
        logits = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])
        probs = CrossEncoderReranker.sigmoid(logits)

        # 1. Bounds: strictly within [0, 1]
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

        # 2. Symmetry: sigmoid(0) == 0.5
        assert probs[2] == pytest.approx(0.5, abs=1e-5)

        # 3. Monotonicity: strictly increasing
        assert np.all(np.diff(probs) > 0)

        # 4. Extreme bounds handling
        extremes = np.array([-1000.0, 1000.0])
        safe_probs = CrossEncoderReranker.sigmoid(extremes)
        assert safe_probs[0] == pytest.approx(0.0, abs=1e-5)
        assert safe_probs[1] == pytest.approx(1.0, abs=1e-5)

    def test_min_max_scale_properties(self):
        logits = np.array([2.0, 4.0, 6.0, 10.0])
        scaled = CrossEncoderReranker.min_max_scale(logits)

        assert scaled[0] == pytest.approx(0.0)
        assert scaled[-1] == pytest.approx(1.0)
        assert np.all(np.diff(scaled) > 0)

        # Constant array edge case
        constant = np.array([5.0, 5.0, 5.0])
        scaled_const = CrossEncoderReranker.min_max_scale(constant)
        assert np.all(scaled_const == 1.0)


class TestCrossEncoderReranker:
    """Tests CrossEncoder pairwise cross-attention scoring and pruning behavior."""

    @pytest.fixture
    def mock_candidates(self) -> list[RetrievedCandidate]:
        return [
            RetrievedCandidate(
                chunk_id="chunk_leave_policy",
                content="Full-time employees receive 20 days of paid annual vacation leave each year.",
                document_id="doc_handbook",
                chunk_index=0,
                metadata={"document_type": "HR_POLICY", "page": 1},
                dense_rank=1,
                sparse_rank=1,
                dense_score=0.88,
                sparse_score=14.2,
                rrf_score=0.032,
            ),
            RetrievedCandidate(
                chunk_id="chunk_code_review",
                content="All pull requests must receive approval from at least two senior engineers before merging.",
                document_id="doc_engineering",
                chunk_index=3,
                metadata={"document_type": "ENGINEERING", "page": 5},
                dense_rank=2,
                sparse_rank=2,
                dense_score=0.75,
                sparse_score=8.5,
                rrf_score=0.030,
            ),
            RetrievedCandidate(
                chunk_id="chunk_distractor_irrelevant",
                content="Photosynthesis converts solar light into chemical energy stored in glucose carbohydrate molecules.",
                document_id="doc_biology",
                chunk_index=1,
                metadata={"document_type": "GENERAL"},
                dense_rank=3,
                sparse_rank=3,
                dense_score=0.40,
                sparse_score=1.1,
                rrf_score=0.015,
            ),
        ]

    def test_cross_attention_scoring_accuracy(self, mock_candidates):
        query = "How many days of paid vacation do employees receive annually?"
        result = default_reranker.rerank(
            query=query,
            candidates=mock_candidates,
            threshold=0.1,
            top_n=5,
        )

        assert isinstance(result, RankedContext)
        assert len(result.chunks) >= 1

        # The vacation chunk must rank first with high confidence
        top_chunk: RerankedChunk = result.chunks[0]
        assert top_chunk.chunk_id == "chunk_leave_policy"
        assert top_chunk.rerank_position == 1
        assert top_chunk.rerank_score > 0.6
        assert "20 days" in top_chunk.content

    def test_hard_cutoff_threshold_pruning(self, mock_candidates):
        query = "How many days of paid vacation do employees receive annually?"
        # Set aggressive threshold tau=0.5: irrelevant biology chunk should be pruned
        result = default_reranker.rerank(
            query=query,
            candidates=mock_candidates,
            threshold=0.5,
            top_n=5,
        )

        assert result.threshold_applied == 0.5
        retained_ids = [c.chunk_id for c in result.chunks]
        assert "chunk_leave_policy" in retained_ids
        assert "chunk_distractor_irrelevant" not in retained_ids
        assert result.pruned_count >= 1
        assert result.retained_count == len(result.chunks)

    def test_top_n_retention_limit(self, mock_candidates):
        query = "guidelines and policies"
        # Request top_n = 1
        result = default_reranker.rerank(
            query=query,
            candidates=mock_candidates,
            threshold=0.0,
            top_n=1,
        )

        assert len(result.chunks) == 1
        assert result.retained_count == 1
        assert result.pruned_count == len(mock_candidates) - 1

    def test_empty_candidates_handling(self):
        result = default_reranker.rerank(query="any query", candidates=[])
        assert result.total_input_candidates == 0
        assert result.retained_count == 0
        assert result.pruned_count == 0
        assert len(result.chunks) == 0


class TestEndToEndHybridAndRerankPipeline:
    """Tests chaining Module 3 Hybrid Retrieval directly into Module 4 Cross-Encoder Reranker."""

    @pytest.fixture
    def indexed_manager(self, tmp_path: Path) -> IndexManager:
        manager = IndexManager(storage_dir=tmp_path)

        elements = [
            DocumentElement(
                id="hr_1",
                element_type="paragraph",
                content="Remote work allowances provide up to $500 annually for home office equipment purchases.",
                section_title="Remote Work",
                page_number=3,
            ),
            DocumentElement(
                id="hr_2",
                element_type="paragraph",
                content="Health and dental insurance coverage begins on the first calendar day of employment.",
                section_title="Health Benefits",
                page_number=4,
            ),
            DocumentElement(
                id="hr_3",
                element_type="paragraph",
                content="Standard working hours are 9:00 AM to 5:00 PM Monday through Friday with flexible scheduling.",
                section_title="Working Hours",
                page_number=5,
            ),
        ]

        doc = Document.create(
            source_name="handbook_benefits.pdf",
            file_type=DocumentType.PDF,
            elements=elements,
            extra_metadata={"document_type": "HR_POLICY"},
        )
        manager.index_document(doc)
        return manager

    def test_retrieval_and_rerank_chaining(self, indexed_manager: IndexManager):
        hybrid_retriever = HybridRetriever(index_manager=indexed_manager)

        # 1. Module 3: Hybrid Retrieval
        query = "What is the annual home office stipend for remote employees?"
        retrieval_response = hybrid_retriever.retrieve(query=query, top_k_fused=10)
        assert retrieval_response.total_candidates > 0

        # 2. Module 4: Cross-Encoder Reranking & Pruning
        ranked_context = default_reranker.rerank_retrieval_response(
            response=retrieval_response,
            threshold=0.3,
            top_n=3,
        )

        assert isinstance(ranked_context, RankedContext)
        assert len(ranked_context.chunks) <= 3
        assert len(ranked_context.chunks) >= 1

        top_chunk = ranked_context.chunks[0]
        assert "$500" in top_chunk.content
        assert "Remote work" in top_chunk.content
        assert top_chunk.rerank_position == 1
        assert top_chunk.rerank_score >= 0.3


class TestRerankingAPI:
    """Tests FastAPI REST endpoints for cross-encoder reranking."""

    def test_api_rerank_endpoint(self):
        payload = {
            "query": "Where can I find employee vacation rules?",
            "candidates": [
                {
                    "chunk_id": "c_vacation",
                    "content": "Employees are entitled to 20 days paid vacation annually.",
                    "document_id": "doc_1",
                    "chunk_index": 0,
                    "metadata": {"document_type": "HR_POLICY"},
                    "rrf_score": 0.03,
                },
                {
                    "chunk_id": "c_hardware",
                    "content": "Laptops will be refreshed every three years by IT.",
                    "document_id": "doc_2",
                    "chunk_index": 1,
                    "metadata": {"document_type": "IT_SPEC"},
                    "rrf_score": 0.02,
                },
            ],
            "threshold": 0.2,
            "top_n": 2,
            "normalization": "sigmoid",
        }

        resp = client.post("/api/rerank", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["query"] == "Where can I find employee vacation rules?"
        assert data["total_input_candidates"] == 2
        assert len(data["chunks"]) >= 1
        assert data["chunks"][0]["chunk_id"] == "c_vacation"
        assert data["chunks"][0]["rerank_position"] == 1
        assert data["chunks"][0]["rerank_score"] > 0.3

    def test_api_rerank_status_endpoint(self):
        resp = client.get("/api/rerank/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["module"] == "Deep Cross-Encoder Reranking & Pruning"
        assert "cross-encoder" in data["model_name"]
        assert data["default_threshold"] == 0.35
        assert data["default_top_n"] == 5

"""Unit and integration tests for Module 5: Extractive Context Compaction & Optimization."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.context import (
    ContextCompactor,
    ContextDeduplicator,
    LostInTheMiddleReorderer,
    OptimizedContext,
    SentencePruner,
    TokenBudgetManager,
    default_context_compactor,
)
from src.indexing.manager import IndexManager
from src.ingestion.models import Document, DocumentElement, DocumentType
from src.reranking.models import RerankedChunk

client = TestClient(app)


class TestSentencePruner:
    """Tests sentence boundary extraction and salience scoring."""

    def test_sentence_splitting(self):
        text = "First sentence here. Second sentence follows! Third sentence is a question?"
        sentences = SentencePruner.split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence here."
        assert sentences[1] == "Second sentence follows!"
        assert sentences[2] == "Third sentence is a question?"

    def test_pruning_extracts_salient_sentences(self):
        pruner = SentencePruner(default_min_salience=0.2)
        query = "annual vacation leave allowance"
        text = (
            "Employees are granted 20 days of paid annual vacation leave per calendar year. "
            "The office air conditioning operates from 8 AM to 6 PM daily. "
            "Requests for extended time off should be filed through the HR portal."
        )

        compacted, kept = pruner.prune_chunk(query=query, text=text)
        # The AC sentence is irrelevant filler prose and should be pruned
        assert "air conditioning" not in compacted
        assert "20 days" in compacted
        assert len(kept) < 3

    def test_fallback_preservation_avoids_empty(self):
        pruner = SentencePruner(default_min_salience=0.99)
        query = "unrelated query"
        text = "This is a single short passage without strong keyword overlap."

        compacted, kept = pruner.prune_chunk(query=query, text=text)
        assert len(kept) == 1
        assert len(compacted) > 0


class TestContextDeduplicator:
    """Tests chunk seam overlap detection and Jaccard deduplication."""

    def test_deduplicate_chunk_overlap_seam(self):
        dedup = ContextDeduplicator(default_similarity_threshold=0.75)
        seen_terms = []

        # Chunk 1 contains Sentence A and Sentence B (overlap seam)
        chunk_1_sentences = [
            "Employees receive 20 days of paid annual vacation leave.",
            "Vacation requests must be submitted at least two weeks in advance.",
        ]
        unique_1, pruned_1 = dedup.deduplicate_sentences(chunk_1_sentences, seen_terms)
        assert len(unique_1) == 2
        assert pruned_1 == 0

        # Chunk 2 begins with Sentence B (seam duplicate) and continues with Sentence C
        chunk_2_sentences = [
            "Vacation requests must be submitted at least two weeks in advance.",
            "Unused leave days may be carried over up to a maximum of 5 days.",
        ]
        unique_2, pruned_2 = dedup.deduplicate_sentences(chunk_2_sentences, seen_terms)
        assert len(unique_2) == 1
        assert pruned_2 == 1
        assert "Unused leave days" in unique_2[0]
        assert "two weeks in advance" not in unique_2[0]

    def test_jaccard_similarity_calculation(self):
        set_a = {"vacation", "policy", "days", "annual"}
        set_b = {"vacation", "policy", "days", "annual"}
        assert ContextDeduplicator.jaccard_similarity(set_a, set_b) == 1.0

        set_c = {"completely", "different", "words"}
        assert ContextDeduplicator.jaccard_similarity(set_a, set_c) == 0.0


class TestLostInTheMiddleReorderer:
    """Tests U-shaped attention distribution positioning."""

    def test_u_shaped_reordering_five_elements(self):
        # 5 items sorted descending by rank: 1 (best), 2, 3, 4, 5 (lowest)
        items = ["Rank_1", "Rank_2", "Rank_3", "Rank_4", "Rank_5"]
        reordered = LostInTheMiddleReorderer.reorder(items)

        assert len(reordered) == 5
        # Index 0 (Beginning / Top): Rank 1
        assert reordered[0] == "Rank_1"
        # Index 4 (End / Bottom, before prompt): Rank 2
        assert reordered[4] == "Rank_2"
        # Index 1: Rank 3
        assert reordered[1] == "Rank_3"
        # Index 3: Rank 4
        assert reordered[3] == "Rank_4"
        # Index 2 (Center / Middle): Rank 5 (lowest confidence)
        assert reordered[2] == "Rank_5"

    def test_u_shaped_reordering_three_elements(self):
        items = ["Rank_1", "Rank_2", "Rank_3"]
        reordered = LostInTheMiddleReorderer.reorder(items)
        assert reordered == ["Rank_1", "Rank_3", "Rank_2"]

    def test_small_lists_preserved(self):
        items = ["Rank_1", "Rank_2"]
        assert LostInTheMiddleReorderer.reorder(items) == items


class TestTokenBudgetManager:
    """Tests strict prompt budget enforcement and citation formatting."""

    def test_citation_tag_formatting(self):
        mgr = TokenBudgetManager()
        tag = mgr.generate_citation_tag(doc_num=1, chunk_index=2)
        assert tag == "[Doc 1, Chunk 3]"

        formatted = mgr.format_evidence_block(
            citation_tag=tag,
            text="Evidence content here.",
            metadata={"source": "handbook.pdf", "page": 4, "section": "Leave"},
        )
        assert "[Doc 1, Chunk 3]" in formatted
        assert "Source: handbook.pdf" in formatted
        assert "Page 4" in formatted
        assert "Section: Leave" in formatted

    def test_budget_enforcement(self):
        mgr = TokenBudgetManager(default_budget=60)
        candidates = [
            {
                "document_id": "doc_1",
                "chunk_index": 0,
                "extracted_text": "First evidence segment with moderate token count.",
                "metadata": {"source": "doc1.txt"},
            },
            {
                "document_id": "doc_2",
                "chunk_index": 0,
                "extracted_text": "Second evidence segment fitting inside the budget allowance.",
                "metadata": {"source": "doc2.txt"},
            },
            {
                "document_id": "doc_3",
                "chunk_index": 0,
                "extracted_text": "This third evidence segment exceeds the tight 60 token limit and must be dropped.",
                "metadata": {"source": "doc3.txt"},
            },
        ]

        context, accepted, total_tokens = mgr.fit_within_budget(candidates, max_budget=60)
        assert total_tokens <= 60
        assert len(accepted) < len(candidates)
        assert "[Doc 1, Chunk 1]" in context


class TestContextCompactorIntegration:
    """Tests master coordinator integrating pruning, deduplication, U-shaped reordering, and budgeting."""

    @pytest.fixture
    def sample_reranked_chunks(self) -> list[RerankedChunk]:
        return [
            RerankedChunk(
                chunk_id="c1",
                content="Full-time employees receive 20 days of paid vacation annually. Office chairs are ergonomically designed.",
                document_id="doc_leave",
                chunk_index=0,
                metadata={"source": "leave_policy.pdf", "page": 1, "section": "Vacation"},
                rerank_score=0.92,
                raw_score=4.2,
                rerank_position=1,
            ),
            RerankedChunk(
                chunk_id="c2",
                content="Vacation time accrues on a monthly basis starting from day one. Office chairs are ergonomically designed.",
                document_id="doc_leave",
                chunk_index=1,
                metadata={"source": "leave_policy.pdf", "page": 1, "section": "Vacation"},
                rerank_score=0.85,
                raw_score=3.1,
                rerank_position=2,
            ),
            RerankedChunk(
                chunk_id="c3",
                content="Sick leave provides up to 10 days of paid medical leave per calendar year.",
                document_id="doc_sick",
                chunk_index=0,
                metadata={"source": "sick_policy.pdf", "page": 2, "section": "Sick Leave"},
                rerank_score=0.70,
                raw_score=1.8,
                rerank_position=3,
            ),
        ]

    def test_compaction_pipeline_end_to_end(self, sample_reranked_chunks):
        query = "annual vacation leave days"
        compactor = ContextCompactor()

        result = compactor.compact(
            query=query,
            chunks=sample_reranked_chunks,
            max_token_budget=1500,
            enable_sentence_pruning=True,
            enable_deduplication=True,
            enable_lost_in_middle_reordering=True,
        )

        assert isinstance(result, OptimizedContext)
        assert len(result.evidence_items) == 3
        assert result.total_tokens <= 1500
        assert result.total_tokens < result.original_tokens
        assert result.saved_tokens > 0

        # Duplicate sentence "Office chairs are ergonomically designed" pruned from chunk 2
        assert result.dedup_pruned_count >= 1

        # U-shaped reordering: Rank 1 at index 0, Rank 2 at end, Rank 3 in middle
        assert result.evidence_items[0].chunk_id == "c1"
        assert result.evidence_items[1].chunk_id == "c3"
        assert result.evidence_items[2].chunk_id == "c2"

        # Citation tags verified
        assert "[Doc 1, Chunk 1]" in result.formatted_prompt_context
        assert "[Doc 2, Chunk 1]" in result.formatted_prompt_context


class TestFullContextPipelineWithIndexes:
    """Tests end-to-end chain: Document Ingestion -> Hybrid Retrieval -> Rerank -> Compact."""

    def test_ingest_retrieve_rerank_compact_chain(self, tmp_path: Path):
        manager = IndexManager(storage_dir=tmp_path)
        elements = [
            DocumentElement(
                id="el_1",
                element_type="paragraph",
                content="Full-time staff receive 20 days paid vacation leave annually. Requests require 2 weeks notice.",
                section_title="Vacation Policy",
                page_number=1,
            ),
            DocumentElement(
                id="el_2",
                element_type="paragraph",
                content="Requests require 2 weeks notice. Unused vacation cannot exceed 5 carryover days.",
                section_title="Vacation Policy",
                page_number=1,
            ),
        ]
        doc = Document.create(
            source_name="company_policy.pdf",
            file_type=DocumentType.PDF,
            elements=elements,
            extra_metadata={"document_type": "HR_POLICY"},
        )
        manager.index_document(doc)

        from src.reranking.reranker import default_reranker
        from src.retrieval.hybrid import HybridRetriever

        hybrid = HybridRetriever(index_manager=manager)
        retrieval_res = hybrid.retrieve(query="How much vacation leave do staff receive?")
        ranked_context = default_reranker.rerank_retrieval_response(retrieval_res, top_n=3)
        optimized_context = default_context_compactor.compact_ranked_context(ranked_context)

        assert isinstance(optimized_context, OptimizedContext)
        assert len(optimized_context.evidence_items) >= 1
        assert "20 days" in optimized_context.formatted_prompt_context
        assert "[Doc 1, Chunk 1]" in optimized_context.formatted_prompt_context


class TestContextAPI:
    """Tests FastAPI REST routes for context optimization."""

    def test_api_compact_endpoint(self):
        payload = {
            "query": "vacation leave rules",
            "chunks": [
                {
                    "chunk_id": "chk_1",
                    "content": "Employees get 20 days paid vacation annually. The sun rises in the east.",
                    "document_id": "doc_hr",
                    "chunk_index": 0,
                    "metadata": {"source": "hr.pdf", "page": 1},
                    "rerank_score": 0.9,
                    "raw_score": 3.5,
                    "rerank_position": 1,
                }
            ],
            "max_token_budget": 500,
        }

        resp = client.post("/api/context/compact", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["query"] == "vacation leave rules"
        assert len(data["evidence_items"]) == 1
        assert "[Doc 1, Chunk 1]" in data["formatted_prompt_context"]
        assert data["total_tokens"] > 0

    def test_api_context_status_endpoint(self):
        resp = client.get("/api/context/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["default_token_budget"] == 1500

"""Unit and integration tests for Module 6: Grounded LLM Inference & Anti-Hallucination Guardrails."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.context.models import CompactedEvidence, OptimizedContext
from src.generation import (
    AntiHallucinationVerifier,
    ClaimStatus,
    EvidenceSufficiencyClassifier,
    GenerationResponse,
    GroundedGenerator,
    GroundedPromptSynthesizer,
)
from src.query_processing.models import ConversationTurn

client = TestClient(app)


@pytest.fixture
def sample_vacation_evidence() -> CompactedEvidence:
    """Fixture providing a sample compacted evidence item for vacation policies."""
    return CompactedEvidence(
        citation_tag="[Doc 1, Chunk 0]",
        document_id="doc_hr_policy_01",
        chunk_id="chunk_hr_01",
        original_text=(
            "Employees are granted 20 days of paid annual vacation leave per calendar year. "
            "Leave requests must be submitted at least two weeks in advance through the HR portal. "
            "Unused vacation days cannot be rolled over into the following calendar year."
        ),
        extracted_text=(
            "Employees are granted 20 days of paid annual vacation leave per calendar year. "
            "Leave requests must be submitted at least two weeks in advance through the HR portal."
        ),
        token_count=35,
        original_token_count=52,
        compression_ratio=0.67,
        reorder_position=1,
        metadata={"source": "HR_Policy_2024.pdf", "section": "Leave Entitlement"},
    )


@pytest.fixture
def sample_optimized_context(
    sample_vacation_evidence: CompactedEvidence,
) -> OptimizedContext:
    """Fixture providing an OptimizedContext container with evidence."""
    formatted = (
        f"--- EVIDENCE {sample_vacation_evidence.citation_tag} (Source: HR_Policy_2024.pdf) ---\n"
        f"{sample_vacation_evidence.extracted_text}"
    )
    return OptimizedContext(
        query="annual vacation leave days allowance",
        formatted_prompt_context=formatted,
        evidence_items=[sample_vacation_evidence],
        total_tokens=40,
        original_tokens=60,
        saved_tokens=20,
        token_budget=1500,
        dedup_pruned_count=0,
        reordered=True,
        execution_time_ms=1.5,
    )


class TestEvidenceSufficiencyClassifier:
    """Tests the pre-flight Evidence Sufficiency Classifier and Truthful Abstention Protocol."""

    def test_sufficient_evidence_passes(
        self, sample_optimized_context: OptimizedContext
    ):
        classifier = EvidenceSufficiencyClassifier(default_threshold=0.40)
        query = "How many days of paid vacation leave do employees receive per year?"
        assessment = classifier.assess_sufficiency(
            query=query, context=sample_optimized_context
        )

        assert assessment.is_sufficient is True
        assert assessment.sufficiency_score >= 0.40
        assert assessment.abstention_message is None
        assert "vacation" in assessment.matched_aspects
        assert "leave" in assessment.matched_aspects

    def test_insufficient_evidence_triggers_truthful_abstention(
        self, sample_optimized_context: OptimizedContext
    ):
        classifier = EvidenceSufficiencyClassifier(default_threshold=0.40)
        # Query about an unmentioned topic: corporate pet guidelines
        query = "What is the corporate pet policy for bringing dogs into the office?"
        assessment = classifier.assess_sufficiency(
            query=query, context=sample_optimized_context
        )

        assert assessment.is_sufficient is False
        assert assessment.sufficiency_score < 0.40
        assert assessment.abstention_message is not None
        assert (
            "The available documents do not contain sufficient evidence to verify"
            in assessment.abstention_message
        )
        assert len(assessment.missing_aspects) > 0

    def test_empty_evidence_triggers_immediate_abstention(self):
        classifier = EvidenceSufficiencyClassifier(default_threshold=0.40)
        assessment = classifier.assess_sufficiency(
            query="Any question", context=None, evidence_items=[]
        )

        assert assessment.is_sufficient is False
        assert assessment.sufficiency_score == 0.0
        assert assessment.abstention_message is not None
        assert (
            "The available documents do not contain sufficient evidence"
            in assessment.abstention_message
        )


class TestGroundedPromptSynthesizer:
    """Tests prompt construction and citation requirement injection."""

    def test_synthesize_prompt_structure(
        self, sample_optimized_context: OptimizedContext
    ):
        synthesizer = GroundedPromptSynthesizer()
        prompt = synthesizer.synthesize_prompt(
            query="How many vacation days are granted?",
            context=sample_optimized_context,
        )

        assert "EVIDENCE CONTEXT:" in prompt
        assert "[Doc 1, Chunk 0]" in prompt
        assert "USER QUERY:" in prompt
        assert "How many vacation days are granted?" in prompt
        assert "GROUNDED ANSWER" in prompt

    def test_conversation_history_injection(
        self, sample_optimized_context: OptimizedContext
    ):
        synthesizer = GroundedPromptSynthesizer()
        history = [
            ConversationTurn(role="user", content="Hello, I have an HR question."),
            ConversationTurn(
                role="assistant", content="Sure! What would you like to know?"
            ),
        ]
        prompt = synthesizer.synthesize_prompt(
            query="What is the leave allowance?",
            context=sample_optimized_context,
            conversation_history=history,
        )

        assert "CONVERSATION HISTORY:" in prompt
        assert "User: Hello, I have an HR question." in prompt
        assert "Assistant: Sure! What would you like to know?" in prompt


class TestAntiHallucinationVerifier:
    """Tests sentence-level NLI verification and contradiction detection."""

    def test_entailed_claim_verification(
        self, sample_vacation_evidence: CompactedEvidence
    ):
        verifier = AntiHallucinationVerifier(default_tolerance=0.70)
        answer = "Employees are granted 20 days of paid annual vacation leave per calendar year [Doc 1, Chunk 0]."

        report = verifier.verify_answer(
            answer=answer, evidence_items=[sample_vacation_evidence]
        )

        assert report.total_claims == 1
        assert report.entailed_claims_count == 1
        assert report.contradicted_claims_count == 0
        assert report.faithfulness_score >= 0.80
        assert report.hallucination_detected is False
        assert "[Doc 1, Chunk 0]" in report.verified_citations

    def test_numerical_contradiction_detected(
        self, sample_vacation_evidence: CompactedEvidence
    ):
        verifier = AntiHallucinationVerifier(default_tolerance=0.70)
        # Context states 20 days, but claim asserts 60 days
        answer = "Employees receive 60 days of annual vacation leave [Doc 1, Chunk 0]."

        report = verifier.verify_answer(
            answer=answer, evidence_items=[sample_vacation_evidence]
        )

        assert report.total_claims == 1
        assert report.contradicted_claims_count == 1
        assert report.hallucination_detected is True
        assert report.claims[0].status == ClaimStatus.CONTRADICTED
        assert "Numerical contradiction" in (report.claims[0].reasoning or "")

    def test_unverified_citation_flagged(
        self, sample_vacation_evidence: CompactedEvidence
    ):
        verifier = AntiHallucinationVerifier(default_tolerance=0.70)
        # Citation tag [Doc 99, Chunk 99] is fabricated and not in evidence context
        answer = "Employees submit requests through the portal [Doc 99, Chunk 99]."

        report = verifier.verify_answer(
            answer=answer, evidence_items=[sample_vacation_evidence]
        )

        assert "[Doc 99, Chunk 99]" in report.unverified_citations


class TestGroundedGenerator:
    """Tests end-to-end grounded generation coordinator and streaming."""

    def test_generate_sufficient_returns_grounded_answer(
        self, sample_optimized_context: OptimizedContext
    ):
        generator = GroundedGenerator()
        response: GenerationResponse = generator.generate(
            query="What is the annual vacation leave allowance?",
            context=sample_optimized_context,
        )

        assert response.abstained is False
        assert response.abstention_reason is None
        assert response.sufficiency.is_sufficient is True
        assert len(response.citations) > 0
        assert "[Doc 1, Chunk 0]" in response.citations
        assert response.grounding_report.faithfulness_score > 0.0

    def test_generate_insufficient_triggers_abstention_protocol(
        self, sample_optimized_context: OptimizedContext
    ):
        generator = GroundedGenerator()
        response: GenerationResponse = generator.generate(
            query="What are the rules for bring your pet hedgehog to work?",
            context=sample_optimized_context,
        )

        assert response.abstained is True
        assert response.sufficiency.is_sufficient is False
        assert (
            "The available documents do not contain sufficient evidence"
            in response.answer
        )

    def test_generate_stream_yields_sse_events(
        self, sample_optimized_context: OptimizedContext
    ):
        generator = GroundedGenerator()
        events = list(
            generator.generate_stream(
                query="How many vacation leave days are provided?",
                context=sample_optimized_context,
            )
        )

        # Should yield sufficiency, start, tokens, grounding, and done events
        event_types = [e for e in events if e.startswith("event:")]
        assert any("event: sufficiency" in e for e in event_types)
        assert any("event: start" in e for e in event_types)
        assert any("event: token" in e for e in event_types)
        assert any("event: grounding" in e for e in event_types)
        assert any("event: done" in e for e in event_types)


class TestGenerationApiEndpoints:
    """Tests FastAPI REST and SSE endpoints for Module 6."""

    def test_api_generation_status(self):
        resp = client.get("/api/generate/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["abstention_protocol"] == "enabled"
        assert "default_sufficiency_threshold" in data

    def test_api_generate_answer_sufficient(
        self, sample_optimized_context: OptimizedContext
    ):
        payload = {
            "query": "How many days of annual vacation leave are granted?",
            "context": sample_optimized_context.model_dump(),
            "sufficiency_threshold": 0.40,
        }
        resp = client.post("/api/generate/answer", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["abstained"] is False
        assert len(data["citations"]) > 0
        assert "grounding_report" in data

    def test_api_generate_answer_abstention(
        self, sample_optimized_context: OptimizedContext
    ):
        payload = {
            "query": "What is the pet policy for lizards?",
            "context": sample_optimized_context.model_dump(),
            "sufficiency_threshold": 0.40,
        }
        resp = client.post("/api/generate/answer", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["abstained"] is True
        assert (
            "The available documents do not contain sufficient evidence"
            in data["answer"]
        )

    def test_api_generate_stream(self, sample_optimized_context: OptimizedContext):
        payload = {
            "query": "How many vacation leave days are given?",
            "context": sample_optimized_context.model_dump(),
        }
        resp = client.post("/api/generate/stream", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "event: sufficiency" in body
        assert "event: done" in body

    def test_api_full_qa_pipeline_grounded(self, monkeypatch):
        from src.api import generation_routes
        from src.query_processing.models import ProcessedQuery, QueryIntent
        from src.reranking.models import RankedContext, RerankedChunk
        from src.retrieval.models import RetrievalResponse, RetrievedCandidate

        mock_candidate = RetrievedCandidate(
            chunk_id="chunk_hr_01",
            content="Employees are granted 20 days of paid annual vacation leave per calendar year [Doc 1, Chunk 0].",
            document_id="doc_hr",
            chunk_index=0,
            metadata={"source": "HR_Policy_2024.pdf", "page": 1},
            dense_rank=1,
            sparse_rank=1,
            dense_score=0.92,
            sparse_score=12.5,
            rrf_score=0.032,
        )
        mock_retrieval = RetrievalResponse(
            query="How many vacation leave days are given?",
            processed_query=ProcessedQuery(
                original_query="How many vacation leave days are given?",
                rewritten_query="How many vacation leave days are given?",
                query="How many vacation leave days are given?",
                intent=QueryIntent.FACTUAL,
            ),
            candidates=[mock_candidate],
            total_candidates=1,
            dense_count=1,
            sparse_count=1,
            execution_time_ms=1.2,
        )
        mock_ranked = RankedContext(
            query="How many vacation leave days are given?",
            chunks=[
                RerankedChunk(
                    chunk_id="chunk_hr_01",
                    content="Employees are granted 20 days of paid annual vacation leave per calendar year.",
                    document_id="doc_hr",
                    chunk_index=0,
                    metadata={"source": "HR_Policy_2024.pdf", "page": 1},
                    initial_rank=1,
                    initial_score=0.032,
                    rerank_score=0.95,
                    raw_score=4.8,
                    rerank_position=1,
                )
            ],
            total_input_candidates=1,
            retained_count=1,
            pruned_count=0,
            threshold_applied=0.35,
            model_name="mock-cross-encoder",
            execution_time_ms=1.5,
        )

        monkeypatch.setattr(
            generation_routes.default_hybrid_retriever,
            "retrieve",
            lambda **kwargs: mock_retrieval,
        )
        monkeypatch.setattr(
            generation_routes.default_reranker,
            "rerank_retrieval_response",
            lambda **kwargs: mock_ranked,
        )

        payload = {
            "query": "How many vacation leave days are given?",
            "top_k_fused": 10,
            "rerank_top_n": 3,
            "max_token_budget": 1000,
        }
        resp = client.post("/api/generate/pipeline/qa", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sufficiency" in data
        assert "grounding_report" in data
        assert data["abstained"] is False
        assert len(data["citations"]) > 0

    def test_api_full_qa_pipeline_unseeded_abstains(self, monkeypatch):
        from src.api import generation_routes
        from src.query_processing.models import ProcessedQuery, QueryIntent
        from src.reranking.models import RankedContext
        from src.retrieval.models import RetrievalResponse

        mock_empty_retrieval = RetrievalResponse(
            query="Unknown policy question",
            processed_query=ProcessedQuery(
                original_query="Unknown policy question",
                rewritten_query="Unknown policy question",
                query="Unknown policy question",
                intent=QueryIntent.FACTUAL,
            ),
            candidates=[],
            total_candidates=0,
            dense_count=0,
            sparse_count=0,
            execution_time_ms=0.5,
        )
        mock_empty_ranked = RankedContext(
            query="Unknown policy question",
            chunks=[],
            total_input_candidates=0,
            retained_count=0,
            pruned_count=0,
            threshold_applied=0.35,
            model_name="mock-cross-encoder",
            execution_time_ms=0.5,
        )

        monkeypatch.setattr(
            generation_routes.default_hybrid_retriever,
            "retrieve",
            lambda **kwargs: mock_empty_retrieval,
        )
        monkeypatch.setattr(
            generation_routes.default_reranker,
            "rerank_retrieval_response",
            lambda **kwargs: mock_empty_ranked,
        )

        payload = {
            "query": "What is the policy for orbital rocket launching?",
            "top_k_fused": 5,
            "rerank_top_n": 2,
            "max_token_budget": 1000,
        }
        resp = client.post("/api/generate/pipeline/qa", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["abstained"] is True
        assert (
            "The available documents do not contain sufficient evidence"
            in data["answer"]
        )

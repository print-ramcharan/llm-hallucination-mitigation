"""Unit and integration tests for Module 2: Query Processing."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.query_processing import (
    ConversationTurn,
    FilterExtractor,
    IntentDetector,
    ProcessedQuery,
    QueryIntent,
    QueryRewriter,
    default_query_processor,
)

client = TestClient(app)


class TestQueryRewriter:
    @pytest.fixture
    def rewriter(self):
        return QueryRewriter()

    def test_standalone_query_unchanged(self, rewriter):
        query = "What is our remote working policy?"
        rewritten = rewriter.rewrite(query)
        assert rewritten == query

    def test_conversational_ellipsis_rewriting(self, rewriter):
        history = [
            ConversationTurn(role="user", content="What is the leave policy?"),
            ConversationTurn(role="assistant", content="Employees receive 20 days of annual leave."),
        ]
        rewritten = rewriter.rewrite("What about sick leave?", history)
        assert rewritten == "What is the employee sick leave policy?"

    def test_pronoun_resolution(self, rewriter):
        history = [
            ConversationTurn(role="user", content="Tell me about the travel expense policy"),
            ConversationTurn(role="assistant", content="Travel expenses must be submitted within 30 days."),
        ]
        rewritten = rewriter.rewrite("Does it cover international flights?", history)
        assert "travel expense policy" in rewritten.lower()


class TestIntentDetector:
    @pytest.fixture
    def detector(self):
        return IntentDetector()

    def test_informational_intent(self, detector):
        assert detector.detect("What is our remote working policy?") == QueryIntent.INFORMATIONAL
        assert detector.detect("What is the leave policy?") == QueryIntent.INFORMATIONAL

    def test_comparison_intent(self, detector):
        assert detector.detect("Compare our leave policy with remote work policy") == QueryIntent.COMPARISON
        assert detector.detect("What is the difference between casual and sick leave?") == QueryIntent.COMPARISON
        assert detector.detect("Hybrid vs full remote requirements") == QueryIntent.COMPARISON

    def test_summary_intent(self, detector):
        assert detector.detect("Summarize the employee handbook") == QueryIntent.SUMMARY
        assert detector.detect("Give me a brief overview of the code of conduct") == QueryIntent.SUMMARY
        assert detector.detect("TL;DR on retirement benefits") == QueryIntent.SUMMARY

    def test_exploratory_intent(self, detector):
        assert detector.detect("What topics are covered in the guidelines?") == QueryIntent.EXPLORATORY
        assert detector.detect("What policies do we have for new parents?") == QueryIntent.EXPLORATORY
        assert detector.detect("Explore the security architecture documentation") == QueryIntent.EXPLORATORY

    def test_keyword_lookup(self, detector):
        assert detector.detect("leave policy") == QueryIntent.KEYWORD_LOOKUP


class TestFilterExtractor:
    @pytest.fixture
    def extractor(self):
        return FilterExtractor()

    def test_year_extraction(self, extractor):
        query, filters = extractor.extract("What was the 2025 leave policy?")
        assert filters == {"year": 2025}
        assert query == "leave policy"

    def test_department_extraction(self, extractor):
        query, filters = extractor.extract("What does the engineering handbook say about deployment?")
        assert filters == {"department": "engineering"}
        assert query == "deployment"

    def test_both_year_and_department(self, extractor):
        query, filters = extractor.extract("What did the 2024 engineering policy say about remote work?")
        assert filters == {"year": 2024, "department": "engineering"}
        assert query == "engineering remote work policy"


class TestQueryProcessorPipeline:
    @pytest.fixture
    def processor(self):
        return default_query_processor

    def test_user_acceptance_criteria(self, processor):
        """User acceptance test:

        Send: 'What did the 2024 engineering policy say about remote work?'
        Receive: query='engineering remote work policy', filters={'year': 2024, 'department': 'engineering'}
        """
        result: ProcessedQuery = processor.process(
            "What did the 2024 engineering policy say about remote work?"
        )

        assert result.query == "engineering remote work policy"
        assert result.filters == {
            "year": 2024,
            "department": "engineering",
        }
        assert result.intent == QueryIntent.INFORMATIONAL
        assert "engineering" in result.extracted_keywords
        assert "remote" in result.extracted_keywords

    def test_full_conversational_workflow(self, processor):
        history = [
            ConversationTurn(role="user", content="What is the leave policy?"),
            ConversationTurn(role="assistant", content="Employees receive 20 days off annually."),
        ]
        result = processor.process("What about sick leave?", conversation_history=history)

        assert result.is_conversational is True
        assert result.rewritten_query == "What is the employee sick leave policy?"
        assert "sick leave" in result.query


class TestQueryProcessingAPI:
    def test_api_process_query_endpoint(self):
        payload = {
            "query": "What did the 2024 engineering policy say about remote work?",
        }
        response = client.post("/api/query/process", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["query"] == "engineering remote work policy"
        assert data["filters"] == {"year": 2024, "department": "engineering"}
        assert data["intent"] == "informational"
        assert data["rewritten_query"] == "What did the 2024 engineering policy say about remote work?"

    def test_api_conversational_query(self):
        payload = {
            "query": "What about sick leave?",
            "conversation_history": [
                {"role": "user", "content": "What is the leave policy?"},
                {"role": "assistant", "content": "Employees receive 20 days leave."},
            ],
        }
        response = client.post("/api/query/process", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["is_conversational"] is True
        assert "sick leave" in data["rewritten_query"].lower()

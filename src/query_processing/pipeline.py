"""Unified QueryProcessor pipeline orchestrating rewriting, intent detection, and filter extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.query_processing.filter_extractor import FilterExtractor
from src.query_processing.intent import IntentDetector
from src.query_processing.models import ConversationTurn, ProcessedQuery, QueryIntent
from src.query_processing.rewriter import QueryRewriter


class QueryProcessor:
    """End-to-end Query Processing engine (Module 2).

    Pipeline stages:
        Raw Query -> Conversational Rewriting -> Intent Detection -> Filter Extraction -> Standard ProcessedQuery Object
    """

    def __init__(
        self,
        rewriter: Optional[QueryRewriter] = None,
        intent_detector: Optional[IntentDetector] = None,
        filter_extractor: Optional[FilterExtractor] = None,
    ) -> None:
        self.rewriter = rewriter or QueryRewriter()
        self.intent_detector = intent_detector or IntentDetector()
        self.filter_extractor = filter_extractor or FilterExtractor()

    def process(
        self,
        query: str,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> ProcessedQuery:
        """Process a raw user query through the complete query understanding pipeline."""
        raw_query = query.strip()

        # 1. Contextual Rewriting
        rewritten_query = self.rewriter.rewrite(raw_query, conversation_history)
        is_conversational = bool(conversation_history and rewritten_query != raw_query)

        # 2. Intent Detection
        intent: QueryIntent = self.intent_detector.detect(rewritten_query)

        # 3. Filter Extraction & Search Query Distillation
        search_query, filters = self.filter_extractor.extract(rewritten_query)

        # 4. Extract distinct keywords
        keywords = [w.lower() for w in search_query.split() if len(w) > 2]

        metadata: Dict[str, Any] = {
            "has_filters": bool(filters),
            "history_turns": len(conversation_history) if conversation_history else 0,
        }

        return ProcessedQuery(
            original_query=raw_query,
            rewritten_query=rewritten_query,
            query=search_query,
            intent=intent,
            filters=filters,
            is_conversational=is_conversational,
            extracted_keywords=keywords,
            metadata=metadata,
        )


default_query_processor = QueryProcessor()

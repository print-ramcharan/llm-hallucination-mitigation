"""Domain models and schemas for Module 2 Query Processing."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """Classification of user query intent."""

    INFORMATIONAL = "informational"
    FACTUAL = "factual"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    EXPLORATORY = "exploratory"
    KEYWORD_LOOKUP = "keyword_lookup"


class ConversationTurn(BaseModel):
    """Single turn in an interactive dialogue."""

    role: Literal["user", "assistant"]
    content: str


class ProcessedQuery(BaseModel):
    """Standardized representation of an understood, rewritten, and filtered query."""

    original_query: str = Field(..., description="Raw query directly entered by the user.")
    rewritten_query: str = Field(..., description="Contextually enriched query resolving anaphoras and ellipsis.")
    query: str = Field(..., description="Cleaned, distilled search query optimized for vector and BM25 indexing.")
    intent: QueryIntent = Field(QueryIntent.INFORMATIONAL, description="Detected intent of the query.")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Structured metadata filters (e.g. year, department).")
    is_conversational: bool = Field(False, description="Flag indicating if conversation history influenced this query.")
    extracted_keywords: List[str] = Field(default_factory=list, description="Key domain/entity tokens identified.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary debugging and trace info.")

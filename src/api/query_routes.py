"""FastAPI REST routes for Module 2: Query Processing."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.query_processing.models import ConversationTurn, ProcessedQuery
from src.query_processing.pipeline import default_query_processor

router = APIRouter(prefix="/api/query", tags=["Query Processing"])


class ProcessQueryRequest(BaseModel):
    """Payload for submitting a raw query to be understood, rewritten, and filtered."""

    query: str = Field(
        ...,
        description="Raw user query string",
        json_schema_extra={"example": "What did the 2024 engineering policy say about remote work?"},
    )
    conversation_history: Optional[List[ConversationTurn]] = Field(
        default=None,
        description="Optional list of prior conversation turns for coreference and ellipsis resolution",
    )


@router.post("/process", response_model=ProcessedQuery, status_code=status.HTTP_200_OK)
def process_query(payload: ProcessQueryRequest) -> ProcessedQuery:
    """Process a user query through query understanding, conversational rewriting,

    intent detection, and metadata filter extraction.
    """
    return default_query_processor.process(
        query=payload.query,
        conversation_history=payload.conversation_history,
    )

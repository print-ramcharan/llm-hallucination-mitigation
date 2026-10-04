"""Data models and schemas for Module 7: Cross-Session External Memory & Client Sync."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class MemoryPermissions(BaseModel):
    """Fine-grained Read/Write permission controls for cross-session external memory."""

    read_enabled: bool = Field(
        default=True,
        description="Whether past episodic memory and summaries can be injected into active queries.",
    )
    write_enabled: bool = Field(
        default=True,
        description="Whether completed interactions and verified facts can be written to external memory.",
    )
    auto_summarize: bool = Field(
        default=True,
        description="Whether to periodically condense completed turns into persistent hierarchical summaries.",
    )
    max_injected_memories: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of relevant past episodic memories to inject per query.",
    )
    max_injected_tokens: int = Field(
        default=250,
        ge=50,
        le=1000,
        description="Hard token budget allocated to memory context to avoid prompt bloating.",
    )


class Session(BaseModel):
    """Session representation tracking session lifecycle, user profile, and active documents."""

    session_id: str = Field(..., description="Unique UUID for this conversational session.")
    user_id: str = Field(default="default_user", description="Owner/user profile identifier.")
    title: str = Field(default="New Session", description="Human-readable title for the session.")
    created_at: str = Field(default_factory=_utc_now_iso, description="UTC creation timestamp.")
    updated_at: str = Field(default_factory=_utc_now_iso, description="UTC last update timestamp.")
    active_document_ids: list[str] = Field(
        default_factory=list,
        description="List of document IDs currently active or in-scope for this session.",
    )
    turn_count: int = Field(default=0, description="Total number of conversational turns in this session.")
    is_active: bool = Field(default=True, description="Whether the session is currently active or archived.")
    summary: str | None = Field(
        default=None,
        description="Latest persistent hierarchical summary of this session.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary session metadata (e.g. client type, tags, device).",
    )


class EpisodicMemoryItem(BaseModel):
    """An individual episodic memory record representing a past query-answer interaction."""

    memory_id: str = Field(..., description="Unique identifier for this memory item.")
    session_id: str = Field(..., description="Session identifier this interaction belongs to.")
    user_id: str = Field(default="default_user", description="User profile identifier.")
    turn_index: int = Field(default=0, description="Turn index within the session.")
    query: str = Field(..., description="Original user prompt or query.")
    intent: str | None = Field(default=None, description="Classified intent (e.g. FACTUAL, COMPARATIVE).")
    answer: str = Field(..., description="Grounded response or conclusion generated.")
    verified_facts: list[str] = Field(
        default_factory=list,
        description="Atomic factual propositions verified as entailed by evidence.",
    )
    referenced_doc_ids: list[str] = Field(
        default_factory=list,
        description="Document identifiers cited or referenced in this interaction.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Citation tags associated with this interaction (e.g. ['[Doc 1, Chunk 0]']).",
    )
    timestamp: str = Field(default_factory=_utc_now_iso, description="UTC timestamp of the interaction.")
    importance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Salience/importance weight for memory retrieval ranking.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorical tags (e.g. ['web_capture', 'qa_interaction', 'decision']).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata including client provenance or browser capture source.",
    )


class HierarchicalSummary(BaseModel):
    """Condensed multi-turn or cross-session persistent summary."""

    summary_id: str = Field(..., description="Unique identifier for the summary.")
    session_id: str = Field(..., description="Session identifier.")
    level: int = Field(
        default=1,
        description="Hierarchical level: 1 = session-level summary, 2 = multi-session topic summary.",
    )
    title: str = Field(..., description="Summary headline or topic.")
    summary_text: str = Field(..., description="Dense narrative summarizing key findings and decisions.")
    key_entities: list[str] = Field(
        default_factory=list,
        description="Key entities, concepts, or documents referenced in the summarized turns.",
    )
    turn_count: int = Field(default=0, description="Number of turns condensed in this summary.")
    created_at: str = Field(default_factory=_utc_now_iso, description="UTC timestamp when summary was created.")


class MemorySearchQuery(BaseModel):
    """Payload for performing semantic search over external episodic memory."""

    query: str = Field(..., description="Search query or current user prompt.")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID to constrain search to a single session.",
    )
    user_id: str | None = Field(
        default="default_user",
        description="User profile identifier.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of episodic memories to retrieve.",
    )
    min_similarity: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity cutoff threshold.",
    )
    include_summaries: bool = Field(
        default=True,
        description="Whether to search and return matching hierarchical summaries alongside turns.",
    )


class MemorySearchResult(BaseModel):
    """Results from semantic episodic memory search."""

    query: str = Field(..., description="Search query executed.")
    memories: list[EpisodicMemoryItem] = Field(
        default_factory=list,
        description="Top-k retrieved episodic memory items.",
    )
    summaries: list[HierarchicalSummary] = Field(
        default_factory=list,
        description="Relevant persistent summaries matching the query.",
    )
    scores: list[float] = Field(
        default_factory=list,
        description="Similarity scores corresponding to retrieved memories.",
    )
    total_found: int = Field(default=0, description="Total matching memory items found.")


class MemorySyncContext(BaseModel):
    """Compact context snippet injected before inference to mitigate cross-session context amnesia."""

    session_id: str | None = Field(default=None, description="Current session ID.")
    read_enabled: bool = Field(default=True, description="Whether memory reading was permitted.")
    injected_memories: list[EpisodicMemoryItem] = Field(
        default_factory=list,
        description="Episodic memories selected for injection.",
    )
    injected_summaries: list[HierarchicalSummary] = Field(
        default_factory=list,
        description="Hierarchical summaries selected for injection.",
    )
    formatted_memory_context: str = Field(
        default="",
        description="Strictly token-budgeted markdown snippet ready for prompt injection.",
    )
    estimated_tokens: int = Field(
        default=0,
        description="Estimated token count of the formatted memory context.",
    )


class WebContextCaptureRequest(BaseModel):
    """Payload sent by Browser Extension or Web Client to capture external browsing context."""

    url: str = Field(..., description="URL of the web page where content was captured.")
    title: str = Field(..., description="Title of the web page or article.")
    selected_text: str = Field(..., description="Highlighted text or excerpt captured by user.")
    full_content: str | None = Field(
        default=None,
        description="Optional broader article or section content.",
    )
    session_id: str | None = Field(
        default=None,
        description="Target session ID to store the captured web context into.",
    )
    user_id: str = Field(default="default_user", description="User profile identifier.")
    tags: list[str] = Field(
        default_factory=lambda: ["web_capture"],
        description="Categorical tags for indexing.",
    )


class MemoryExport(BaseModel):
    """Structured export payload for auditing, downloading, or transferring session memories."""

    session: Session = Field(..., description="Session metadata.")
    turns: list[EpisodicMemoryItem] = Field(
        default_factory=list,
        description="All episodic interaction turns.",
    )
    summaries: list[HierarchicalSummary] = Field(
        default_factory=list,
        description="All hierarchical persistent summaries.",
    )
    markdown_text: str = Field(
        default="",
        description="Full human-readable Markdown export of the session history.",
    )

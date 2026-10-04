"""FastAPI REST routes for Module 7: Cross-Session External Memory & Client Sync."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.memory.models import (
    EpisodicMemoryItem,
    HierarchicalSummary,
    MemoryPermissions,
    MemorySearchQuery,
    MemorySearchResult,
    MemorySyncContext,
    Session,
    WebContextCaptureRequest,
)
from src.memory.session_registry import default_session_registry
from src.memory.store import default_memory_store
from src.memory.summarizer import default_hierarchical_summarizer
from src.memory.synchronizer import default_memory_synchronizer

router = APIRouter(prefix="/api/memory", tags=["Cross-Session External Memory & Sync"])


class CreateSessionRequest(BaseModel):
    """Payload to create a new conversational session."""

    title: str = Field(default="New Session", description="Session title.")
    user_id: str = Field(default="default_user", description="User profile identifier.")
    active_document_ids: list[str] = Field(
        default_factory=list,
        description="Optional active document IDs to associate with this session.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Session metadata.")


class UpdateSessionRequest(BaseModel):
    """Payload to update session metadata."""

    title: str | None = Field(default=None, description="Updated session title.")
    active_document_ids: list[str] | None = Field(
        default=None, description="Updated list of active document IDs."
    )
    is_active: bool | None = Field(default=None, description="Active status.")
    metadata: dict[str, Any] | None = Field(default=None, description="Extra metadata.")


class SessionDetailsResponse(BaseModel):
    """Full session state including turns and hierarchical summaries."""

    session: Session
    permissions: MemoryPermissions
    turns: list[EpisodicMemoryItem]
    summaries: list[HierarchicalSummary]


class PostInferenceSyncRequest(BaseModel):
    """Payload to record an interaction turn into episodic memory."""

    session_id: str = Field(..., description="Target session ID.")
    query: str = Field(..., description="User query.")
    answer: str = Field(..., description="Generated answer.")
    intent: str | None = Field(default=None, description="Detected intent.")
    verified_facts: list[str] = Field(
        default_factory=list,
        description="Verified factual claims.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Citation tags referenced.",
    )
    referenced_doc_ids: list[str] = Field(
        default_factory=list,
        description="Referenced document IDs.",
    )
    tags: list[str] = Field(
        default_factory=lambda: ["qa_interaction"],
        description="Categorical tags.",
    )
    user_id: str = Field(default="default_user", description="User identifier.")


# =============================================================================
# Session Endpoints
# =============================================================================


@router.get(
    "/sessions",
    response_model=list[Session],
    status_code=status.HTTP_200_OK,
    summary="List all conversational sessions",
)
def list_sessions(
    user_id: str | None = Query(default=None, description="Filter by user ID"),
    is_active: bool | None = Query(default=None, description="Filter by active status"),
) -> list[Session]:
    """Retrieve all sessions ordered by most recently updated."""
    return default_session_registry.list_sessions(user_id=user_id, is_active=is_active)


@router.post(
    "/sessions",
    response_model=Session,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversational session",
)
def create_session(payload: CreateSessionRequest) -> Session:
    """Create a new conversational session with default permissions."""
    return default_session_registry.create_session(
        user_id=payload.user_id,
        title=payload.title,
        active_document_ids=payload.active_document_ids,
        metadata=payload.metadata,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get full session details, turns, and persistent summaries",
)
def get_session_details(session_id: str) -> SessionDetailsResponse:
    """Fetch session metadata, permissions, episodic turns, and summaries."""
    session, turns, summaries = default_session_registry.inspect_session_memory(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )
    permissions = default_session_registry.get_permissions(session_id)
    return SessionDetailsResponse(
        session=session,
        permissions=permissions,
        turns=turns,
        summaries=summaries,
    )


@router.patch(
    "/sessions/{session_id}",
    response_model=Session,
    status_code=status.HTTP_200_OK,
    summary="Update session title, active documents, or status",
)
def update_session(session_id: str, payload: UpdateSessionRequest) -> Session:
    """Update mutable attributes of a session."""
    updated = default_session_registry.update_session(
        session_id=session_id,
        title=payload.title,
        active_document_ids=payload.active_document_ids,
        is_active=payload.is_active,
        metadata=payload.metadata,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )
    return updated


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a session and all its associated memories",
)
def delete_session(session_id: str) -> dict[str, Any]:
    """Permanently delete a session, its permissions, and episodic memory entries."""
    deleted = default_session_registry.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )
    return {"status": "deleted", "session_id": session_id}


# =============================================================================
# Read / Write Permissions Endpoints
# =============================================================================


@router.get(
    "/sessions/{session_id}/permissions",
    response_model=MemoryPermissions,
    status_code=status.HTTP_200_OK,
    summary="Get read/write permissions for a session",
)
def get_session_permissions(session_id: str) -> MemoryPermissions:
    """Retrieve fine-grained memory permissions for a specific session."""
    session = default_session_registry.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )
    return default_session_registry.get_permissions(session_id)


@router.put(
    "/sessions/{session_id}/permissions",
    response_model=MemoryPermissions,
    status_code=status.HTTP_200_OK,
    summary="Update read/write permissions for a session",
)
def update_session_permissions(
    session_id: str, permissions: MemoryPermissions
) -> MemoryPermissions:
    """Update and persist read/write memory permissions for a session."""
    session = default_session_registry.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )
    return default_session_registry.set_permissions(session_id, permissions)


# =============================================================================
# Memory Clearing, Summarization, and Export
# =============================================================================


@router.post(
    "/sessions/{session_id}/clear",
    status_code=status.HTTP_200_OK,
    summary="Clear all episodic memories and summaries for a session",
)
def clear_session_memory(session_id: str) -> dict[str, Any]:
    """Wipe memories for a session without deleting the session itself."""
    session = default_session_registry.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )
    default_session_registry.clear_session_memory(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.post(
    "/sessions/{session_id}/summarize",
    response_model=HierarchicalSummary,
    status_code=status.HTTP_200_OK,
    summary="Force hierarchical summarization of session memories",
)
def force_session_summary(session_id: str) -> HierarchicalSummary:
    """Condense all interactions in the session into a fresh persistent summary."""
    session = default_session_registry.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )
    memories = default_memory_store.list_session_memories(session_id)
    summary = default_hierarchical_summarizer.summarize_session(session, memories)
    return default_memory_store.save_summary(summary)


@router.get(
    "/sessions/{session_id}/export",
    summary="Export session memories as structured JSON or Markdown",
)
def export_session_memory(
    session_id: str,
    format: str = Query(default="json", pattern=r"^(json|markdown)$"),
):
    """Export the session history, verified facts, and summaries."""
    export_data = default_session_registry.export_session_memory(session_id)
    if not export_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )

    if format == "markdown":
        return PlainTextResponse(export_data.markdown_text, media_type="text/markdown")
    return export_data


# =============================================================================
# Semantic Memory Search & Browser Client Capture
# =============================================================================


@router.post(
    "/search",
    response_model=MemorySearchResult,
    status_code=status.HTTP_200_OK,
    summary="Perform semantic vector search over external memory",
)
def search_memory(payload: MemorySearchQuery) -> MemorySearchResult:
    """Search past episodic interaction turns and summaries by semantic similarity."""
    memories, scores = default_memory_store.search_memories(
        query=payload.query,
        session_id=payload.session_id,
        user_id=payload.user_id,
        top_k=payload.top_k,
        min_similarity=payload.min_similarity,
    )

    summaries = []
    if payload.include_summaries:
        summaries, _ = default_memory_store.search_summaries(
            query=payload.query,
            session_id=payload.session_id,
            top_k=2,
            min_similarity=payload.min_similarity,
        )

    return MemorySearchResult(
        query=payload.query,
        memories=memories,
        summaries=summaries,
        scores=scores,
        total_found=len(memories),
    )


@router.post(
    "/capture-web",
    response_model=EpisodicMemoryItem,
    status_code=status.HTTP_201_CREATED,
    summary="Capture web page text and context from Browser Extension or Client",
)
def capture_web_context(payload: WebContextCaptureRequest) -> EpisodicMemoryItem:
    """Ingest highlighted web text, URL, and page title directly into session memory."""
    return default_memory_synchronizer.capture_web_context(payload)


@router.post(
    "/sync/pre",
    response_model=MemorySyncContext,
    status_code=status.HTTP_200_OK,
    summary="Pre-inference memory sync: inspect token-budgeted memory context injection",
)
def pre_inference_sync(
    query: str = Query(..., description="Current query"),
    session_id: str | None = Query(default=None, description="Session ID"),
    user_id: str = Query(default="default_user", description="User ID"),
    max_tokens: int | None = Query(default=None, description="Max token budget"),
) -> MemorySyncContext:
    """Preview or retrieve the compact memory block that will be injected before inference."""
    return default_memory_synchronizer.pre_inference_sync(
        query=query,
        session_id=session_id,
        user_id=user_id,
        max_tokens=max_tokens,
    )


@router.post(
    "/sync/post",
    response_model=EpisodicMemoryItem,
    status_code=status.HTTP_201_CREATED,
    summary="Post-inference memory sync: store interaction turn and verified facts",
)
def post_inference_sync(payload: PostInferenceSyncRequest) -> EpisodicMemoryItem:
    """Store verified interaction turn into external episodic memory."""
    item = default_memory_synchronizer.post_inference_sync(
        session_id=payload.session_id,
        query=payload.query,
        answer=payload.answer,
        intent=payload.intent,
        referenced_doc_ids=payload.referenced_doc_ids,
        tags=payload.tags,
        user_id=payload.user_id,
    )
    if not item:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Memory writing is disabled for this session.",
        )
    return item


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Get status and telemetry of external memory store",
)
def get_memory_status() -> dict[str, Any]:
    """Retrieve memory status, active session counts, and storage directories."""
    sessions = default_memory_store.list_sessions()
    active_sessions = [s for s in sessions if s.is_active]
    return {
        "status": "ready",
        "module": "Cross-Session External Memory & Client Sync",
        "storage_dir": str(default_memory_store.storage_dir),
        "database_file": str(default_memory_store.db_path),
        "total_sessions": len(sessions),
        "active_sessions": len(active_sessions),
        "indexed_memory_vectors": len(default_memory_store._memory_vectors),
        "indexed_summary_vectors": len(default_memory_store._summary_vectors),
    }

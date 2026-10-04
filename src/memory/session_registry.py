"""Session Registry & Memory Controller managing session lifecycles and fine-grained controls."""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from src.memory.models import (
    EpisodicMemoryItem,
    HierarchicalSummary,
    MemoryExport,
    MemoryPermissions,
    Session,
)
from src.memory.store import ExternalMemoryStore, default_memory_store


class SessionRegistry:
    """Manages session lifecycle, active working documents, and fine-grained read/write memory controls.

    Exposes governance APIs allowing users and clients to inspect, edit, clear, and export
    remembered external memory at both the session level and individual memory turn level.
    """

    def __init__(self, store: ExternalMemoryStore | None = None) -> None:
        self.store = store or default_memory_store

    def create_session(
        self,
        user_id: str = "default_user",
        title: str = "New Session",
        active_document_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new conversational session and initialize its permissions."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        session = Session(
            session_id=session_id,
            user_id=user_id,
            title=title.strip() or "New Session",
            created_at=now,
            updated_at=now,
            active_document_ids=active_document_ids or [],
            turn_count=0,
            is_active=True,
            summary=None,
            metadata=metadata or {},
        )
        self.store.save_session(session)

        # Inherit or initialize default permissions for this session
        default_perms = self.store.get_permissions("global")
        self.store.save_permissions(session_id, default_perms)

        return session

    def get_session(self, session_id: str) -> Session | None:
        """Fetch session metadata by session ID."""
        return self.store.get_session(session_id)

    def get_or_create_default_session(self, user_id: str = "default_user") -> Session:
        """Get an existing active session or automatically create a default one."""
        active_sessions = self.store.list_sessions(user_id=user_id, is_active=True)
        if active_sessions:
            return active_sessions[0]
        return self.create_session(
            user_id=user_id,
            title="Default Workspace Session",
            metadata={"auto_created": True},
        )

    def list_sessions(
        self, user_id: str | None = None, is_active: bool | None = None
    ) -> list[Session]:
        """List sessions filtered by user or activity state."""
        return self.store.list_sessions(user_id=user_id, is_active=is_active)

    def update_session(
        self,
        session_id: str,
        title: str | None = None,
        active_document_ids: list[str] | None = None,
        is_active: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session | None:
        """Update mutable fields of a session."""
        session = self.store.get_session(session_id)
        if not session:
            return None

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if title is not None:
            session.title = title.strip() or session.title
        if active_document_ids is not None:
            session.active_document_ids = active_document_ids
        if is_active is not None:
            session.is_active = is_active
        if metadata is not None:
            session.metadata.update(metadata)
        session.updated_at = now

        return self.store.save_session(session)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session, its permissions, and all associated memories."""
        return self.store.delete_session(session_id)

    # =========================================================================
    # Read / Write Permission Governance
    # =========================================================================

    def get_permissions(self, session_id: str = "global") -> MemoryPermissions:
        """Fetch read/write permissions for a specific session or fallback to global."""
        return self.store.get_permissions(session_id)

    def set_permissions(
        self, session_id: str, permissions: MemoryPermissions
    ) -> MemoryPermissions:
        """Update and persist read/write controls for a session."""
        return self.store.save_permissions(session_id, permissions)

    # =========================================================================
    # Inspection, Clearing & Export
    # =========================================================================

    def inspect_session_memory(
        self, session_id: str
    ) -> tuple[Session | None, list[EpisodicMemoryItem], list[HierarchicalSummary]]:
        """Return full session state, chronological interaction turns, and summaries."""
        session = self.store.get_session(session_id)
        turns = self.store.list_session_memories(session_id)
        summaries = self.store.list_session_summaries(session_id)
        return session, turns, summaries

    def clear_session_memory(self, session_id: str) -> bool:
        """Clear all remembered turns and summaries for a session without deleting the session."""
        cleared = self.store.clear_session_memories(session_id)
        return cleared >= 0

    def export_session_memory(self, session_id: str) -> MemoryExport | None:
        """Generate structured and human-readable Markdown export of session memory."""
        session = self.store.get_session(session_id)
        if not session:
            return None

        turns = self.store.list_session_memories(session_id)
        summaries = self.store.list_session_summaries(session_id)

        # Generate Markdown export
        md_lines = [
            f"# Session Memory Export: {session.title}",
            f"- **Session ID:** `{session.session_id}`",
            f"- **User:** `{session.user_id}`",
            f"- **Created:** {session.created_at}",
            f"- **Total Turns:** {session.turn_count}",
            f"- **Active Documents:** {', '.join(session.active_document_ids) if session.active_document_ids else 'None'}",
            "",
            "## Persistent Summaries",
        ]

        if summaries:
            for s in summaries:
                md_lines.append(f"### {s.title} ({s.created_at})")
                md_lines.append(s.summary_text)
                if s.key_entities:
                    md_lines.append(f"*Key Entities:* {', '.join(s.key_entities)}")
                md_lines.append("")
        else:
            md_lines.append("_No persistent summaries available for this session._\n")

        md_lines.append("## Chronological Episodic Interaction Turns")
        if turns:
            for t in turns:
                md_lines.append(f"### Turn {t.turn_index + 1}: {t.query} [{t.timestamp}]")
                if t.intent:
                    md_lines.append(f"- **Intent:** `{t.intent}`")
                md_lines.append(f"- **Answer:** {t.answer}")
                if t.verified_facts:
                    md_lines.append("- **Verified Facts:**")
                    for f in t.verified_facts:
                        md_lines.append(f"  - {f}")
                if t.citations:
                    md_lines.append(f"- **Citations:** {', '.join(t.citations)}")
                md_lines.append("")
        else:
            md_lines.append("_No interaction turns recorded._")

        return MemoryExport(
            session=session,
            turns=turns,
            summaries=summaries,
            markdown_text="\n".join(md_lines),
        )


default_session_registry = SessionRegistry()

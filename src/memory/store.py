"""External Conversation Vector Store & SQLite Persistence Engine for Module 7."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from src.indexing.embeddings import EmbeddingEngine, default_embedding_engine
from src.memory.models import (
    EpisodicMemoryItem,
    HierarchicalSummary,
    MemoryPermissions,
    Session,
)


class ExternalMemoryStore:
    """Hybrid SQLite + Vector Index persistence store for cross-session episodic memory.

    Stores structured session metadata, fine-grained permissions, and conversation turns
    in an ACID SQLite database, while simultaneously indexing semantic embeddings of
    past queries, answers, verified facts, and summaries in FAISS for sub-millisecond retrieval.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        storage_dir: Path | str | None = None,
        embedding_engine: EmbeddingEngine | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/memory")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = Path(db_path) if db_path else self.storage_dir / "memory.db"
        self.embedding_engine = embedding_engine or default_embedding_engine

        # Vector memory storage structures
        self.vector_index_path = self.storage_dir / "memory_vector.npy"
        self.vector_map_path = self.storage_dir / "memory_vector_map.json"

        # In-memory vector cache: memory_id -> embedding vector
        self._memory_vectors: dict[str, np.ndarray] = {}
        self._summary_vectors: dict[str, np.ndarray] = {}

        # Initialize SQLite schema
        self._init_db()
        self._load_vector_cache()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection to SQLite with WAL mode enabled for concurrent read safety."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Initialize database tables for sessions, permissions, memories, and summaries."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active_document_ids_json TEXT NOT NULL,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    summary TEXT,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS permissions (
                    entity_id TEXT PRIMARY KEY,
                    read_enabled INTEGER NOT NULL DEFAULT 1,
                    write_enabled INTEGER NOT NULL DEFAULT 1,
                    auto_summarize INTEGER NOT NULL DEFAULT 1,
                    max_injected_memories INTEGER NOT NULL DEFAULT 3,
                    max_injected_tokens INTEGER NOT NULL DEFAULT 250
                );

                CREATE TABLE IF NOT EXISTS episodic_memories (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT,
                    answer TEXT NOT NULL,
                    verified_facts_json TEXT NOT NULL,
                    referenced_doc_ids_json TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance_score REAL NOT NULL DEFAULT 1.0,
                    tags_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS hierarchical_summaries (
                    summary_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    level INTEGER NOT NULL DEFAULT 1,
                    title TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    key_entities_json TEXT NOT NULL,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_memories_session ON episodic_memories(session_id);
                CREATE INDEX IF NOT EXISTS idx_memories_user ON episodic_memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_summaries_session ON hierarchical_summaries(session_id);
                """
            )
            conn.commit()

    def _load_vector_cache(self) -> None:
        """Load persisted vector embeddings from disk if available."""
        if self.vector_index_path.exists() and self.vector_map_path.exists():
            try:
                vectors = np.load(str(self.vector_index_path))
                with open(self.vector_map_path, encoding="utf-8") as f:
                    meta = json.load(f)
                memory_ids = meta.get("memory_ids", [])
                summary_ids = meta.get("summary_ids", [])

                split_point = len(memory_ids)
                if len(vectors) == split_point + len(summary_ids):
                    for idx, mid in enumerate(memory_ids):
                        self._memory_vectors[mid] = vectors[idx]
                    for idx, sid in enumerate(summary_ids):
                        self._summary_vectors[sid] = vectors[split_point + idx]
            except Exception:
                # If cache is corrupt, it can be re-indexed from SQLite
                self._memory_vectors.clear()
                self._summary_vectors.clear()

    def _save_vector_cache(self) -> None:
        """Save vector cache and ID mapping to disk."""
        try:
            mem_items = list(self._memory_vectors.items())
            sum_items = list(self._summary_vectors.items())

            all_vecs = [v for _, v in mem_items] + [v for _, v in sum_items]
            if all_vecs:
                stacked = np.vstack(all_vecs).astype(np.float32)
                np.save(str(self.vector_index_path), stacked)
            elif self.vector_index_path.exists():
                self.vector_index_path.unlink()

            meta = {
                "memory_ids": [k for k, _ in mem_items],
                "summary_ids": [k for k, _ in sum_items],
            }
            with open(self.vector_map_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    # =========================================================================
    # Session Operations
    # =========================================================================

    def save_session(self, session: Session) -> Session:
        """Upsert a session record in SQLite."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, user_id, title, created_at, updated_at,
                    active_document_ids_json, turn_count, is_active, summary, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    title=excluded.title,
                    updated_at=excluded.updated_at,
                    active_document_ids_json=excluded.active_document_ids_json,
                    turn_count=excluded.turn_count,
                    is_active=excluded.is_active,
                    summary=excluded.summary,
                    metadata_json=excluded.metadata_json
                """,
                (
                    session.session_id,
                    session.user_id,
                    session.title,
                    session.created_at,
                    session.updated_at,
                    json.dumps(session.active_document_ids),
                    session.turn_count,
                    1 if session.is_active else 0,
                    session.summary,
                    json.dumps(session.metadata),
                ),
            )
            conn.commit()
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return Session(
                session_id=row["session_id"],
                user_id=row["user_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                active_document_ids=json.loads(row["active_document_ids_json"]),
                turn_count=row["turn_count"],
                is_active=bool(row["is_active"]),
                summary=row["summary"],
                metadata=json.loads(row["metadata_json"]),
            )

    def list_sessions(
        self, user_id: str | None = None, is_active: bool | None = None
    ) -> list[Session]:
        """List sessions optionally filtered by user ID and active status."""
        query = "SELECT * FROM sessions WHERE 1=1"
        params: list[Any] = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if is_active is not None:
            query += " AND is_active = ?"
            params.append(1 if is_active else 0)
        query += " ORDER BY updated_at DESC"

        sessions: list[Session] = []
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                sessions.append(
                    Session(
                        session_id=row["session_id"],
                        user_id=row["user_id"],
                        title=row["title"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        active_document_ids=json.loads(row["active_document_ids_json"]),
                        turn_count=row["turn_count"],
                        is_active=bool(row["is_active"]),
                        summary=row["summary"],
                        metadata=json.loads(row["metadata_json"]),
                    )
                )
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session, its permissions, episodic turns, and summaries."""
        # Clean vector cache
        mems = self.list_session_memories(session_id)
        for m in mems:
            self._memory_vectors.pop(m.memory_id, None)
        sums = self.list_session_summaries(session_id)
        for s in sums:
            self._summary_vectors.pop(s.summary_id, None)
        self._save_vector_cache()

        with self._get_connection() as conn:
            conn.execute("DELETE FROM permissions WHERE entity_id = ?", (session_id,))
            conn.execute("DELETE FROM hierarchical_summaries WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM episodic_memories WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Permissions Operations
    # =========================================================================

    def get_permissions(self, entity_id: str = "global") -> MemoryPermissions:
        """Get read/write permissions for a session or global default."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM permissions WHERE entity_id = ?", (entity_id,)
            )
            row = cursor.fetchone()
            if not row:
                # Return default permissions
                return MemoryPermissions()
            return MemoryPermissions(
                read_enabled=bool(row["read_enabled"]),
                write_enabled=bool(row["write_enabled"]),
                auto_summarize=bool(row["auto_summarize"]),
                max_injected_memories=row["max_injected_memories"],
                max_injected_tokens=row["max_injected_tokens"],
            )

    def save_permissions(
        self, entity_id: str, permissions: MemoryPermissions
    ) -> MemoryPermissions:
        """Persist read/write permissions for a session or global default."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO permissions (
                    entity_id, read_enabled, write_enabled, auto_summarize,
                    max_injected_memories, max_injected_tokens
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    read_enabled=excluded.read_enabled,
                    write_enabled=excluded.write_enabled,
                    auto_summarize=excluded.auto_summarize,
                    max_injected_memories=excluded.max_injected_memories,
                    max_injected_tokens=excluded.max_injected_tokens
                """,
                (
                    entity_id,
                    1 if permissions.read_enabled else 0,
                    1 if permissions.write_enabled else 0,
                    1 if permissions.auto_summarize else 0,
                    permissions.max_injected_memories,
                    permissions.max_injected_tokens,
                ),
            )
            conn.commit()
        return permissions

    # =========================================================================
    # Episodic Memory Operations
    # =========================================================================

    def _build_memory_embedding_text(self, item: EpisodicMemoryItem) -> str:
        """Construct semantic text passage to represent memory item in vector space."""
        parts = [f"Query: {item.query}"]
        if item.intent:
            parts.append(f"Intent: {item.intent}")
        if item.verified_facts:
            facts = " ".join(item.verified_facts[:5])
            parts.append(f"Verified Facts: {facts}")
        parts.append(f"Answer: {item.answer[:300]}")
        return " | ".join(parts)

    def save_memory_item(self, item: EpisodicMemoryItem) -> EpisodicMemoryItem:
        """Store an episodic memory turn in SQLite and index its dense vector."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO episodic_memories (
                    memory_id, session_id, user_id, turn_index, query, intent,
                    answer, verified_facts_json, referenced_doc_ids_json, citations_json,
                    timestamp, importance_score, tags_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    query=excluded.query,
                    intent=excluded.intent,
                    answer=excluded.answer,
                    verified_facts_json=excluded.verified_facts_json,
                    referenced_doc_ids_json=excluded.referenced_doc_ids_json,
                    citations_json=excluded.citations_json,
                    importance_score=excluded.importance_score,
                    tags_json=excluded.tags_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    item.memory_id,
                    item.session_id,
                    item.user_id,
                    item.turn_index,
                    item.query,
                    item.intent,
                    item.answer,
                    json.dumps(item.verified_facts),
                    json.dumps(item.referenced_doc_ids),
                    json.dumps(item.citations),
                    item.timestamp,
                    item.importance_score,
                    json.dumps(item.tags),
                    json.dumps(item.metadata),
                ),
            )
            # Update session turn_count & updated_at
            conn.execute(
                "UPDATE sessions SET turn_count = turn_count + 1, updated_at = ? WHERE session_id = ?",
                (item.timestamp, item.session_id),
            )
            conn.commit()

        # Compute and index vector embedding
        text_to_embed = self._build_memory_embedding_text(item)
        try:
            vec = self.embedding_engine.embed_query(text_to_embed)
            self._memory_vectors[item.memory_id] = vec
            self._save_vector_cache()
        except Exception:
            pass

        return item

    def get_memory_item(self, memory_id: str) -> EpisodicMemoryItem | None:
        """Retrieve an episodic memory item by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM episodic_memories WHERE memory_id = ?", (memory_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_memory_item(row)

    def list_session_memories(self, session_id: str) -> list[EpisodicMemoryItem]:
        """List all episodic memories for a given session sorted chronologically."""
        memories: list[EpisodicMemoryItem] = []
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM episodic_memories WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,),
            )
            for row in cursor.fetchall():
                memories.append(self._row_to_memory_item(row))
        return memories

    def delete_memory_item(self, memory_id: str) -> bool:
        """Delete an individual memory item and remove it from vector cache."""
        self._memory_vectors.pop(memory_id, None)
        self._save_vector_cache()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM episodic_memories WHERE memory_id = ?", (memory_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_session_memories(self, session_id: str) -> int:
        """Clear all episodic memories and summaries for a specific session."""
        mems = self.list_session_memories(session_id)
        for m in mems:
            self._memory_vectors.pop(m.memory_id, None)
        sums = self.list_session_summaries(session_id)
        for s in sums:
            self._summary_vectors.pop(s.summary_id, None)
        self._save_vector_cache()

        with self._get_connection() as conn:
            conn.execute("DELETE FROM hierarchical_summaries WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM episodic_memories WHERE session_id = ?", (session_id,))
            conn.execute("UPDATE sessions SET turn_count = 0, summary = NULL WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount

    def _row_to_memory_item(self, row: sqlite3.Row) -> EpisodicMemoryItem:
        """Convert SQLite row to EpisodicMemoryItem model."""
        return EpisodicMemoryItem(
            memory_id=row["memory_id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            turn_index=row["turn_index"],
            query=row["query"],
            intent=row["intent"],
            answer=row["answer"],
            verified_facts=json.loads(row["verified_facts_json"]),
            referenced_doc_ids=json.loads(row["referenced_doc_ids_json"]),
            citations=json.loads(row["citations_json"]),
            timestamp=row["timestamp"],
            importance_score=row["importance_score"],
            tags=json.loads(row["tags_json"]),
            metadata=json.loads(row["metadata_json"]),
        )

    # =========================================================================
    # Hierarchical Summary Operations
    # =========================================================================

    def save_summary(self, summary: HierarchicalSummary) -> HierarchicalSummary:
        """Persist hierarchical summary in SQLite and index its dense vector."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO hierarchical_summaries (
                    summary_id, session_id, level, title, summary_text,
                    key_entities_json, turn_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(summary_id) DO UPDATE SET
                    title=excluded.title,
                    summary_text=excluded.summary_text,
                    key_entities_json=excluded.key_entities_json,
                    turn_count=excluded.turn_count
                """,
                (
                    summary.summary_id,
                    summary.session_id,
                    summary.level,
                    summary.title,
                    summary.summary_text,
                    json.dumps(summary.key_entities),
                    summary.turn_count,
                    summary.created_at,
                ),
            )
            # Update latest summary in session
            conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (summary.summary_text, summary.created_at, summary.session_id),
            )
            conn.commit()

        # Vector indexing for summary
        try:
            vec = self.embedding_engine.embed_query(
                f"Summary: {summary.title} | {summary.summary_text}"
            )
            self._summary_vectors[summary.summary_id] = vec
            self._save_vector_cache()
        except Exception:
            pass

        return summary

    def list_session_summaries(self, session_id: str) -> list[HierarchicalSummary]:
        """Retrieve all hierarchical summaries for a session."""
        summaries: list[HierarchicalSummary] = []
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM hierarchical_summaries WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            )
            for row in cursor.fetchall():
                summaries.append(
                    HierarchicalSummary(
                        summary_id=row["summary_id"],
                        session_id=row["session_id"],
                        level=row["level"],
                        title=row["title"],
                        summary_text=row["summary_text"],
                        key_entities=json.loads(row["key_entities_json"]),
                        turn_count=row["turn_count"],
                        created_at=row["created_at"],
                    )
                )
        return summaries

    # =========================================================================
    # Semantic Search Operations
    # =========================================================================

    def search_memories(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 3,
        min_similarity: float = 0.25,
    ) -> tuple[list[EpisodicMemoryItem], list[float]]:
        """Perform semantic vector similarity search over past episodic memories.

        Falls back gracefully to keyword/token-overlap matching if vector search is unavailable.
        """
        if not query.strip():
            return [], []

        # Candidate pool filtered by session / user from SQLite
        all_candidates = []
        with self._get_connection() as conn:
            sql = "SELECT * FROM episodic_memories WHERE 1=1"
            params: list[Any] = []
            if session_id:
                sql += " AND session_id = ?"
                params.append(session_id)
            if user_id:
                sql += " AND user_id = ?"
                params.append(user_id)
            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                all_candidates.append(self._row_to_memory_item(row))

        if not all_candidates:
            return [], []

        # 1. Try dense vector search using inner-product (cosine similarity)
        scored_candidates: list[tuple[EpisodicMemoryItem, float]] = []
        try:
            query_vec = self.embedding_engine.embed_query(query)
            for item in all_candidates:
                if item.memory_id in self._memory_vectors:
                    vec = self._memory_vectors[item.memory_id]
                    # Since vectors are L2-normalized, dot product is cosine similarity
                    sim = float(np.dot(query_vec, vec))
                else:
                    # Token overlap fallback
                    sim = self._compute_token_overlap(query, item.query + " " + item.answer)
                # Scale by importance score
                effective_score = sim * item.importance_score
                if effective_score >= min_similarity:
                    scored_candidates.append((item, effective_score))
        except Exception:
            # Fallback to lexical token overlap
            for item in all_candidates:
                sim = self._compute_token_overlap(query, item.query + " " + item.answer)
                if sim >= min_similarity:
                    scored_candidates.append((item, sim))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored_candidates[:top_k]

        return [item for item, _ in top_matches], [score for _, score in top_matches]

    def search_summaries(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 2,
        min_similarity: float = 0.25,
    ) -> tuple[list[HierarchicalSummary], list[float]]:
        """Search persistent summaries matching query semantics."""
        all_summaries = []
        with self._get_connection() as conn:
            sql = "SELECT * FROM hierarchical_summaries WHERE 1=1"
            params: list[Any] = []
            if session_id:
                sql += " AND session_id = ?"
                params.append(session_id)
            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                all_summaries.append(
                    HierarchicalSummary(
                        summary_id=row["summary_id"],
                        session_id=row["session_id"],
                        level=row["level"],
                        title=row["title"],
                        summary_text=row["summary_text"],
                        key_entities=json.loads(row["key_entities_json"]),
                        turn_count=row["turn_count"],
                        created_at=row["created_at"],
                    )
                )

        if not all_summaries:
            return [], []

        scored: list[tuple[HierarchicalSummary, float]] = []
        try:
            query_vec = self.embedding_engine.embed_query(query)
            for s in all_summaries:
                if s.summary_id in self._summary_vectors:
                    vec = self._summary_vectors[s.summary_id]
                    sim = float(np.dot(query_vec, vec))
                else:
                    sim = self._compute_token_overlap(query, s.title + " " + s.summary_text)
                if sim >= min_similarity:
                    scored.append((s, sim))
        except Exception:
            for s in all_summaries:
                sim = self._compute_token_overlap(query, s.title + " " + s.summary_text)
                if sim >= min_similarity:
                    scored.append((s, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        return [s for s, _ in top], [sc for _, sc in top]

    @staticmethod
    def _compute_token_overlap(str1: str, str2: str) -> float:
        """Token-level Jaccard similarity fallback."""
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / float(union) if union > 0 else 0.0

    def clear_all(self) -> None:
        """Wipe all session and memory data (useful for test isolation)."""
        self._memory_vectors.clear()
        self._summary_vectors.clear()
        if self.vector_index_path.exists():
            self.vector_index_path.unlink()
        if self.vector_map_path.exists():
            self.vector_map_path.unlink()
        with self._get_connection() as conn:
            conn.execute("DELETE FROM permissions")
            conn.execute("DELETE FROM hierarchical_summaries")
            conn.execute("DELETE FROM episodic_memories")
            conn.execute("DELETE FROM sessions")
            conn.commit()


default_memory_store = ExternalMemoryStore()

"""Metadata store for persistent chunk payloads and metadata tag references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ingestion.models import DocumentChunk


class MetadataStore:
    """Stores full chunk content and structured metadata tags mapped by chunk_id."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/indexes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.storage_dir / "metadata_store.json"

        # chunk_id -> Chunk Record (content + metadata)
        self._store: dict[str, dict[str, Any]] = {}
        # doc_id -> list of chunk_ids
        self._doc_to_chunks: dict[str, list[str]] = {}

        self.load()

    @property
    def total_chunks(self) -> int:
        """Total chunks stored."""
        return len(self._store)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Store chunk records with their content and standardized metadata."""
        if not chunks:
            return

        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id", chunk.id) if chunk.metadata else chunk.id
            doc_id = chunk.document_id

            record = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "token_count": chunk.token_count or chunk.metadata.get("token_count", 0),
                "metadata": chunk.metadata,
            }

            self._store[chunk_id] = record
            self._doc_to_chunks.setdefault(doc_id, []).append(chunk_id)

        self.save()

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a chunk record by chunk_id."""
        return self._store.get(chunk_id)

    def get_by_document(self, doc_id: str) -> list[dict[str, Any]]:
        """Retrieve all chunk records belonging to a document."""
        chunk_ids = self._doc_to_chunks.get(doc_id, [])
        return [self._store[cid] for cid in chunk_ids if cid in self._store]

    def remove_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document."""
        chunk_ids = self._doc_to_chunks.pop(doc_id, [])
        for cid in chunk_ids:
            self._store.pop(cid, None)
        self.save()

    def save(self) -> None:
        """Persist metadata store to JSON."""
        try:
            payload = {
                "store": self._store,
                "doc_to_chunks": self._doc_to_chunks,
            }
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            print(f"Warning: Failed to save metadata store: {exc}")

    def load(self) -> None:
        """Load metadata store from JSON."""
        if self.store_path.is_file():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    self._store = payload.get("store", {})
                    self._doc_to_chunks = payload.get("doc_to_chunks", {})
            except Exception as exc:
                print(f"Warning: Failed to load metadata store: {exc}")
                self._store = {}
                self._doc_to_chunks = {}

    def clear(self) -> None:
        """Clear all stored metadata."""
        self._store = {}
        self._doc_to_chunks = {}
        if self.store_path.is_file():
            self.store_path.unlink()

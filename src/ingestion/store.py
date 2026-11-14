"""Document storage and repository for ingested documents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.ingestion.models import Document, DocumentMetadata


class DocumentStore:
    """Stores and manages ingested Document objects in memory and on disk."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        if storage_dir is None:
            storage_dir = Path("data") / "documents"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Document] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load persisted documents from disk into cache."""
        for file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                doc = Document.model_validate(data)
                self._cache[doc.id] = doc
            except Exception:
                continue

    def add(self, document: Document, owner_id: str | None = None) -> Document:
        """Save a document into memory and disk."""
        if owner_id:
            self.assign_owner(document, owner_id)
        self._cache[document.id] = document
        file_path = self.storage_dir / f"{document.id}.json"
        file_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return document

    @staticmethod
    def assign_owner(document: Document, owner_id: str) -> None:
        """Apply a stable owner namespace to the document and all indexed chunks."""
        # Document IDs are content-derived; include the owner namespace so two users
        # uploading identical bytes cannot overwrite each other's index records.
        scoped_id = f"doc_{hashlib.sha256((owner_id + ':' + document.metadata.content_hash).encode()).hexdigest()[:16]}"
        document.id = scoped_id
        document.metadata.document_id = scoped_id
        document.metadata.extra["owner_id"] = owner_id
        for idx, chunk in enumerate(document.chunks):
            chunk.id = f"{scoped_id}_chunk_{idx}"
            chunk.document_id = scoped_id
            chunk.metadata["chunk_id"] = chunk.id
            chunk.metadata["owner_id"] = owner_id
            chunk.metadata["prev_chunk_id"] = (
                f"{scoped_id}_chunk_{idx - 1}" if idx else None
            )
            chunk.metadata["next_chunk_id"] = (
                f"{scoped_id}_chunk_{idx + 1}"
                if idx + 1 < len(document.chunks)
                else None
            )
        for idx, element in enumerate(document.elements):
            element.id = f"{scoped_id}_el_{idx}"

    def get(self, doc_id: str, owner_id: str | None = None) -> Document | None:
        """Retrieve a document by ID."""
        document = self._cache.get(doc_id)
        if (
            document
            and owner_id
            and document.metadata.extra.get("owner_id") != owner_id
        ):
            return None
        return document

    def list_all(self, owner_id: str | None = None) -> list[DocumentMetadata]:
        """Return metadata for all stored documents, ordered by creation time descending."""
        docs = [
            d
            for d in self._cache.values()
            if owner_id is None or d.metadata.extra.get("owner_id") == owner_id
        ]
        docs.sort(key=lambda d: d.metadata.created_at, reverse=True)
        return [d.metadata for d in docs]

    def list_all_documents(self, owner_id: str | None = None) -> list[Document]:
        """Return full document objects, ordered by creation time descending."""
        docs = [
            d
            for d in self._cache.values()
            if owner_id is None or d.metadata.extra.get("owner_id") == owner_id
        ]
        docs.sort(key=lambda d: d.metadata.created_at, reverse=True)
        return docs

    def delete(self, doc_id: str, owner_id: str | None = None) -> bool:
        """Delete a document by ID."""
        if self.get(doc_id, owner_id):
            del self._cache[doc_id]
            file_path = self.storage_dir / f"{doc_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def clear(self) -> None:
        """Clear all stored documents."""
        self._cache.clear()
        for file in self.storage_dir.glob("*.json"):
            file.unlink()


# Global default store
default_store = DocumentStore()

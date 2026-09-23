"""Document storage and repository for ingested documents."""

from __future__ import annotations

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

    def add(self, document: Document) -> Document:
        """Save a document into memory and disk."""
        self._cache[document.id] = document
        file_path = self.storage_dir / f"{document.id}.json"
        file_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return document

    def get(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID."""
        return self._cache.get(doc_id)

    def list_all(self) -> list[DocumentMetadata]:
        """Return metadata for all stored documents, ordered by creation time descending."""
        docs = list(self._cache.values())
        docs.sort(key=lambda d: d.metadata.created_at, reverse=True)
        return [d.metadata for d in docs]

    def list_all_documents(self) -> list[Document]:
        """Return full document objects, ordered by creation time descending."""
        docs = list(self._cache.values())
        docs.sort(key=lambda d: d.metadata.created_at, reverse=True)
        return docs

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if doc_id in self._cache:
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

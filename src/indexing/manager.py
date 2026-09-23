"""Unified Index Manager for coordinating Vector Index, BM25 Index, and Metadata Store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.indexing.bm25_index import BM25Index
from src.indexing.embeddings import EmbeddingEngine, default_embedding_engine
from src.indexing.metadata_store import MetadataStore
from src.indexing.vector_index import FaissVectorIndex
from src.ingestion.models import Document


class IndexManager:
    """Coordinates dense vector indexing, BM25 sparse lexical indexing, and metadata storage.

    Fulfills Module 1 requirements:
      - Sentence-Transformers L2-normalized embeddings
      - FAISS Vector Index (cosine similarity via inner product)
      - BM25 Index for Module 3 hybrid retrieval
      - Metadata Store (chunk + embedding ref + metadata)
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        embedding_engine: EmbeddingEngine | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/indexes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_engine = embedding_engine or default_embedding_engine
        self.vector_index = FaissVectorIndex(
            dimension=self.embedding_engine.dimension,
            storage_dir=self.storage_dir,
        )
        self.bm25_index = BM25Index(storage_dir=self.storage_dir)
        self.metadata_store = MetadataStore(storage_dir=self.storage_dir)

    def index_document(self, document: Document) -> dict[str, Any]:
        """Index a document's chunks across Vector Index, BM25 Index, and Metadata Store."""
        chunks = document.chunks
        if not chunks:
            return {
                "document_id": document.id,
                "indexed_chunks": 0,
                "status": "No chunks to index",
            }

        # 1. Generate L2-normalized embeddings using sentence-transformers
        embeddings = self.embedding_engine.embed_chunks(chunks)

        # 2. Add to FAISS Vector Index
        self.vector_index.add_chunks(chunks, embeddings)

        # 3. Add to BM25 Sparse Index (ready for Module 3 hybrid retrieval)
        self.bm25_index.add_chunks(chunks)

        # 4. Add to Metadata Store
        self.metadata_store.add_chunks(chunks)

        return {
            "document_id": document.id,
            "indexed_chunks": len(chunks),
            "vector_index_total": self.vector_index.total_vectors,
            "bm25_index_total": self.bm25_index.total_documents,
            "metadata_store_total": self.metadata_store.total_chunks,
            "embedding_model": self.embedding_engine.model_name,
            "dimension": self.embedding_engine.dimension,
            "status": "indexed_successfully",
        }

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from all indexes and metadata store."""
        self.vector_index.remove_document(doc_id)
        self.bm25_index.remove_document(doc_id)
        self.metadata_store.remove_document(doc_id)

    def get_stats(self) -> dict[str, Any]:
        """Retrieve telemetry and counts for all indexes."""
        return {
            "vector_chunks": self.vector_index.total_vectors,
            "bm25_documents": self.bm25_index.total_documents,
            "metadata_chunks": self.metadata_store.total_chunks,
            "embedding_model": self.embedding_engine.model_name,
            "embedding_dimension": self.embedding_engine.dimension,
            "storage_directory": str(self.storage_dir),
        }

    def clear(self) -> None:
        """Wipe all indexes and data store."""
        self.vector_index.clear()
        self.bm25_index.clear()
        self.metadata_store.clear()


default_index_manager = IndexManager()

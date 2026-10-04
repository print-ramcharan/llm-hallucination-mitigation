"""Sparse Lexical Retriever using BM25+ inverted index."""

from __future__ import annotations

from typing import Any

from src.indexing.bm25_index import BM25Index
from src.indexing.metadata_store import MetadataStore


class SparseRetriever:
    """Retrieves candidates using BM25 lexical token matching.

    Ensures exact keywords, terms, acronyms, and product codes are found even when
    semantic embeddings dilute specific terminology.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        metadata_store: MetadataStore,
    ) -> None:
        self.bm25_index = bm25_index
        self.metadata_store = metadata_store

    def retrieve(
        self,
        query: str,
        top_k: int = 25,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Run sparse BM25 search for the top_k most relevant chunks.

        Args:
            query: Keyword query string.
            top_k: Number of candidates to retrieve.
            filters: Optional metadata filters.

        Returns:
            List of (chunk_id, bm25_score) ordered descending by lexical relevance.
        """
        if not query.strip() or self.bm25_index.total_documents == 0:
            return []

        oversample_k = (
            min(max(top_k * 4, 50), self.bm25_index.total_documents)
            if filters
            else top_k
        )
        raw_results = self.bm25_index.search(query, top_k=oversample_k)

        if not filters:
            return raw_results[:top_k]

        filtered_results: list[tuple[str, float]] = []
        for chunk_id, score in raw_results:
            record = self.metadata_store.get(chunk_id)
            if record and self._matches_filters(record, filters):
                filtered_results.append((chunk_id, score))
                if len(filtered_results) >= top_k:
                    break

        # Fallback: if user specified a document_id filter and no candidates passed oversampling,
        # fallback directly to the document's chunks to guarantee retrieval for the target document
        if not filtered_results and "document_id" in filters:
            doc_chunks = self.metadata_store.get_by_document(filters["document_id"])
            for record in doc_chunks[:top_k]:
                filtered_results.append((record["chunk_id"], 0.1))

        return filtered_results

    @staticmethod
    def _matches_filters(record: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check whether a chunk metadata record satisfies all filter criteria."""
        chunk_meta = record.get("metadata", {})
        for key, target_val in filters.items():
            val = chunk_meta.get(key)
            if val is None:
                val = record.get(key)

            if val is None:
                return False

            if isinstance(val, str) and isinstance(target_val, str):
                if val.strip().lower() != target_val.strip().lower():
                    return False
            elif val != target_val and str(val).strip().lower() != str(target_val).strip().lower():
                return False

        return True

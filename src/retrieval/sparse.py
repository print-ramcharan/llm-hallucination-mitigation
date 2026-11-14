"""Sparse Lexical Retriever using BM25+ inverted index."""

from __future__ import annotations

from typing import Any

from src.indexing.bm25_index import BM25Index
from src.indexing.metadata_store import MetadataStore
from src.retrieval.filters import matches_metadata_filters


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

        if not filters:
            return self.bm25_index.search(query, top_k=top_k)

        # Fast C-native pre-filtering if document_id filter is specified
        target_doc_id = filters.get("document_id") or filters.get("doc_id")
        if target_doc_id:
            pre_filtered = self.bm25_index.search(
                query, top_k=top_k, doc_id=str(target_doc_id)
            )
            if pre_filtered:
                # Double-check remaining filters if any
                valid_pre = []
                for cid, sc in pre_filtered:
                    rec = self.metadata_store.get(cid)
                    if rec and self._matches_filters(rec, filters):
                        valid_pre.append((cid, sc))
                if valid_pre:
                    return valid_pre[:top_k]

        oversample_k = min(max(top_k * 8, 80), self.bm25_index.total_documents)
        raw_results = self.bm25_index.search(query, top_k=oversample_k)

        filtered_results: list[tuple[str, float]] = []
        for chunk_id, score in raw_results:
            record = self.metadata_store.get(chunk_id)
            if record and self._matches_filters(record, filters):
                filtered_results.append((chunk_id, score))
                if len(filtered_results) >= top_k:
                    break

        # Expand if not enough filtered candidates found
        if (
            len(filtered_results) < top_k
            and oversample_k < self.bm25_index.total_documents
        ):
            full_results = self.bm25_index.search(
                query, top_k=self.bm25_index.total_documents
            )
            seen_ids = {cid for cid, _ in filtered_results}
            for chunk_id, score in full_results[oversample_k:]:
                if chunk_id in seen_ids:
                    continue
                record = self.metadata_store.get(chunk_id)
                if record and self._matches_filters(record, filters):
                    filtered_results.append((chunk_id, score))
                    if len(filtered_results) >= top_k:
                        break

        # Fallback: if user specified a document_id / doc_id filter and no candidates passed oversampling,
        # fallback directly to the document's chunks to guarantee retrieval for the target document
        target_doc_id = filters.get("document_id") or filters.get("doc_id")
        if not filtered_results and target_doc_id:
            doc_chunks = self.metadata_store.get_by_document(str(target_doc_id))
            for record in doc_chunks[:top_k]:
                if self._matches_filters(record, filters):
                    filtered_results.append((record["chunk_id"], 0.1))

        return filtered_results

    @staticmethod
    def _matches_filters(record: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check whether a chunk metadata record satisfies all filter criteria."""
        return matches_metadata_filters(record, filters)

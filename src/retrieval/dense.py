"""Dense Vector Retriever using FAISS and Sentence-Transformers L2-normalized embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.indexing.embeddings import EmbeddingEngine, default_embedding_engine
from src.indexing.metadata_store import MetadataStore
from src.indexing.vector_index import FaissVectorIndex
from src.retrieval.filters import matches_metadata_filters


class DenseRetriever:
    """Retrieves candidates using dense semantic embeddings and FAISS Inner Product search.

    Computes cosine similarity against L2-normalized chunk vectors and applies
    structured metadata filtering when specified.
    """

    def __init__(
        self,
        vector_index: FaissVectorIndex,
        metadata_store: MetadataStore,
        embedding_engine: EmbeddingEngine | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.metadata_store = metadata_store
        self.embedding_engine = embedding_engine or default_embedding_engine

    def retrieve(
        self,
        query: str,
        top_k: int = 25,
        filters: dict[str, Any] | None = None,
        query_vector: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        """Run dense vector search for the top_k most similar chunks.

        Args:
            query: Search query string.
            top_k: Number of candidates to retrieve.
            filters: Optional metadata filters (e.g. {'document_type': 'HR_POLICY'}).
            query_vector: Optional pre-computed L2-normalized embedding for query.

        Returns:
            List of (chunk_id, cosine_similarity_score) ordered descending by similarity.
        """
        if not query.strip() or self.vector_index.total_vectors == 0:
            return []

        # 1. Encode query to L2-normalized dense vector if not supplied
        if query_vector is None:
            query_vector = self.embedding_engine.embed_query(query)

        # 2. Fast path if no filters
        if not filters:
            return self.vector_index.search(query_vector, top_k=top_k)

        # 3. Dynamic oversampling to guarantee high recall with metadata filters
        oversample_k = min(max(top_k * 8, 80), self.vector_index.total_vectors)
        raw_results = self.vector_index.search(query_vector, top_k=oversample_k)

        filtered_results: list[tuple[str, float]] = []
        for chunk_id, score in raw_results:
            record = self.metadata_store.get(chunk_id)
            if record and self._matches_filters(record, filters):
                filtered_results.append((chunk_id, score))
                if len(filtered_results) >= top_k:
                    break

        # If not enough filtered candidates found, expand search to remaining vectors
        if (
            len(filtered_results) < top_k
            and oversample_k < self.vector_index.total_vectors
        ):
            full_results = self.vector_index.search(
                query_vector, top_k=self.vector_index.total_vectors
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
                    cid = record["chunk_id"]
                    score = 0.1
                    if cid in self.vector_index._embeddings_cache:
                        try:
                            score = float(
                                np.dot(
                                    query_vector,
                                    self.vector_index._embeddings_cache[cid],
                                )
                            )
                        except Exception:
                            pass
                    filtered_results.append((cid, score))

        return filtered_results

    @staticmethod
    def _matches_filters(record: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check whether a chunk metadata record satisfies all filter criteria."""
        return matches_metadata_filters(record, filters)

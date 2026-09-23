"""Dense Vector Retriever using FAISS and Sentence-Transformers L2-normalized embeddings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.indexing.embeddings import EmbeddingEngine, default_embedding_engine
from src.indexing.metadata_store import MetadataStore
from src.indexing.vector_index import FaissVectorIndex


class DenseRetriever:
    """Retrieves candidates using dense semantic embeddings and FAISS Inner Product search.

    Computes cosine similarity against L2-normalized chunk vectors and applies
    structured metadata filtering when specified.
    """

    def __init__(
        self,
        vector_index: FaissVectorIndex,
        metadata_store: MetadataStore,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> None:
        self.vector_index = vector_index
        self.metadata_store = metadata_store
        self.embedding_engine = embedding_engine or default_embedding_engine

    def retrieve(
        self,
        query: str,
        top_k: int = 25,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """Run dense vector search for the top_k most similar chunks.

        Args:
            query: Search query string.
            top_k: Number of candidates to retrieve.
            filters: Optional metadata filters (e.g. {'document_type': 'HR_POLICY'}).

        Returns:
            List of (chunk_id, cosine_similarity_score) ordered descending by similarity.
        """
        if not query.strip() or self.vector_index.total_vectors == 0:
            return []

        # 1. Encode query to L2-normalized dense vector
        query_vector = self.embedding_engine.embed_query(query)

        # 2. If filters are active, oversample candidates to compensate for post-filtering
        oversample_k = (
            min(max(top_k * 4, 50), self.vector_index.total_vectors)
            if filters
            else top_k
        )
        raw_results = self.vector_index.search(query_vector, top_k=oversample_k)

        # 3. Apply metadata filtering if specified
        if not filters:
            return raw_results[:top_k]

        filtered_results: List[Tuple[str, float]] = []
        for chunk_id, score in raw_results:
            record = self.metadata_store.get(chunk_id)
            if record and self._matches_filters(record, filters):
                filtered_results.append((chunk_id, score))
                if len(filtered_results) >= top_k:
                    break

        return filtered_results

    @staticmethod
    def _matches_filters(record: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check whether a chunk metadata record satisfies all filter criteria."""
        chunk_meta = record.get("metadata", {})
        for key, target_val in filters.items():
            val = chunk_meta.get(key)
            if val is None:
                # Fall back to root-level properties (e.g. document_id)
                val = record.get(key)

            if val is None:
                return False

            # Case-insensitive string matching
            if isinstance(val, str) and isinstance(target_val, str):
                if val.strip().lower() != target_val.strip().lower():
                    return False
            elif val != target_val:
                return False

        return True

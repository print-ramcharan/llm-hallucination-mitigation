"""FAISS Vector Index for semantic similarity search with L2-normalized embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.ingestion.models import DocumentChunk


class FaissVectorIndex:
    """FAISS-backed vector index mapping chunks to dense embeddings.

    Uses IndexFlatIP (Inner Product) which computes exact Cosine Similarity
    when vectors are L2-normalized.
    """

    def __init__(self, dimension: int = 384, storage_dir: Path | None = None) -> None:
        self.dimension = dimension
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/indexes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_dir / "vector_index.faiss"
        self.map_path = self.storage_dir / "vector_id_map.json"

        # Internal FAISS index
        self._index: faiss.Index = faiss.IndexFlatIP(self.dimension)
        # Vector ID (0..N-1) -> chunk_id
        self._id_to_chunk: list[str] = []
        # doc_id -> list of chunk_ids
        self._doc_to_chunks: dict[str, list[str]] = {}
        # chunk_id -> raw embedding vector
        self._embeddings_cache: dict[str, np.ndarray] = {}

        # Automatically load existing index if present
        self.load()

    @property
    def total_vectors(self) -> int:
        """Return the number of indexed vectors."""
        return self._index.ntotal

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks and their corresponding L2-normalized embeddings into the index."""
        if not chunks or embeddings.size == 0:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        vectors = np.asarray(embeddings, dtype=np.float32)

        # Append vectors to FAISS index
        self._index.add(vectors)

        for chunk, vec in zip(chunks, vectors):
            chunk_id = chunk.metadata.get("chunk_id", chunk.id) if chunk.metadata else chunk.id
            doc_id = chunk.document_id

            self._id_to_chunk.append(chunk_id)
            self._doc_to_chunks.setdefault(doc_id, []).append(chunk_id)
            self._embeddings_cache[chunk_id] = vec

        self.save()

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Search the index for the top_k most similar chunks using cosine similarity.

        Returns:
            List of (chunk_id, cosine_similarity_score) sorted descending by similarity.
        """
        if self.total_vectors == 0:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = np.expand_dims(q_vec, axis=0)

        k = min(top_k, self.total_vectors)
        distances, indices = self._index.search(q_vec, k)

        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self._id_to_chunk):
                chunk_id = self._id_to_chunk[idx]
                results.append((chunk_id, float(dist)))

        return results

    def remove_document(self, doc_id: str) -> None:
        """Remove all vectors belonging to a document and rebuild the index."""
        if doc_id not in self._doc_to_chunks:
            return

        chunks_to_remove = set(self._doc_to_chunks.pop(doc_id, []))
        for cid in chunks_to_remove:
            self._embeddings_cache.pop(cid, None)

        # Reconstruct remaining index
        new_id_to_chunk: list[str] = []
        new_vectors_list: list[np.ndarray] = []

        for cid in self._id_to_chunk:
            if cid not in chunks_to_remove and cid in self._embeddings_cache:
                new_id_to_chunk.append(cid)
                new_vectors_list.append(self._embeddings_cache[cid])

        self._id_to_chunk = new_id_to_chunk
        self._index = faiss.IndexFlatIP(self.dimension)

        if new_vectors_list:
            rebuilt_matrix = np.vstack(new_vectors_list).astype(np.float32)
            self._index.add(rebuilt_matrix)

        self.save()

    def save(self) -> None:
        """Persist the FAISS index and ID maps to disk."""
        try:
            faiss.write_index(self._index, str(self.index_path))
            map_data = {
                "id_to_chunk": self._id_to_chunk,
                "doc_to_chunks": self._doc_to_chunks,
                # Convert float arrays to lists for json persistence of cache
                "embeddings": {
                    cid: vec.tolist() for cid, vec in self._embeddings_cache.items()
                },
            }
            with open(self.map_path, "w", encoding="utf-8") as f:
                json.dump(map_data, f)
        except Exception as exc:
            # Non-fatal log or pass if disk write fails
            print(f"Warning: Failed to persist FAISS index: {exc}")

    def load(self) -> None:
        """Load the FAISS index and ID maps from disk if available."""
        if self.index_path.is_file() and self.map_path.is_file():
            try:
                self._index = faiss.read_index(str(self.index_path))
                with open(self.map_path, "r", encoding="utf-8") as f:
                    map_data = json.load(f)
                    self._id_to_chunk = map_data.get("id_to_chunk", [])
                    self._doc_to_chunks = map_data.get("doc_to_chunks", {})
                    raw_emb = map_data.get("embeddings", {})
                    self._embeddings_cache = {
                        cid: np.array(vec, dtype=np.float32)
                        for cid, vec in raw_emb.items()
                    }
            except Exception as exc:
                print(f"Warning: Failed to load FAISS index from disk: {exc}")
                self._index = faiss.IndexFlatIP(self.dimension)
                self._id_to_chunk = []
                self._doc_to_chunks = {}
                self._embeddings_cache = {}

    def clear(self) -> None:
        """Wipe the in-memory index and delete persisted files."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._id_to_chunk = []
        self._doc_to_chunks = {}
        self._embeddings_cache = {}
        if self.index_path.is_file():
            self.index_path.unlink()
        if self.map_path.is_file():
            self.map_path.unlink()

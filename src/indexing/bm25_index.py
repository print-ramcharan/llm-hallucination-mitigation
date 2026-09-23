"""BM25 Lexical Index for sparse keyword search (required for Module 3 Hybrid Retrieval)."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Plus

# pyrefly: ignore [missing-import]
from src.ingestion.models import DocumentChunk

# Word extraction tokenizer for BM25 lexical analysis
_WORD_REGEX = re.compile(r"\w+")


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase lexical terms for BM25 indexing."""
    if not text:
        return []
    return [w.lower() for w in _WORD_REGEX.findall(text)]


class BM25Index:
    """BM25 sparse lexical retrieval index for keyword-based ranking.

    Prepares tokenized inverted representation for downstream Module 3 hybrid retrieval.
    Uses BM25Plus for guaranteed positive IDF across both small and large corpora.
    """

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/indexes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "bm25_index.pkl"

        self._chunk_ids: list[str] = []
        self._doc_to_chunks: dict[str, list[str]] = {}
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Plus | None = None

        self.load()

    @property
    def total_documents(self) -> int:
        """Return the number of documents/chunks in the BM25 index."""
        return len(self._chunk_ids)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add new chunks to the BM25 corpus and re-fit the index."""
        if not chunks:
            return

        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id", chunk.id) if chunk.metadata else chunk.id
            doc_id = chunk.document_id

            tokens = bm25_tokenize(chunk.content)
            self._chunk_ids.append(chunk_id)
            self._tokenized_corpus.append(tokens)
            self._doc_to_chunks.setdefault(doc_id, []).append(chunk_id)

        # Re-fit BM25 on updated corpus
        if self._tokenized_corpus:
            self._bm25 = BM25Plus(self._tokenized_corpus)

        self.save()

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search the BM25 index for the most relevant chunks matching the query.

        Returns:
            List of (chunk_id, bm25_score) sorted descending by relevance.
        """
        if not self._bm25 or not self._chunk_ids or not query.strip():
            return []

        query_tokens = bm25_tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        query_set = set(query_tokens)
        results: list[tuple[str, float]] = []
        for idx in top_indices[:top_k]:
            # Ensure the document actually contains matching terms
            doc_tokens = set(self._tokenized_corpus[idx])
            if query_set.intersection(doc_tokens):
                results.append((self._chunk_ids[idx], float(scores[idx])))

        return results

    def remove_document(self, doc_id: str) -> None:
        """Remove all chunks associated with a document and re-fit the BM25 model."""
        if doc_id not in self._doc_to_chunks:
            return

        chunks_to_remove = set(self._doc_to_chunks.pop(doc_id, []))

        new_chunk_ids: list[str] = []
        new_tokenized_corpus: list[list[str]] = []

        for cid, tokens in zip(self._chunk_ids, self._tokenized_corpus):
            if cid not in chunks_to_remove:
                new_chunk_ids.append(cid)
                new_tokenized_corpus.append(tokens)

        self._chunk_ids = new_chunk_ids
        self._tokenized_corpus = new_tokenized_corpus

        if self._tokenized_corpus:
            self._bm25 = BM25Plus(self._tokenized_corpus)
        else:
            self._bm25 = None

        self.save()

    def save(self) -> None:
        """Persist the BM25 state to disk."""
        try:
            state = {
                "chunk_ids": self._chunk_ids,
                "doc_to_chunks": self._doc_to_chunks,
                "tokenized_corpus": self._tokenized_corpus,
            }
            with open(self.index_path, "wb") as f:
                pickle.dump(state, f)
        except Exception as exc:
            print(f"Warning: Failed to persist BM25 index: {exc}")

    def load(self) -> None:
        """Load persisted BM25 state from disk."""
        if self.index_path.is_file():
            try:
                with open(self.index_path, "rb") as f:
                    state = pickle.load(f)
                    self._chunk_ids = state.get("chunk_ids", [])
                    self._doc_to_chunks = state.get("doc_to_chunks", {})
                    self._tokenized_corpus = state.get("tokenized_corpus", [])
                    if self._tokenized_corpus:
                        self._bm25 = BM25Plus(self._tokenized_corpus)
            except Exception as exc:
                print(f"Warning: Failed to load BM25 index from disk: {exc}")
                self._chunk_ids = []
                self._doc_to_chunks = {}
                self._tokenized_corpus = []
                self._bm25 = None

    def clear(self) -> None:
        """Clear the in-memory BM25 index and remove persisted file."""
        self._chunk_ids = []
        self._doc_to_chunks = {}
        self._tokenized_corpus = []
        self._bm25 = None
        if self.index_path.is_file():
            self.index_path.unlink()

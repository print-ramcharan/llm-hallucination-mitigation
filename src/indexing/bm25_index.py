"""BM25 Lexical Index for sparse keyword search with SQLite FTS5 acceleration and BM25Plus fallback."""

from __future__ import annotations

import pickle
import re
import sqlite3
from pathlib import Path

import numpy as np
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


class SQLiteFTS5Index:
    """C-native SQLite FTS5 full-text search index with built-in BM25 scoring and instant pre-filtering."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/indexes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / "fts5_bm25.db"
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        doc_id UNINDEXED,
                        content,
                        tokenize='porter unicode61'
                    );
                    """
                )
                conn.commit()
        except Exception as exc:
            print(f"Warning: Failed to init SQLite FTS5 table: {exc}")

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        rows = []
        for chunk in chunks:
            chunk_id = (
                chunk.metadata.get("chunk_id", chunk.id) if chunk.metadata else chunk.id
            )
            rows.append((chunk_id, str(chunk.document_id), chunk.content))
        try:
            with self._get_connection() as conn:
                conn.executemany(
                    "INSERT INTO chunks_fts(chunk_id, doc_id, content) VALUES (?, ?, ?);",
                    rows,
                )
                conn.commit()
        except Exception:
            pass

    def search(
        self, query: str, top_k: int = 5, doc_id: str | None = None
    ) -> list[tuple[str, float]]:
        cleaned_words = [re.sub(r"[^\w]", "", w) for w in query.split()]
        valid_words = [w for w in cleaned_words if len(w) >= 2]
        if not valid_words:
            return []
        fts_query = " OR ".join(valid_words)
        try:
            with self._get_connection() as conn:
                if doc_id:
                    cursor = conn.execute(
                        """
                        SELECT chunk_id, -bm25(chunks_fts) as score
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ? AND doc_id = ?
                        ORDER BY score DESC LIMIT ?;
                        """,
                        (fts_query, str(doc_id), top_k),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT chunk_id, -bm25(chunks_fts) as score
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ?
                        ORDER BY score DESC LIMIT ?;
                        """,
                        (fts_query, top_k),
                    )
                return [(row[0], float(max(0.001, row[1]))) for row in cursor.fetchall()]
        except Exception:
            return []

    def remove_document(self, doc_id: str) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?;", (str(doc_id),))
                conn.commit()
        except Exception:
            pass

    def clear(self) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM chunks_fts;")
                conn.commit()
        except Exception:
            pass


class BM25Index:
    """BM25 sparse lexical retrieval index for keyword-based ranking.

    Combines Python BM25Plus for precise ranking with C-native SQLite FTS5
    for sub-millisecond keyword search and metadata pre-filtering.
    """

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/indexes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "bm25_index.pkl"

        self._chunk_ids: list[str] = []
        self._doc_to_chunks: dict[str, list[str]] = {}
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Plus | None = None
        self.fts5: SQLiteFTS5Index = SQLiteFTS5Index(storage_dir=self.storage_dir)

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
            chunk_id = (
                chunk.metadata.get("chunk_id", chunk.id) if chunk.metadata else chunk.id
            )
            doc_id = chunk.document_id

            tokens = bm25_tokenize(chunk.content)
            self._chunk_ids.append(chunk_id)
            self._tokenized_corpus.append(tokens)
            doc_chunks = self._doc_to_chunks.setdefault(doc_id, [])
            if chunk_id not in doc_chunks:
                doc_chunks.append(chunk_id)

        # Re-fit BM25 on updated corpus
        if self._tokenized_corpus:
            self._bm25 = BM25Plus(self._tokenized_corpus)

        # Sync with C-native SQLite FTS5
        self.fts5.add_chunks(chunks)
        self.save()

    def search(
        self, query: str, top_k: int = 5, doc_id: str | None = None
    ) -> list[tuple[str, float]]:
        """Search the BM25 index for the most relevant chunks matching the query.

        Returns:
            List of (chunk_id, bm25_score) sorted descending by relevance.
        """
        # If doc_id filter is requested, leverage C-native SQLite FTS5 for instant pre-filtering
        if doc_id:
            fts_results = self.fts5.search(query, top_k=top_k, doc_id=doc_id)
            if fts_results:
                return fts_results

        if not self._bm25 or not self._chunk_ids or not query.strip():
            return []

        query_tokens = bm25_tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        scores = np.asarray(raw_scores, dtype=np.float32)
        nonzero_idx = np.flatnonzero(scores > 0)
        if len(nonzero_idx) == 0:
            return []

        # Sort only nonzero scores descending using numpy partial sort for efficiency
        top_k_bound = min(top_k * 8, len(nonzero_idx))
        if len(nonzero_idx) > top_k_bound:
            partitioned = np.argpartition(scores[nonzero_idx], -top_k_bound)[-top_k_bound:]
            candidate_indices = nonzero_idx[partitioned]
            sorted_order = np.argsort(scores[candidate_indices])[::-1]
            top_indices = candidate_indices[sorted_order]
        else:
            sorted_order = np.argsort(scores[nonzero_idx])[::-1]
            top_indices = nonzero_idx[sorted_order]

        query_set = set(query_tokens)
        results: list[tuple[str, float]] = []
        for idx in top_indices:
            # Ensure the document actually contains matching terms
            doc_tokens = set(self._tokenized_corpus[idx])
            if query_set.intersection(doc_tokens):
                results.append((self._chunk_ids[idx], float(scores[idx])))
                if len(results) >= top_k:
                    break

        return results

    def remove_document(self, doc_id: str) -> None:
        """Remove all chunks associated with a document and re-fit the BM25 model."""
        if doc_id not in self._doc_to_chunks:
            return

        chunks_to_remove = set(self._doc_to_chunks.pop(doc_id, []))

        new_chunk_ids: list[str] = []
        new_tokenized_corpus: list[list[str]] = []

        for cid, tokens in zip(self._chunk_ids, self._tokenized_corpus, strict=True):
            if cid not in chunks_to_remove:
                new_chunk_ids.append(cid)
                new_tokenized_corpus.append(tokens)

        self._chunk_ids = new_chunk_ids
        self._tokenized_corpus = new_tokenized_corpus

        if self._tokenized_corpus:
            self._bm25 = BM25Plus(self._tokenized_corpus)
        else:
            self._bm25 = None

        self.fts5.remove_document(doc_id)
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
        self.fts5.clear()
        if self.index_path.is_file():
            self.index_path.unlink()

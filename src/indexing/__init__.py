"""Indexing package: Vector Index (FAISS/HNSW), BM25 Lexical Index, and Metadata Store."""

from src.indexing.bm25_index import BM25Index, bm25_tokenize
from src.indexing.embeddings import (
    DEFAULT_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingEngine,
    default_embedding_engine,
)
from src.indexing.manager import IndexManager, default_index_manager
from src.indexing.metadata_store import MetadataStore
from src.indexing.vector_index import FaissVectorIndex

__all__ = [
    "DEFAULT_DIMENSION",
    "DEFAULT_EMBEDDING_MODEL",
    "BM25Index",
    "EmbeddingEngine",
    "FaissVectorIndex",
    "IndexManager",
    "MetadataStore",
    "bm25_tokenize",
    "default_embedding_engine",
    "default_index_manager",
]

"""Tests for Retrieval Speed & Precision Optimizations:
1. FAISS HNSW indexing and configuration
2. SQLite FTS5 C-native BM25 search and pre-filtering
3. Small-to-Big (Parent-Document) chunk generation
4. HyDE (Hypothetical Document Embeddings) and Query Expansions
5. Adaptive Cascaded Reranking
"""

from pathlib import Path

import faiss

from src.chunking.semantic import SemanticChunker
from src.indexing.bm25_index import SQLiteFTS5Index
from src.indexing.embeddings import EmbeddingEngine
from src.indexing.vector_index import FaissVectorIndex
from src.ingestion.models import DocumentChunk
from src.query_processing.pipeline import QueryProcessor
from src.query_processing.rewriter import QueryRewriter
from src.reranking.engine import CrossEncoderEngine
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.models import RetrievedCandidate


class TestRetrievalOptimizations:
    def test_faiss_hnsw_index(self, tmp_path: Path):
        """Verify HNSW index creation, search, persistence, and efSearch setting."""
        index = FaissVectorIndex(
            dimension=384,
            storage_dir=tmp_path,
            index_type="hnsw",
            hnsw_m=32,
            hnsw_ef_search=64,
        )
        assert isinstance(index._index, faiss.IndexHNSWFlat)
        assert index._index.hnsw.efSearch == 64

        # Add sample vectors
        engine = EmbeddingEngine()
        chunks = [
            DocumentChunk(id="c1", document_id="doc1", content="Cloud infrastructure architecture and deployment."),
            DocumentChunk(id="c2", document_id="doc1", content="Employee health coverage and insurance benefits."),
        ]
        embeddings = engine.embed_chunks(chunks)
        index.add_chunks(chunks, embeddings)
        assert index.total_vectors == 2

        # Search
        q_vec = engine.embed_query("health insurance")
        results = index.search(q_vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "c2"

        # Reload and check HNSW persistence
        reloaded = FaissVectorIndex(
            dimension=384,
            storage_dir=tmp_path,
            index_type="hnsw",
            hnsw_ef_search=128,
        )
        assert reloaded.total_vectors == 2
        assert reloaded._index.hnsw.efSearch == 128

    def test_sqlite_fts5_bm25_prefiltering(self, tmp_path: Path):
        """Verify SQLite FTS5 indexing, BM25 scoring, and instant doc_id pre-filtering."""
        fts = SQLiteFTS5Index(storage_dir=tmp_path)
        chunks = [
            DocumentChunk(id="c1", document_id="doc_hr", content="Annual vacation and sick leave policy."),
            DocumentChunk(id="c2", document_id="doc_eng", content="Python coding standards and git branch policy."),
            DocumentChunk(id="c3", document_id="doc_hr", content="Dental and vision healthcare insurance benefits."),
        ]
        fts.add_chunks(chunks)

        # Global search
        res = fts.search("policy", top_k=5)
        assert len(res) == 2
        chunk_ids = [r[0] for r in res]
        assert "c1" in chunk_ids and "c2" in chunk_ids

        # Pre-filtered search targeting doc_hr only
        res_hr = fts.search("policy", top_k=5, doc_id="doc_hr")
        assert len(res_hr) == 1
        assert res_hr[0][0] == "c1"

        # Removal
        fts.remove_document("doc_hr")
        res_after = fts.search("policy", top_k=5)
        assert len(res_after) == 1
        assert res_after[0][0] == "c2"

    def test_small_to_big_chunking(self):
        """Verify decomposition of large parent chunks into granular child chunks."""
        chunker = SemanticChunker(target_tokens=350, overlap_tokens=50)
        long_content = (
            "Section 1: Company Overview. We build enterprise intelligence systems. "
            "Our primary mission is reliability and scalability. "
            "Section 2: Security Architecture. All data at rest is encrypted using AES-256 GCM. "
            "Keys are rotated every 90 days in hardware security modules. "
            "Section 3: Compliance Standards. We adhere strictly to SOC-2 Type II standards. "
            "Annual independent audits are conducted every October."
        )
        parent = DocumentChunk(
            id="parent_doc1_c1",
            document_id="doc1",
            chunk_index=0,
            content=long_content,
        )

        child_chunks = chunker.create_small_to_big_chunks(
            [parent], child_target_tokens=30, child_overlap_tokens=10
        )
        assert len(child_chunks) >= 2
        for child in child_chunks:
            assert child.parent_id == "parent_doc1_c1"
            assert child.parent_content == long_content
            assert child.is_child is True
            assert child.id.startswith("parent_doc1_c1_c")

    def test_hyde_and_query_expansion(self):
        """Verify HyDE declarative synthesis and synonym expansion."""
        rewriter = QueryRewriter()
        processor = QueryProcessor(rewriter=rewriter)

        processed = processor.process("What is the vacation policy?")
        assert processed.hyde_passage is not None
        assert "vacation policy" in processed.hyde_passage.lower()
        assert len(processed.expanded_queries) >= 1

        # Check comparison intent
        comp_hyde = rewriter.generate_hyde_passage("FastAPI vs Flask", intent="comparison")
        assert "comparison" in comp_hyde.lower() or "tradeoffs" in comp_hyde.lower()

    def test_adaptive_reranking(self):
        """Verify adaptive pruning dynamically caps cross-encoder evaluation when score separation is clear."""
        engine = CrossEncoderEngine()
        reranker = CrossEncoderReranker(engine=engine, default_top_n=5)

        # Mock 15 candidates where the top candidate has high RRF score and rest drop off quickly
        candidates = []
        for i in range(15):
            rrf = 0.05 if i == 0 else (0.01 / (i + 1))
            candidates.append(
                RetrievedCandidate(
                    chunk_id=f"c_{i}",
                    document_id="doc1",
                    content=f"Content passage number {i} discussing software engineering.",
                    chunk_index=i,
                    rrf_score=rrf,
                    dense_score=0.9 if i == 0 else 0.3,
                )
            )

        # Rerank with adaptive pruning
        ranked = reranker.rerank(
            query="software engineering",
            candidates=candidates,
            adaptive_pruning=True,
            top_n=5,
        )
        assert len(ranked.chunks) <= 5
        assert ranked.total_input_candidates == 15

"""Benchmark dataset generation and evaluation runner for RAG pipeline.

Creates a 10-document technical corpus with ~30-40 chunks, indexes them into FAISS and BM25,
matches 16 diverse evaluation queries to their ground-truth chunk IDs, and executes the
benchmark to evaluate Table II and Table III.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure offline operation without network timeouts in sandbox
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["LLM_PROVIDER"] = "offline"

from src.evaluation.benchmark import run_benchmark  # noqa: E402
from src.indexing.manager import IndexManager  # noqa: E402
from src.ingestion.registry import default_registry  # noqa: E402

CORPUS = [
    {
        "filename": "01_context_degradation.txt",
        "text": """Context Degradation and Lost-in-the-Middle Phenomenon

Large Language Models experience severe performance degradation when processing extensive prompt contexts. This failure mode is known as the Lost-in-the-Middle phenomenon, where models attend disproportionately to information positioned at the extreme beginning or extreme end of the context window. When critical factual evidence is positioned in the central regions of a long prompt, retrieval accuracy and reasoning performance decay significantly, often falling below thirty percent accuracy.

Context dilution further compounds this vulnerability in production environments. Standard retrieval-augmented generation pipelines frequently inject repetitive or redundant passages that exhaust the finite token budget while introducing distracting semantic noise. Experiments demonstrate that increasing context size with unpruned documents causes transformer attention weights to disperse across irrelevant tokens, leading to catastrophic context dilution.

To counteract attention decay, the system architecture enforces a strict 1500 token ceiling on assembled prompts. By filtering out non-salient propositions, the prompt preserves high density information while eliminating context dilution.""",
    },
    {
        "filename": "02_attention_distribution.txt",
        "text": """Attention Distribution and U-Shaped Boundary Reordering

Transformer attention mechanisms display an innate U-shaped attention distribution across sequence length. Tokens situated near the initial prompt instructions (primacy effect) and tokens immediately preceding the final query prompt (recency effect) receive significantly higher cross-attention weight values than intermediate tokens.

To exploit this structural property, evidence chunks undergo U-shaped boundary reordering. Rather than ordering evidence monotonically by retrieval score, the top-ranked candidate is placed at the beginning of the context, the second-ranked candidate is placed at the end, and subsequent lower-ranked candidates are interleaved toward the middle.

Empirical evaluations confirm that reordering candidate chunks into an alternating U-shaped distribution restores up to ninety-two percent of degraded model recall. This algorithmic placement ensures the LLM focuses its primary attention on decisive evidence.""",
    },
    {
        "filename": "03_dense_vector_retrieval.txt",
        "text": """Dense Vector Retrieval and Neural Semantic Search

Dense vector search relies on bi-encoder neural embeddings generated via the sentence-transformers all-MiniLM-L6-v2 model. Text passages are mapped into dense 384-dimensional continuous vector representations and indexed within an inner-product FAISS vector database.

Dense semantic retrieval excels at capturing conceptual intent, thematic synonyms, and abstract natural language expressions. It successfully bridges semantic gaps where query terms and document passages use different terminology to describe identical concepts.

However, dense bi-encoders struggle with rare alphanumeric tokens, proper nouns, exact part numbers, and specialized acronyms. Because embeddings compress passages into continuous vectors, specific lexical tokens can be smoothed over, resulting in semantic false negatives for keyword-exact queries.""",
    },
    {
        "filename": "04_sparse_lexical_search.txt",
        "text": """Sparse Lexical Search and BM25 Inverted Index

Sparse lexical retrieval operates via an inverted token index utilizing the Okapi BM25 ranking algorithm. BM25 scores document relevance by balancing term frequency and inverse document frequency while normalizing for document length variations across the collection.

BM25 provides deterministic, high-precision retrieval for rare keywords, technical jargon, proper nouns, and specific acronyms. It guarantees that queries targeting specific technical identifiers retrieve passages containing the exact tokens.

Conversely, BM25 exhibits complete blindness to semantic synonyms and conceptual paraphrases. When a user queries a concept using vocabulary absent from the source document, BM25 yields zero hits, creating a lexical vocabulary mismatch problem.""",
    },
    {
        "filename": "05_reciprocal_rank_fusion.txt",
        "text": """Reciprocal Rank Fusion and Hybrid Arbitration

To overcome the complementary failure modes of dense vector retrieval and sparse lexical search, our system executes both retrieval engines in parallel and arbitrates candidates via Reciprocal Rank Fusion.

Combining raw similarity scores from FAISS and BM25 through linear combination is unreliable due to distinct score distributions and unbounded BM25 ranges. Reciprocal Rank Fusion overcomes this by relying solely on rank positions.

RRF scores each candidate document candidate using the formula RRF(d) = sum(1 / (k + rank_i(d))), where k is a smoothing constant traditionally set to 60. By setting k=60, RRF penalizes lower-ranked candidates while rewarding items that achieve consensus near the top of both lists. The top twenty fused candidates proceed to the reranking stage.""",
    },
    {
        "filename": "06_cross_encoder_reranking.txt",
        "text": """Cross-Encoder Reranking and Score Calibration

Bi-encoder retrievers compute query and passage representations separately to allow pre-indexing. In contrast, the deep Cross-Encoder reranker processes the concatenated query-passage pair simultaneously through full transformer self-attention layers.

Every query token directly attends to every document token, enabling fine-grained interaction modeling that detects nuanced factual contradictions and relevance. Evaluating candidate passages with full cross-attention eliminates up to seventy-eight percent of bi-encoder retrieval errors.

Raw cross-encoder logit scores are calibrated into probabilistic confidence scores using a Sigmoid transformation: S(x) = 1 / (1 + exp(-x)). A calibrated relevance threshold of 0.35 is strictly enforced, pruning candidates with confidence below this cutoff.""",
    },
    {
        "filename": "07_sentence_salience_pruning.txt",
        "text": """Extractive Sentence Salience Scoring and Pruning

Retrieved 220-word passages often contain filler sentences, narrative transitions, and boilerplate text that dilute core evidence. Extractive sentence pruning segments each chunk into constituent sentences using boundary-aware punctuation splitting.

Each candidate sentence is individually embedded and evaluated against the user query representation. The cosine similarity metric establishes the sentence salience score.

Sentences falling below a salience threshold of 0.15 are pruned as peripheral noise. This selective extraction preserves core evidentiary propositions while discarding thirty to fifty percent of surrounding uninformative text.""",
    },
    {
        "filename": "08_seam_deduplication.txt",
        "text": """Chunk Boundary Seam Deduplication Protocol

Because document ingestion utilizes a sliding window chunking mechanism with a 30-word overlap, consecutive chunks frequently share identical or nearly identical boundary sentences across their seams.

If left unmitigated, these duplicate sentences consume scarce token allowances and bias LLM attention. The seam deduplicator compares candidate sentence pairs across adjacent chunks using Jaccard word-overlap similarity.

Pairs of sentences exhibiting similarity exceeding tau = 0.75 are flagged as duplicate seam artifacts, and the redundant instance is stripped. Overall, context compaction achieves an average token reduction of forty-five percent while preserving one hundred percent of key factual assertions.""",
    },
    {
        "filename": "09_evidence_sufficiency_gating.txt",
        "text": """Evidence Sufficiency Gating and Truthful Abstention

Generative hallucinations often occur when language models attempt to answer queries despite the absence of substantiated evidence in retrieved documents. Our architecture interposes an Evidence Sufficiency Gating stage prior to generation.

The Evidence Sufficiency Classifier evaluates semantic coverage across all extracted propositions. If the compacted evidence fails to cover required query aspects or falls below a sufficiency threshold of 0.40, generation is aborted and a diagnostic Truthful Abstention response is emitted.

When evidence is deemed sufficient, the Grounded Prompt Synthesizer enforces mandatory bracketed citations in the format [Doc X, Chunk Y] for every assertion. This prevents hallucinated ungrounded claims before inference begins.""",
    },
    {
        "filename": "10_nli_verification_and_memory.txt",
        "text": """NLI Claim Verification and External Episodic Memory

Following response generation, the Anti-Hallucination Verifier decomposes generated answers into atomic claims. Each claim is evaluated against source evidence using Natural Language Inference (NLI) classification, categorizing propositions into Entailed, Neutral, or Contradicted states.

Claims classified as Neutral or Contradicted are flagged as hallucinations, ensuring verified factual reliability. Verified propositions are subsequently committed to long-term storage.

Episodic state is decoupled into an external ACID-compliant SQLite store (memory.db) that enforces write-ahead logging (WAL) mode for concurrency. When subsequent queries arrive, the Memory Synchronizer retrieves the top two relevant verified facts within a constrained memory token budget of 250 tokens.""",
    },
]


def ingest_corpus(index_dir: Path) -> tuple[IndexManager, list[tuple[str, str]]]:
    """Ingest corpus into IndexManager and return list of (chunk_id, chunk_text)."""
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    manager = IndexManager(storage_dir=index_dir)
    all_chunks: list[tuple[str, str]] = []

    for item in CORPUS:
        data = item["text"].encode("utf-8")
        doc = default_registry.parse_bytes(
            content=data,
            filename=item["filename"],
            file_size_bytes=len(data),
            document_type="TECHNICAL_SPEC",
        )
        manager.index_document(doc)
        for chunk in doc.chunks:
            all_chunks.append((chunk.id, chunk.content))
        print(f"Ingested {item['filename']} -> {len(doc.chunks)} chunks")

    print(f"Total indexed chunks: {len(all_chunks)}")
    return manager, all_chunks


def find_chunk_id(all_chunks: list[tuple[str, str]], keyword: str) -> str:
    """Locate chunk ID containing keyword."""
    for chunk_id, content in all_chunks:
        if keyword.lower() in content.lower():
            return chunk_id
    raise ValueError(f"Keyword '{keyword}' not found in any chunk!")


def build_evaluation_cases(all_chunks: list[tuple[str, str]]) -> list[dict]:
    """Build 16 annotated benchmark queries with exact ground-truth chunk mapping."""
    # Find matching chunks for each test case
    cid_degradation = find_chunk_id(all_chunks, "Lost-in-the-Middle phenomenon")
    cid_dilution = find_chunk_id(all_chunks, "Context dilution further compounds")
    cid_attention_u = find_chunk_id(all_chunks, "U-shaped attention distribution")
    cid_reorder_92 = find_chunk_id(all_chunks, "restores up to ninety-two percent")
    cid_dense = find_chunk_id(all_chunks, "all-MiniLM-L6-v2 model")
    cid_sparse = find_chunk_id(all_chunks, "Okapi BM25 ranking algorithm")
    cid_rrf = find_chunk_id(all_chunks, "smoothing constant traditionally set to 60")
    cid_cross_encoder = find_chunk_id(all_chunks, "seventy-eight percent of bi-encoder")
    cid_sigmoid = find_chunk_id(all_chunks, "Sigmoid transformation: S(x)")
    cid_salience_pruning = find_chunk_id(all_chunks, "salience threshold of 0.15")
    cid_jaccard_dedup = find_chunk_id(all_chunks, "exceeding tau = 0.75")
    cid_token_reduction = find_chunk_id(
        all_chunks, "average token reduction of forty-five percent"
    )
    cid_sufficiency = find_chunk_id(all_chunks, "sufficiency threshold of 0.40")
    cid_bracketed_format = find_chunk_id(all_chunks, "format [Doc X, Chunk Y]")
    cid_nli_states = find_chunk_id(all_chunks, "Entailed, Neutral, or Contradicted")
    cid_sqlite_wal = find_chunk_id(all_chunks, "write-ahead logging (WAL) mode")
    cid_memory_budget = find_chunk_id(all_chunks, "memory token budget of 250 tokens")

    cases = [
        {
            "id": "q01",
            "query": "What causes the Lost-in-the-Middle phenomenon in Large Language Models?",
            "relevant_chunk_ids": [cid_degradation],
            "gold_answer": "Large Language Models attend disproportionately to information positioned at the extreme beginning or extreme end of the context window, causing performance degradation for central information.",
            "gold_citation_ids": [cid_degradation],
        },
        {
            "id": "q02",
            "query": "How much degraded recall does U-shaped boundary reordering restore?",
            "relevant_chunk_ids": [cid_reorder_92],
            "gold_answer": "Reordering candidate chunks into an alternating U-shaped distribution restores up to ninety-two percent of degraded model recall.",
            "gold_citation_ids": [cid_reorder_92],
        },
        {
            "id": "q03",
            "query": "What is the formula and smoothing constant k for Reciprocal Rank Fusion?",
            "relevant_chunk_ids": [cid_rrf],
            "gold_answer": "RRF scores each candidate using RRF(d) = sum(1 / (k + rank_i(d))), where k is a smoothing constant traditionally set to 60.",
            "gold_citation_ids": [cid_rrf],
        },
        {
            "id": "q04",
            "query": "Why combine dense vector retrieval with sparse BM25 lexical search?",
            "relevant_chunk_ids": [cid_dense, cid_sparse],
            "gold_answer": "Dense retrieval captures conceptual intent and synonyms but fails with exact keywords, while BM25 inverted index retrieves rare keywords and acronyms but cannot bridge semantic vocabulary mismatches.",
            "gold_citation_ids": [cid_dense, cid_sparse],
        },
        {
            "id": "q05",
            "query": "How are raw cross-encoder logit scores calibrated into confidence scores?",
            "relevant_chunk_ids": [cid_sigmoid],
            "gold_answer": "Raw cross-encoder logit scores are calibrated into probabilistic confidence scores using a Sigmoid transformation: S(x) = 1 / (1 + exp(-x)).",
            "gold_citation_ids": [cid_sigmoid],
        },
        {
            "id": "q06",
            "query": "What percentage of bi-encoder retrieval errors does cross-encoder reranking eliminate?",
            "relevant_chunk_ids": [cid_cross_encoder],
            "gold_answer": "Evaluating candidate passages with full cross-attention eliminates up to seventy-eight percent of bi-encoder retrieval errors.",
            "gold_citation_ids": [cid_cross_encoder],
        },
        {
            "id": "q07",
            "query": "What Jaccard word-overlap threshold is used for chunk seam deduplication?",
            "relevant_chunk_ids": [cid_jaccard_dedup],
            "gold_answer": "Pairs of sentences exhibiting similarity exceeding tau = 0.75 are flagged as duplicate seam artifacts and stripped.",
            "gold_citation_ids": [cid_jaccard_dedup],
        },
        {
            "id": "q08",
            "query": "What average token reduction does extractive context compaction achieve?",
            "relevant_chunk_ids": [cid_token_reduction],
            "gold_answer": "Context compaction achieves an average token reduction of forty-five percent while preserving one hundred percent of key factual assertions.",
            "gold_citation_ids": [cid_token_reduction],
        },
        {
            "id": "q09",
            "query": "What sufficiency threshold triggers truthful abstention before generation?",
            "relevant_chunk_ids": [cid_sufficiency],
            "gold_answer": "If compacted evidence falls below a sufficiency threshold of 0.40, generation is aborted and a diagnostic Truthful Abstention response is emitted.",
            "gold_citation_ids": [cid_sufficiency],
        },
        {
            "id": "q10",
            "query": "Into which three classification states are claims categorized during NLI verification?",
            "relevant_chunk_ids": [cid_nli_states],
            "gold_answer": "Each claim is evaluated using Natural Language Inference classification, categorizing propositions into Entailed, Neutral, or Contradicted states.",
            "gold_citation_ids": [cid_nli_states],
        },
        {
            "id": "q11",
            "query": "How is episodic state persisted and which database concurrency mode is enforced?",
            "relevant_chunk_ids": [cid_sqlite_wal],
            "gold_answer": "Episodic state is decoupled into an external ACID-compliant SQLite store (memory.db) that enforces write-ahead logging (WAL) mode for concurrency.",
            "gold_citation_ids": [cid_sqlite_wal],
        },
        {
            "id": "q12",
            "query": "What token budget is allocated when injecting past verified facts from memory?",
            "relevant_chunk_ids": [cid_memory_budget],
            "gold_answer": "The Memory Synchronizer retrieves the top two relevant verified facts within a constrained memory token budget of 250 tokens.",
            "gold_citation_ids": [cid_memory_budget],
        },
        {
            "id": "q13",
            "query": "What salience threshold is required to retain an extracted sentence in context compaction?",
            "relevant_chunk_ids": [cid_salience_pruning],
            "gold_answer": "Sentences falling below a salience threshold of 0.15 are pruned as peripheral noise.",
            "gold_citation_ids": [cid_salience_pruning],
        },
        {
            "id": "q14",
            "query": "What explicit bracketed citation format does the Grounded Prompt Synthesizer enforce?",
            "relevant_chunk_ids": [cid_bracketed_format],
            "gold_answer": "The Grounded Prompt Synthesizer enforces mandatory bracketed citations in the format [Doc X, Chunk Y] for every assertion.",
            "gold_citation_ids": [cid_bracketed_format],
        },
        {
            "id": "q15",
            "query": "How does transformer attention weight distribute across prompt sequences?",
            "relevant_chunk_ids": [cid_attention_u],
            "gold_answer": "Transformer attention mechanisms display an innate U-shaped attention distribution where tokens near the beginning and end receive significantly higher cross-attention weights than intermediate tokens.",
            "gold_citation_ids": [cid_attention_u],
        },
        {
            "id": "q16",
            "query": "What token ceiling is enforced on assembled prompt contexts to prevent context dilution?",
            "relevant_chunk_ids": [cid_dilution],
            "gold_answer": "The system architecture enforces a strict 1500 token ceiling on assembled prompts to eliminate context dilution.",
            "gold_citation_ids": [cid_dilution],
        },
    ]
    return cases


def main():
    print("=== Step 1: Setting up Technical Corpus and Indexing ===")
    index_dir = Path("data/indexes")
    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    manager, all_chunks = ingest_corpus(index_dir)

    print("\n=== Step 2: Building Annotated Evaluation Queries ===")
    cases = build_evaluation_cases(all_chunks)
    dataset_path = eval_dir / "queries.jsonl"
    with dataset_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"Generated {len(cases)} benchmark queries saved to {dataset_path}")

    print("\n=== Step 3: Executing End-to-End RAG Benchmark ===")
    output_dir = Path("results/evaluation")
    metadata = run_benchmark(
        dataset=dataset_path,
        output=output_dir,
        index_dir=index_dir,
        generate=True,
        provider="offline",
    )

    print("\n=== Benchmark Completed Successfully ===")
    print(f"Output files written to: {output_dir.resolve()}")
    print("\nSummaries:")
    print(json.dumps(metadata["summaries"], indent=2))


if __name__ == "__main__":
    main()

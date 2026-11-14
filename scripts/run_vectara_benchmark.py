"""Open RAG Benchmark (vectara/open_ragbench) evaluation runner.

Ingests and indexes multi-domain arXiv papers from vectara/open_ragbench,
maps queries to their ground-truth section chunk IDs and reference answers,
and evaluates the RAG pipeline to generate Table II and Table III.
"""

from __future__ import annotations

import json
import os
import re
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

BENCHMARK_DOCS = [
    "2410.14077v2",  # cs.SY (Systems and Control)
    "2401.07294v4",  # stat.ME (Statistics & Multilevel Metamodels)
    "2410.11774v2",  # cs.CV (Computer Vision)
    "2406.17972v3",  # cs.LG / cs.CL (Machine Learning & Language Modeling)
    "2412.10128v2",  # cs.LG / stat.AP (Latent Factor Models)
]


def prepare_and_index_corpus(
    corpus_dir: Path, index_dir: Path, map_cache_file: Path | None = None
) -> dict[tuple[str, int], list[str]]:
    """Parse papers, extract sections, index into IndexManager, and return section->chunks map."""
    if (
        map_cache_file
        and map_cache_file.exists()
        and index_dir.exists()
        and (index_dir / "vector_index.faiss").exists()
    ):
        with open(map_cache_file, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        section_chunks = {
            (k.split(":::")[0], int(k.split(":::")[1])): v for k, v in raw_map.items()
        }
        print(f"Loaded existing index and mapped sections from {map_cache_file}")
        return section_chunks

    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    manager = IndexManager(storage_dir=index_dir)
    section_chunks: dict[tuple[str, int], list[str]] = {}

    total_chunks = 0
    for doc_id in BENCHMARK_DOCS:
        paper_file = corpus_dir / f"{doc_id}.json"
        if not paper_file.exists():
            raise FileNotFoundError(f"Missing corpus file: {paper_file}")

        with open(paper_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        title = data.get("title", "")
        sections = data.get("sections", [])

        full_md = f"# {title}\n\n"
        for s in sections:
            sid = s.get("section_id", 0)
            full_md += (
                f"\n\n## Section {sid}\n[Section: {sid}]\n\n{s.get('text', '')}\n"
            )
            for tbl_name, tbl_md in s.get("tables", {}).items():
                full_md += f"\n\n### {tbl_name}\n\n{tbl_md}\n"

        raw_bytes = full_md.encode("utf-8")
        parsed = default_registry.parse_bytes(
            content=raw_bytes,
            filename=f"{doc_id}.md",
            file_size_bytes=len(raw_bytes),
            document_type="TECHNICAL_SPEC",
        )

        manager.index_document(parsed)
        total_chunks += len(parsed.chunks)

        for c in parsed.chunks:
            match = re.search(r"\[Section:\s*(\d+)\]", c.content)
            if match:
                sid = int(match.group(1))
                section_chunks.setdefault((doc_id, sid), []).append(c.id)

        print(f"Indexed {doc_id} ('{title[:35]}...') -> {len(parsed.chunks)} chunks")

    print(f"\nTotal Indexed Chunks: {total_chunks}")
    print(f"Total Mapped Sections: {len(section_chunks)}")

    if map_cache_file:
        raw_map = {f"{k[0]}:::{k[1]}": v for k, v in section_chunks.items()}
        map_cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(map_cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_map, f, indent=2)

    return section_chunks


def build_evaluation_queries(
    qrels_file: Path,
    queries_file: Path,
    answers_file: Path,
    section_chunks: dict[tuple[str, int], list[str]],
    output_file: Path,
) -> list[dict]:
    """Build evaluation JSONL mapping queries to relevant chunks and gold answers."""
    with open(qrels_file, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(answers_file, "r", encoding="utf-8") as f:
        answers = json.load(f)

    target_docs = set(BENCHMARK_DOCS)
    eval_cases = []

    for qid, target in qrels.items():
        doc_id = target["doc_id"]
        sec_id = target["section_id"]
        if doc_id not in target_docs:
            continue

        relevant_chunks = section_chunks.get((doc_id, sec_id), [])
        if not relevant_chunks:
            # Fallback: check if any chunk from that doc exists
            continue

        q_info = queries.get(qid, {})
        query_text = (
            q_info.get("query", "") if isinstance(q_info, dict) else str(q_info)
        )
        gold_ans = answers.get(qid, "")

        eval_cases.append(
            {
                "id": qid,
                "query": query_text,
                "relevant_chunk_ids": relevant_chunks,
                "gold_answer": gold_ans,
                "gold_citation_ids": relevant_chunks,
                "metadata": {
                    "doc_id": doc_id,
                    "section_id": sec_id,
                    "type": q_info.get("type", "") if isinstance(q_info, dict) else "",
                    "source": q_info.get("source", "")
                    if isinstance(q_info, dict)
                    else "",
                },
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for case in eval_cases:
            f.write(json.dumps(case) + "\n")

    print(f"Generated {len(eval_cases)} evaluation cases -> {output_file}")
    return eval_cases


def main():
    data_dir = Path("data/open_ragbench")
    corpus_dir = data_dir / "pdf/arxiv/corpus"
    qrels_file = data_dir / "pdf/arxiv/qrels.json"
    queries_file = data_dir / "pdf/arxiv/queries.json"
    answers_file = data_dir / "pdf/arxiv/answers.json"

    index_dir = data_dir / "benchmark_index"
    eval_file = data_dir / "benchmark_queries.jsonl"
    results_dir = data_dir / "results"
    map_cache_file = data_dir / "benchmark_section_map.json"

    print("=== Step 1: Indexing Corpus Documents from vectara/open_ragbench ===")
    section_chunks = prepare_and_index_corpus(corpus_dir, index_dir, map_cache_file)

    print("\n=== Step 2: Preparing Benchmark Evaluation Cases ===")
    build_evaluation_queries(
        qrels_file, queries_file, answers_file, section_chunks, eval_file
    )

    print("\n=== Step 3: Executing RAG Benchmark Pipeline ===")
    summary = run_benchmark(
        dataset=eval_file,
        output=results_dir,
        index_dir=index_dir,
        generate=True,
        provider="offline",
    )

    print("\n================== TABLE II: RETRIEVAL RESULTS ==================")
    print(f"{'Method':<26} | {'Recall@10':<10} | {'MRR@10':<10} | {'nDCG@10':<10}")
    print("-" * 65)
    for row in summary["summaries"][:4]:
        method = row["method"]
        rec = (
            f"{row.get('recall_at_10', 0.0):.4f}"
            if row.get("recall_at_10") is not None
            else "N/A"
        )
        mrr = (
            f"{row.get('mrr_at_10', 0.0):.4f}"
            if row.get("mrr_at_10") is not None
            else "N/A"
        )
        ndcg = (
            f"{row.get('ndcg_at_10', 0.0):.4f}"
            if row.get("ndcg_at_10") is not None
            else "N/A"
        )
        print(f"{method:<26} | {rec:<10} | {mrr:<10} | {ndcg:<10}")

    print("\n========= TABLE III: ANSWER QUALITY AND RESOURCE MEASUREMENTS =========")
    print(f"{'Metric':<35} | {'Conventional RAG':<18} | {'Full Pipeline':<18}")
    print("-" * 75)

    conv = next(s for s in summary["summaries"] if s["method"] == "Conventional RAG")
    full = next(s for s in summary["summaries"] if s["method"] == "Full Pipeline")

    qa_conv = f"F1: {conv['qa_f1']:.4f}" if conv.get("qa_f1") is not None else "N/A"
    qa_full = f"F1: {full['qa_f1']:.4f}" if full.get("qa_f1") is not None else "N/A"
    print(f"{'Official QA score (state metric)':<35} | {qa_conv:<18} | {qa_full:<18}")

    cit_conv = (
        f"{conv['citation_precision']:.2f} / {conv['citation_recall']:.2f}"
        if conv.get("citation_precision") is not None
        else "N/A"
    )
    cit_full = (
        f"{full['citation_precision']:.2f} / {full['citation_recall']:.2f}"
        if full.get("citation_precision") is not None
        else "N/A"
    )
    print(f"{'Citation precision / recall':<35} | {cit_conv:<18} | {cit_full:<18}")

    tok_conv = (
        f"{conv['mean_input_tokens']:.1f}"
        if conv.get("mean_input_tokens") is not None
        else "N/A"
    )
    tok_full = (
        f"{full['mean_input_tokens']:.1f}"
        if full.get("mean_input_tokens") is not None
        else "N/A"
    )
    print(f"{'Mean input tokens':<35} | {tok_conv:<18} | {tok_full:<18}")

    med_conv = (
        f"{conv['median_end_to_end_latency_ms']:.2f}"
        if conv.get("median_end_to_end_latency_ms") is not None
        else "N/A"
    )
    med_full = (
        f"{full['median_end_to_end_latency_ms']:.2f}"
        if full.get("median_end_to_end_latency_ms") is not None
        else "N/A"
    )
    print(f"{'Median end-to-end latency (ms)':<35} | {med_conv:<18} | {med_full:<18}")

    p95_conv = (
        f"{conv['p95_end_to_end_latency_ms']:.2f}"
        if conv.get("p95_end_to_end_latency_ms") is not None
        else "N/A"
    )
    p95_full = (
        f"{full['p95_end_to_end_latency_ms']:.2f}"
        if full.get("p95_end_to_end_latency_ms") is not None
        else "N/A"
    )
    print(f"{'P95 end-to-end latency (ms)':<35} | {p95_conv:<18} | {p95_full:<18}")

    ram_mb = summary.get("process_rss_mb")
    ram_str = f"{ram_mb:.1f}" if ram_mb else "N/A"
    print(f"{'Peak RAM (MB)':<35} | {ram_str:<18} | {ram_str:<18}")

    gpu_mb = summary.get("peak_gpu_memory_mb")
    gpu_str = f"{gpu_mb:.1f}" if gpu_mb is not None else "0.0 (CPU)"
    print(f"{'Peak GPU memory (MB)':<35} | {gpu_str:<18} | {gpu_str:<18}")

    idx_mb = summary.get("index_size_mb", 0.0)
    idx_str = f"{idx_mb:.2f}"
    print(f"{'Index size (MB)':<35} | {idx_str:<18} | {idx_str:<18}")


if __name__ == "__main__":
    main()

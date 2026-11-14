"""Reproducible evaluation for retrieval and grounded-answering experiments.

Input is JSONL. Each row must contain ``query`` and ``relevant_chunk_ids``; optional
``gold_answer`` enables token-level QA EM/F1 and optional ``gold_citation_ids`` enables
citation precision/recall. No score is inferred when its annotations are absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

from src.chunking.tokenizer import count_tokens
from src.context.compactor import ContextCompactor
from src.generation.engine import EngineFactory
from src.generation.generator import GroundedGenerator
from src.indexing.manager import IndexManager
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievedCandidate
from src.retrieval.sparse import SparseRetriever


def _normalize(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def qa_scores(prediction: str, reference: str) -> tuple[float, float]:
    pred, gold = _normalize(prediction), _normalize(reference)
    em = float(pred == gold)
    overlap = sum(
        (
            __import__("collections").Counter(pred)
            & __import__("collections").Counter(gold)
        ).values()
    )
    precision = overlap / len(pred) if pred else float(not gold)
    recall = overlap / len(gold) if gold else float(not pred)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return em, f1


def retrieval_metrics(
    ranked: list[str], relevant: set[str], k: int = 10
) -> dict[str, float]:
    top = ranked[:k]
    hits = [1 if item in relevant else 0 for item in top]
    recall = len(set(top) & relevant) / len(relevant) if relevant else math.nan
    reciprocal_rank = next((1.0 / i for i, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / math.log2(i + 1) for i, hit in enumerate(hits, 1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return {
        "recall_at_10": recall,
        "mrr_at_10": reciprocal_rank,
        "ndcg_at_10": dcg / ideal if ideal else math.nan,
    }


def citation_scores(predicted: set[str], gold: set[str]) -> tuple[float, float]:
    precision = len(predicted & gold) / len(predicted) if predicted else float(not gold)
    recall = len(predicted & gold) / len(gold) if gold else math.nan
    return precision, recall


def _candidate(
    index: IndexManager, chunk_id: str, score: float, rank: int
) -> RetrievedCandidate | None:
    record = index.metadata_store.get(chunk_id)
    if not record:
        return None
    return RetrievedCandidate(
        chunk_id=chunk_id,
        content=record.get("content", record.get("text", "")),
        document_id=record.get("document_id", ""),
        chunk_index=record.get("chunk_index", 0),
        metadata=record.get("metadata", {}),
        dense_score=score,
        sparse_score=score,
        rrf_score=score,
    )


def _memory_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def _gpu_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return None


def run_benchmark(
    dataset: Path,
    output: Path,
    index_dir: Path,
    generate: bool = True,
    provider: str | None = None,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Dataset is empty")
    for n, row in enumerate(rows, 1):
        if not isinstance(row.get("query"), str) or not isinstance(
            row.get("relevant_chunk_ids"), list
        ):
            raise ValueError(f"Row {n} requires query and relevant_chunk_ids")

    index = IndexManager(storage_dir=index_dir)
    dense = DenseRetriever(
        index.vector_index, index.metadata_store, index.embedding_engine
    )
    sparse = SparseRetriever(index.bm25_index, index.metadata_store)
    hybrid = HybridRetriever(index_manager=index)
    reranker = CrossEncoderReranker()
    compactor = ContextCompactor()
    engine = EngineFactory.create_engine(provider) if provider else None
    generator = GroundedGenerator(engine=engine)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    methods = (
        "BM25",
        "Dense FAISS",
        "BM25 + Dense (RRF)",
        "Hybrid + Cross-Encoder",
        "Conventional RAG",
        "Full Pipeline",
    )
    result_rows: list[dict[str, Any]] = []
    tracemalloc.start()
    for q in rows:
        query = q["query"]
        relevant = set(map(str, q["relevant_chunk_ids"]))
        pipeline_start = time.perf_counter()
        bm = sparse.retrieve(query, top_k=10)
        dense_start = time.perf_counter()
        dn = dense.retrieve(query, top_k=10)
        dense_retrieval_ms = (time.perf_counter() - dense_start) * 1000
        fused = hybrid.retrieve(query, top_k_fused=20)
        ce = reranker.rerank(query, fused.candidates, top_n=10)
        ranked = {
            "BM25": [x[0] for x in bm],
            "Dense FAISS": [x[0] for x in dn],
            "BM25 + Dense (RRF)": [x.chunk_id for x in fused.candidates[:10]],
            "Hybrid + Cross-Encoder": [x.chunk_id for x in ce.chunks[:10]],
        }
        generation = None
        conventional_answer = ""
        conventional_latency = math.nan
        full_pipeline_latency = math.nan
        conventional_tokens = math.nan
        if generate:
            compacted = compactor.compact(query=query, chunks=ce.chunks)
            generation = generator.generate(query=query, context=compacted)
            full_pipeline_latency = (time.perf_counter() - pipeline_start) * 1000
            baseline_context = "\n\n".join(
                index.metadata_store.get(chunk_id).get("content", "")
                for chunk_id, _ in dn
                if index.metadata_store.get(chunk_id)
            )
            conventional_tokens = count_tokens(baseline_context)
            start = time.perf_counter()
            conventional_answer = generator.engine.generate(
                prompt=f"Context:\n{baseline_context}\n\nQuestion: {query}\nAnswer:",
                system_prompt="Answer the question using the provided context. If it does not contain the answer, say so.",
                temperature=0.0,
                max_tokens=1024,
            )
            conventional_latency = (
                dense_retrieval_ms + (time.perf_counter() - start) * 1000
            )
        rank_by_method = {
            **ranked,
            "Conventional RAG": ranked["Dense FAISS"],
            "Full Pipeline": ranked["Hybrid + Cross-Encoder"],
        }
        for method in methods:
            scores = retrieval_metrics(rank_by_method[method], relevant)
            is_full = method == "Full Pipeline"
            if is_full and generate and compacted and compacted.evidence_items:
                pred_citations = {e.chunk_id for e in compacted.evidence_items}
            elif method == "Conventional RAG" and generate:
                pred_citations = {chunk_id for chunk_id, _ in dn}
            else:
                pred_citations = set(rank_by_method[method])
            citation_precision = citation_recall = math.nan
            gold_citations = set(map(str, q.get("gold_citation_ids", [])))
            if gold_citations:
                citation_precision, citation_recall = citation_scores(
                    pred_citations, gold_citations
                )
            answer = (
                generation.answer
                if is_full and generation
                else (conventional_answer if method == "Conventional RAG" else "")
            )
            em = f1 = math.nan
            if answer and q.get("gold_answer"):
                em, f1 = qa_scores(answer, q["gold_answer"])
            result_rows.append(
                {
                    "query_id": q.get("id", ""),
                    "query": query,
                    "method": method,
                    **scores,
                    "qa_em": em,
                    "qa_f1": f1,
                    "citation_precision": citation_precision,
                    "citation_recall": citation_recall,
                    "answer": answer if method == methods[-1] else "",
                    "latency_ms": fused.execution_time_ms
                    if method.startswith("BM25 + Dense")
                    else math.nan,
                    "generation_latency_ms": full_pipeline_latency
                    if is_full and generation
                    else (
                        conventional_latency
                        if method == "Conventional RAG"
                        else math.nan
                    ),
                    "input_tokens": compacted.total_tokens
                    if is_full and generation
                    else (
                        conventional_tokens
                        if method == "Conventional RAG" and generate
                        else math.nan
                    ),
                    "answer_abstained": generation.abstained
                    if is_full and generation
                    else "",
                }
            )

    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    index_bytes = sum(p.stat().st_size for p in index_dir.glob("*") if p.is_file())
    metrics = [
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "qa_em",
        "qa_f1",
        "citation_precision",
        "citation_recall",
    ]
    summaries = []
    for method in methods:
        subset = [r for r in result_rows if r["method"] == method]
        summary: dict[str, Any] = {"method": method, "queries": len(subset)}
        for metric in metrics:
            vals = [
                r[metric]
                for r in subset
                if isinstance(r[metric], (int, float)) and math.isfinite(r[metric])
            ]
            summary[metric] = statistics.mean(vals) if vals else None
        latencies = [
            r["generation_latency_ms"]
            for r in subset
            if isinstance(r["generation_latency_ms"], (int, float))
            and math.isfinite(r["generation_latency_ms"])
        ]
        summary["median_end_to_end_latency_ms"] = (
            statistics.median(latencies) if latencies else None
        )
        summary["p95_end_to_end_latency_ms"] = (
            sorted(latencies)[
                min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)
            ]
            if latencies
            else None
        )
        token_counts = [
            r["input_tokens"]
            for r in subset
            if isinstance(r["input_tokens"], (int, float))
            and math.isfinite(r["input_tokens"])
        ]
        summary["mean_input_tokens"] = (
            statistics.mean(token_counts) if token_counts else None
        )
        summaries.append(summary)
    paired = []
    rng = random.Random(20261104)
    for metric in metrics:
        by_method = {
            m: {
                r["query_id"] or r["query"]: r[metric]
                for r in result_rows
                if r["method"] == m
            }
            for m in methods
        }
        for baseline in methods[:-1]:
            keys = [
                key
                for key, value in by_method[baseline].items()
                if key in by_method[methods[-1]]
                and isinstance(value, (int, float))
                and isinstance(by_method[methods[-1]][key], (int, float))
                and math.isfinite(value)
                and math.isfinite(by_method[methods[-1]][key])
            ]
            differences = [
                by_method[methods[-1]][key] - by_method[baseline][key] for key in keys
            ]
            if differences:
                boot_means = sorted(
                    statistics.mean(rng.choices(differences, k=len(differences)))
                    for _ in range(2000)
                )
                paired.append(
                    {
                        "metric": metric,
                        "comparison": f"{methods[-1]} - {baseline}",
                        "n_paired_queries": len(differences),
                        "mean_difference": statistics.mean(differences),
                        "std_difference": statistics.stdev(differences)
                        if len(differences) > 1
                        else 0.0,
                        "paired_bootstrap_95ci_low": boot_means[49],
                        "paired_bootstrap_95ci_high": boot_means[1949],
                        "bootstrap_seed": 20261104,
                        "bootstrap_resamples": 2000,
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    metadata = {
        "dataset": str(dataset),
        "queries": len(rows),
        "index_directory": str(index_dir),
        "index_size_mb": index_bytes / (1024 * 1024),
        "process_rss_mb": _memory_mb(),
        "python_peak_traced_mb": peak_python / (1024 * 1024),
        "peak_gpu_memory_mb": _gpu_mb(),
        "gpu_memory_note": "null means CUDA unavailable or PyTorch not installed",
        "qa_note": "Exact match and token F1; available only for generated full-pipeline answers with gold_answer annotations.",
        "citation_note": "Precision/recall compare retrieved chunk IDs against gold_citation_ids annotations.",
        "summaries": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "paper_tables.tex").write_text(
        _latex(summaries, metadata), encoding="utf-8"
    )
    if paired:
        with (output / "paired_comparisons.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    return metadata


def _latex(summaries: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    def fmt(v: Any) -> str:
        return "--" if v is None else f"{v:.3f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Retrieval and answer quality}",
        r"\begin{tabular}{lrrr}",
        r"\toprule Method & Recall@10 & MRR@10 & nDCG@10 \\",
        r"\midrule",
    ]
    for x in summaries:
        lines.append(
            f"{x['method']} & {fmt(x['recall_at_10'])} & {fmt(x['mrr_at_10'])} & {fmt(x['ndcg_at_10'])} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        "% System resource measurements; record workload/hardware alongside these values.",
        f"% Index size (MB): {meta['index_size_mb']:.3f}",
        f"% Process RSS (MB): {meta['process_rss_mb'] if meta['process_rss_mb'] is not None else '--'}",
        f"% Peak GPU memory (MB): {meta['peak_gpu_memory_mb'] if meta['peak_gpu_memory_mb'] is not None else '--'}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Annotated JSONL benchmark"
    )
    parser.add_argument("--index-dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation"))
    parser.add_argument(
        "--no-generate", action="store_true", help="Skip generation and QA metrics"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="offline",
        help="Inference engine provider (offline, gemini, groq, openrouter)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmark(
                args.dataset,
                args.output,
                args.index_dir,
                not args.no_generate,
                args.provider,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Application-facing evaluation service for annotated query sets."""

from __future__ import annotations

import math
import random
import statistics
import threading
import time
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.auth import current_user_id
from src.chunking.tokenizer import count_tokens
from src.context.compactor import default_context_compactor
from src.evaluation.benchmark import citation_scores, qa_scores, retrieval_metrics
from src.generation.engine import EngineFactory
from src.generation.generator import default_grounded_generator
from src.indexing.manager import default_index_manager
from src.reranking.reranker import default_reranker
from src.retrieval.hybrid import default_hybrid_retriever


class EvaluationCase(BaseModel):
    id: str | None = None
    query: str = Field(min_length=1, max_length=4000)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    gold_citation_ids: list[str] = Field(default_factory=list)
    document_id: str | None = None


class EvaluationRunRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1, max_length=100)
    provider: str | None = None
    generate_answers: bool = True
    max_token_budget: int = Field(default=1500, ge=100, le=8000)
    context_budgets: list[int] = Field(
        default_factory=lambda: [500, 1000, 1500], min_length=1, max_length=5
    )

    @model_validator(mode="after")
    def unique_case_ids(self):
        ids = [case.id for case in self.cases if case.id]
        if len(ids) != len(set(ids)):
            raise ValueError("Each query id must be unique within a run")
        if any(budget < 100 or budget > 8000 for budget in self.context_budgets):
            raise ValueError("Context budgets must be between 100 and 8000 tokens")
        if len(set(self.context_budgets)) != len(self.context_budgets):
            raise ValueError("Context budgets must be unique")
        return self


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def _index_size_mb() -> float:
    path = default_index_manager.storage_dir
    return sum(f.stat().st_size for f in path.glob("*") if f.is_file()) / (1024 * 1024)


class _PeakRssMonitor:
    def __init__(self) -> None:
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil

            self._process = psutil.Process()
        except ImportError:
            self._process = None

    def start(self) -> None:
        if self._process is None:
            return
        self.peak_mb = self._process.memory_info().rss / (1024 * 1024)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(0.05):
            try:
                current = self._process.memory_info().rss / (1024 * 1024)
                self.peak_mb = max(self.peak_mb or 0.0, current)
            except Exception:
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


def evaluate(request: EvaluationRunRequest) -> dict[str, Any]:
    monitor = _PeakRssMonitor()
    monitor.start()
    try:
        return _evaluate(request, monitor)
    finally:
        monitor.stop()


def _evaluate(
    request: EvaluationRunRequest, monitor: _PeakRssMonitor
) -> dict[str, Any]:
    retrieval_methods = [
        "BM25",
        "Dense FAISS",
        "BM25 + Dense (RRF)",
        "Hybrid + Cross-Encoder",
    ]
    full_methods = [
        f"Full Pipeline · {budget} tokens" for budget in request.context_budgets
    ]
    methods = [*retrieval_methods, "Conventional RAG", *full_methods]
    results: list[dict[str, Any]] = []
    engine = (
        EngineFactory.create_engine(request.provider)
        if request.provider
        else default_grounded_generator.engine
    )
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    for case_index, case in enumerate(request.cases):
        query_id = case.id or f"q{case_index + 1}"
        query = case.query
        filters = (
            {"document_id": case.document_id, "owner_id": current_user_id()}
            if case.document_id
            else {"owner_id": current_user_id()}
        )
        bm_start = time.perf_counter()
        bm25 = default_hybrid_retriever.sparse_retriever.retrieve(
            query, top_k=10, filters=filters
        )
        bm_ms = (time.perf_counter() - bm_start) * 1000
        dense_start = time.perf_counter()
        dense = default_hybrid_retriever.dense_retriever.retrieve(
            query, top_k=10, filters=filters
        )
        dense_ms = (time.perf_counter() - dense_start) * 1000
        full_started = time.perf_counter()
        fused = default_hybrid_retriever.retrieve(
            query, filters=filters, top_k_fused=20
        )
        reranked = default_reranker.rerank(query, fused.candidates, top_n=10)
        full_retrieval_ms = (time.perf_counter() - full_started) * 1000
        full_outputs: dict[str, dict[str, Any]] = {}
        conventional_answer = None
        conventional_latency = None
        for budget, method in zip(request.context_budgets, full_methods):
            pipeline_started = time.perf_counter()
            compacted = default_context_compactor.compact(
                query, reranked.chunks, budget
            )
            answer = None
            if request.generate_answers:
                answer = default_grounded_generator.generate(
                    query=query, context=compacted, provider=request.provider
                )
            full_outputs[method] = {
                "answer": answer,
                "latency_ms": full_retrieval_ms
                + (time.perf_counter() - pipeline_started) * 1000,
                "input_tokens": compacted.total_tokens,
                "abstained": answer.abstained if answer else None,
            }
        if request.generate_answers:
            baseline_context = "\n\n".join(
                default_index_manager.metadata_store.get(chunk_id)["content"]
                for chunk_id, _ in dense
                if default_index_manager.metadata_store.get(chunk_id)
            )
            baseline_started = time.perf_counter()
            conventional_text = engine.generate(
                prompt=f"Context:\n{baseline_context}\n\nQuestion: {query}\nAnswer:",
                system_prompt="Answer using only the supplied context. If it is insufficient, say so.",
                temperature=0.0,
                max_tokens=1024,
            )
            conventional_latency = (
                dense_ms + (time.perf_counter() - baseline_started) * 1000
            )
            conventional_answer = {
                "answer": conventional_text,
                "latency_ms": conventional_latency,
                "input_tokens": count_tokens(baseline_context),
            }

        ranks = {
            "BM25": [x[0] for x in bm25],
            "Dense FAISS": [x[0] for x in dense],
            "BM25 + Dense (RRF)": [x.chunk_id for x in fused.candidates[:10]],
            "Hybrid + Cross-Encoder": [x.chunk_id for x in reranked.chunks[:10]],
            "Conventional RAG": [x[0] for x in dense],
            **{
                method: [x.chunk_id for x in reranked.chunks[:10]]
                for method in full_methods
            },
        }
        latencies = {
            "BM25": bm_ms,
            "Dense FAISS": dense_ms,
            "BM25 + Dense (RRF)": fused.execution_time_ms,
            "Hybrid + Cross-Encoder": fused.execution_time_ms
            + reranked.execution_time_ms,
            "Conventional RAG": conventional_latency,
            **{method: full_outputs[method]["latency_ms"] for method in full_methods},
        }
        for method in methods:
            retrieval = retrieval_metrics(ranks[method], set(case.relevant_chunk_ids))
            citations = (
                citation_scores(set(ranks[method]), set(case.gold_citation_ids))
                if case.gold_citation_ids
                else (None, None)
            )
            full_output = full_outputs.get(method)
            full_answer = full_output["answer"] if full_output else None
            answer = (
                full_answer.answer
                if full_answer
                else (
                    conventional_answer["answer"]
                    if method == "Conventional RAG" and conventional_answer
                    else None
                )
            )
            qa_em = qa_f1 = None
            if answer is not None and case.gold_answer:
                qa_em, qa_f1 = qa_scores(answer, case.gold_answer)
            results.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "method": method,
                    "retrieved_chunk_ids": ranks[method],
                    **{
                        key: (
                            None
                            if isinstance(value, float) and math.isnan(value)
                            else value
                        )
                        for key, value in retrieval.items()
                    },
                    "qa_em": qa_em,
                    "qa_f1": qa_f1,
                    "citation_precision": citations[0],
                    "citation_recall": citations[1],
                    "answer": answer,
                    "latency_ms": latencies[method],
                    "input_tokens": (
                        full_output["input_tokens"]
                        if full_output
                        else conventional_answer["input_tokens"]
                        if method == "Conventional RAG" and conventional_answer
                        else None
                    ),
                    "answer_abstained": full_answer.abstained if full_answer else None,
                }
            )

    summaries = []
    for method in methods:
        selected = [row for row in results if row["method"] == method]
        summary: dict[str, Any] = {"method": method, "queries": len(request.cases)}
        for metric in (
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "qa_em",
            "qa_f1",
            "citation_precision",
            "citation_recall",
        ):
            values = [row[metric] for row in selected if row[metric] is not None]
            summary[metric] = statistics.mean(values) if values else None
        valid_latencies = [
            row["latency_ms"] for row in selected if row["latency_ms"] is not None
        ]
        summary["median_end_to_end_latency_ms"] = (
            statistics.median(valid_latencies) if valid_latencies else None
        )
        summary["p95_end_to_end_latency_ms"] = (
            sorted(valid_latencies)[math.ceil(0.95 * len(valid_latencies)) - 1]
            if valid_latencies
            else None
        )
        tokens = [
            row["input_tokens"] for row in selected if row["input_tokens"] is not None
        ]
        summary["mean_input_tokens"] = statistics.mean(tokens) if tokens else None
        summaries.append(summary)

    paired = []
    rng = random.Random(20261104)
    indexed = {(row["query_id"], row["method"]): row for row in results}
    for metric in (
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "qa_em",
        "qa_f1",
        "citation_precision",
        "citation_recall",
    ):
        for full_method in full_methods:
            for baseline in ("Conventional RAG", "BM25 + Dense (RRF)"):
                differences = []
            for case_index, case in enumerate(request.cases):
                key = case.id or f"q{case_index + 1}"
                left, right = (
                    indexed[(key, baseline)][metric],
                    indexed[(key, full_method)][metric],
                )
                if left is not None and right is not None:
                    differences.append(right - left)
            if differences:
                observed = statistics.mean(differences)
                bootstrap = sorted(
                    statistics.mean(rng.choices(differences, k=len(differences)))
                    for _ in range(2000)
                )
                extreme = 0
                for _ in range(5000):
                    permuted = statistics.mean(
                        value if rng.random() < 0.5 else -value for value in differences
                    )
                    extreme += abs(permuted) >= abs(observed)
                paired.append(
                    {
                        "metric": metric,
                        "comparison": f"{full_method} - {baseline}",
                        "n_paired_queries": len(differences),
                        "mean_difference": observed,
                        "std_difference": statistics.stdev(differences)
                        if len(differences) > 1
                        else 0.0,
                        "bootstrap_95ci_low": bootstrap[49],
                        "bootstrap_95ci_high": bootstrap[1949],
                        "permutation_p_two_sided": (extreme + 1) / 5001,
                        "bootstrap_resamples": 2000,
                        "permutation_resamples": 5000,
                    }
                )
    try:
        import torch

        gpu_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else None
        )
    except ImportError:
        gpu_mb = None
    return {
        "summaries": summaries,
        "per_query": results,
        "paired_comparisons": paired,
        "resources": {
            "index_size_mb": _index_size_mb(),
            "process_rss_mb": _rss_mb(),
            "process_peak_rss_mb": monitor.peak_mb,
            "peak_gpu_memory_mb": gpu_mb,
        },
        "notes": {
            "qa": "Normalized exact match and token F1; requires gold_answer.",
            "citations": "Chunk-ID overlap with gold_citation_ids; does not establish semantic support.",
            "statistics": "Paired bootstrap 95% CI (2,000 resamples) and two-sided sign-flip permutation p (5,000 resamples); p-values are unadjusted across comparisons.",
            "null": "A null metric indicates missing annotations or unavailable hardware.",
        },
    }

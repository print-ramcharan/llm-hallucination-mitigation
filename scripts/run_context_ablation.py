"""Run matched reranking and context-compaction ablations on a JSONL query set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ABLATION_TOKEN_BUDGET = 1_000_000
_CONDITIONS = (
    (
        "Hybrid + reranking, no compaction",
        False,
        False,
        False,
    ),
    (
        "Hybrid + reranking + sentence pruning",
        True,
        False,
        False,
    ),
    (
        "Hybrid + reranking + pruning + deduplication",
        True,
        True,
        False,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_ablation(dataset: Path, index_dir: Path, output: Path) -> dict[str, Any]:
    """Evaluate the same reranked top-10 candidates with three compaction settings."""
    from src.context.compactor import ContextCompactor
    from src.evaluation.benchmark import (
        citation_scores,
        qa_scores,
        retrieval_metrics,
    )
    from src.generation.engine import EngineFactory
    from src.indexing.manager import IndexManager
    from src.reranking.reranker import CrossEncoderReranker
    from src.retrieval.hybrid import HybridRetriever

    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset}")
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row.get("query"), str) or not isinstance(
            row.get("relevant_chunk_ids"), list
        ):
            raise ValueError(f"Row {row_number} requires query and relevant_chunk_ids")
        if not row["relevant_chunk_ids"]:
            raise ValueError(f"Row {row_number} requires relevant chunk IDs")
        if not isinstance(row.get("gold_answer"), str):
            raise ValueError(f"Row {row_number} requires gold_answer")
        if not isinstance(row.get("gold_citation_ids"), list):
            raise ValueError(f"Row {row_number} requires gold_citation_ids")

    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    index = IndexManager(storage_dir=index_dir)
    hybrid = HybridRetriever(index_manager=index)
    reranker = CrossEncoderReranker()
    compactor = ContextCompactor()
    engine = EngineFactory.create_engine("offline")
    result_rows: list[dict[str, Any]] = []

    for row in rows:
        fused = hybrid.retrieve(row["query"], top_k_fused=20)
        reranked = reranker.rerank(row["query"], fused.candidates, top_n=10)
        for method, pruning, deduplication, reordering in _CONDITIONS:
            context = compactor.compact(
                query=row["query"],
                chunks=reranked.chunks,
                max_token_budget=_ABLATION_TOKEN_BUDGET,
                enable_sentence_pruning=pruning,
                enable_deduplication=deduplication,
                enable_lost_in_middle_reordering=reordering,
            )
            answer = engine.generate(
                prompt=(
                    f"Context:\n{context.formatted_prompt_context}\n\n"
                    f"Question: {row['query']}\nAnswer:"
                ),
                system_prompt=(
                    "Answer the question using the provided context. "
                    "If it does not contain the answer, say so."
                ),
                temperature=0.0,
                max_tokens=1024,
            )
            evidence_ids = [item.chunk_id for item in context.evidence_items]
            retrieval = retrieval_metrics(
                evidence_ids, set(map(str, row["relevant_chunk_ids"]))
            )
            citation_precision, citation_recall = citation_scores(
                set(evidence_ids), set(map(str, row.get("gold_citation_ids", [])))
            )
            exact_match, token_f1 = qa_scores(answer, row["gold_answer"])
            result_rows.append(
                {
                    "query_id": row.get("id", ""),
                    "method": method,
                    **retrieval,
                    "qa_em": exact_match,
                    "qa_f1": token_f1,
                    "citation_precision": citation_precision,
                    "citation_recall": citation_recall,
                    "input_tokens": context.total_tokens,
                    "evidence_count": len(context.evidence_items),
                    "dedup_pruned_count": context.dedup_pruned_count,
                }
            )

    metric_names = (
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "qa_em",
        "qa_f1",
        "citation_precision",
        "citation_recall",
        "input_tokens",
        "evidence_count",
        "dedup_pruned_count",
    )
    summaries = []
    for method, *_ in _CONDITIONS:
        subset = [row for row in result_rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "queries": len(subset),
                **{
                    metric: statistics.mean(row[metric] for row in subset)
                    for metric in metric_names
                },
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_query.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    index_files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(index_dir.iterdir())
        if path.is_file()
    }
    metadata = {
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "queries": len(rows),
        "index_directory": str(index_dir),
        "index_files": index_files,
        "dense_embedding_model": index.embedding_engine.model_name,
        "cross_encoder_model": reranker.engine.model_name,
        "answer_generator": engine.model_name,
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "notes": [
            "All conditions use the same index, query set, reranked top-10 candidates, and relevance labels.",
            "The dataset defines no explicit metadata filters.",
            "A 1,000,000-token cap avoids budget truncation in these component ablations.",
            "Generation uses the deterministic offline synthesizer without sufficiency or claim checks.",
        ],
        "summaries": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data") / "eval" / "queries.jsonl"
    )
    parser.add_argument("--index-dir", type=Path, default=Path("data") / "indexes")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; an existing directory is not overwritten.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_ablation(args.dataset, args.index_dir, args.output),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

# Evaluation protocol

The benchmark runner evaluates the repository's loaded indexes against a query set with chunk-level relevance judgments. It produces per-query CSV, aggregate JSON, a LaTeX retrieval table, and paired per-query bootstrap comparisons (2,000 resamples with a fixed seed). Results are only meaningful when the query set and all annotations are fixed and reported with the paper.

## Prepare annotations

Create a UTF-8 JSONL file, one object per query:

```json
{"id":"q1","query":"What caused the decline?","relevant_chunk_ids":["chunk-id"],"gold_answer":"The decline was caused by ...","gold_citation_ids":["chunk-id"]}
```

`query` and `relevant_chunk_ids` are required. `gold_answer` enables normalized exact match and token F1; `gold_citation_ids` enables citation precision and recall. Gold chunk IDs must come from the indexed corpus. Use multiple independently judged relevant chunks where appropriate. The runner does not manufacture labels, and reports missing annotations as unavailable.

## Run

```powershell
python -m src.evaluation.benchmark --dataset data/eval/queries.jsonl --index-dir data/indexes --output results/evaluation
```

Use `--no-generate` for retrieval-only runs. Generation uses the configured application inference engine and is subject to its model, provider, and network availability. For a controlled paper comparison, pin the same generator, prompt, temperature, hardware, index snapshot, and query order; record provider/model versions and run settings alongside the artifacts. The four retrieval variants are BM25, dense FAISS, RRF fusion, and fusion followed by the project's cross-encoder. Conventional RAG answers use the top dense FAISS passages verbatim with the configured generator; full pipeline answers use hybrid retrieval, cross-encoder ranking, context compaction, and grounded generation. Both answer paths run when generation is enabled, including when gold answers are omitted, so latency remains available. QA metrics remain unavailable without `gold_answer`.

## Run from the application

Start the FastAPI backend and Next.js frontend as described in the README, then open the **Evaluate** tab. Paste or import the same JSON/JSONL annotations. The workspace calls `POST /api/evaluation/run`, which evaluates against the application's live index and pipeline. Select a document in the app to scope cases that do not specify `document_id`. The results view shows aggregate and per-query outputs, resource readings, paired tests, and CSV/LaTeX exports. The endpoint accepts up to 100 cases per request; generation can take time, especially with hosted models.

## Metrics and limitations

Recall@10, MRR@10, and nDCG@10 are computed per query from chunk relevance labels. QA EM/F1 use standard lowercasing, punctuation/article removal, and token overlap; they are not an official dataset-specific QA score unless that is the stated evaluation protocol. Citation precision/recall compare retrieved chunk IDs to `gold_citation_ids`; they do not verify semantic citation support. Input tokens are the compacted evidence context size for the full pipeline. Latency is measured in the run and must be interpreted with hardware, model loading, warm-up, and provider details. Index size is the sum of files in the index directory. Process RSS and Python traced peak are separate memory measures. CUDA peak allocation is reported only when CUDA and PyTorch are available; `null` means unavailable, not zero. Retrieval-only runs do not claim answer latency or QA scores.

Use paired per-query comparisons for statistical inference because every method sees the same query set. The generated `per_query.csv` is the source for paired tests and confidence intervals; report the test, interval method, sample size, and correction for multiple comparisons in the paper. Do not treat chunks or repeated runs as independent queries.

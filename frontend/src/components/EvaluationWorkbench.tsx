"use client";

import React, { useMemo, useState } from "react";
import { Clipboard, Download, FileUp, Play, Table2 } from "lucide-react";
import { EvaluationCase, EvaluationResponse, runApplicationEvaluation } from "@/lib/api";
import { Document } from "@/types/ingestion";

interface Props {
  document: Document | null;
  documentId: string | null;
}

const EXAMPLE = '{"id":"q1","query":"Enter a question answerable from the selected document","relevant_chunk_ids":["paste-a-relevant-chunk-id"],"gold_answer":"Enter the reference answer","gold_citation_ids":["paste-a-supporting-chunk-id"]}';

function parseCases(text: string): EvaluationCase[] {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("Add at least one annotated query first.");
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    parsed = trimmed.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
  }
  const cases = Array.isArray(parsed) ? parsed : [parsed];
  if (!cases.length || cases.length > 100) throw new Error("Provide between 1 and 100 queries.");
  cases.forEach((item, index) => {
    if (!item || typeof item.query !== "string" || !Array.isArray(item.relevant_chunk_ids)) {
      throw new Error(`Query ${index + 1} needs a query string and relevant_chunk_ids array.`);
    }
  });
  return cases as EvaluationCase[];
}

function csvCell(value: unknown): string {
  const text = value == null ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function downloadFile(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function EvaluationWorkbench({ document: selectedDocument, documentId }: Props) {
  const [dataset, setDataset] = useState("");
  const [provider, setProvider] = useState("fallback");
  const [generateAnswers, setGenerateAnswers] = useState(true);
  const [contextBudgets, setContextBudgets] = useState("500,1000,1500");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvaluationResponse | null>(null);
  const [selectedQuery, setSelectedQuery] = useState(0);

  const readyChunks = selectedDocument?.chunks ?? [];
  const currentRows = useMemo(() => result?.per_query.filter((row) => row.query_id === result.per_query[selectedQuery]?.query_id) ?? [], [result, selectedQuery]);

  const addExample = () => setDataset((prev) => prev ? `${prev.trimEnd()}\n${EXAMPLE}` : EXAMPLE);
  const loadFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) setDataset(await file.text());
    event.target.value = "";
  };
  const run = async () => {
    setError(null);
    setResult(null);
    try {
      const cases = parseCases(dataset).map((item) => ({
        ...item,
        document_id: item.document_id || (documentId && documentId !== "all" ? documentId : undefined),
      }));
      const budgets = contextBudgets.split(",").map((value) => Number(value.trim())).filter(Number.isFinite);
      if (!budgets.length || budgets.length > 5 || budgets.some((value) => value < 100 || value > 8000) || new Set(budgets).size !== budgets.length) {
        throw new Error("Enter 1–5 unique context budgets between 100 and 8000 tokens, separated by commas.");
      }
      setRunning(true);
      setResult(await runApplicationEvaluation({
        cases, provider, generate_answers: generateAnswers, context_budgets: budgets,
      }));
      setSelectedQuery(0);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to run the evaluation.");
    } finally {
      setRunning(false);
    }
  };

  const exportCsv = () => {
    if (!result?.per_query.length) return;
    const columns = Object.keys(result.per_query[0]);
    const content = [columns.map(csvCell).join(","), ...result.per_query.map((row) => columns.map((key) => csvCell(row[key])).join(","))].join("\r\n");
    downloadFile("application-evaluation-per-query.csv", content, "text/csv;charset=utf-8");
  };
  const exportLatex = () => {
    if (!result) return;
    const esc = (value: string) => value.replaceAll("&", "\\&").replaceAll("_", "\\_");
    const fmt = (value: unknown) => typeof value === "number" ? value.toFixed(3) : "--";
    const lines = [
      "\\begin{table}[t]", "\\centering", "\\caption{Application evaluation results}",
      "\\begin{tabular}{lrrr}", "\\toprule Method & Recall@10 & MRR@10 & nDCG@10 \\\\", "\\midrule",
      ...result.summaries.map((row) => `${esc(row.method)} & ${fmt(row.recall_at_10)} & ${fmt(row.mrr_at_10)} & ${fmt(row.ndcg_at_10)} \\\\`),
      "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
      "\\begin{table*}[t]", "\\centering", "\\caption{Answer quality and runtime}",
      "\\begin{tabular}{lrrrrrrr}", "\\toprule Method & QA EM & QA F1 & Cite P & Cite R & Median ms & P95 ms & Input tokens \\\\", "\\midrule",
      ...result.summaries.map((row) => `${esc(row.method)} & ${fmt(row.qa_em)} & ${fmt(row.qa_f1)} & ${fmt(row.citation_precision)} & ${fmt(row.citation_recall)} & ${fmt(row.median_end_to_end_latency_ms)} & ${fmt(row.p95_end_to_end_latency_ms)} & ${fmt(row.mean_input_tokens)} \\\\`),
      "\\bottomrule", "\\end{tabular}", "\\end{table*}",
      `% Index size (MB): ${result.resources.index_size_mb.toFixed(3)}`,
      `% Current process RSS (MB): ${result.resources.process_rss_mb ?? "--"}`,
      `% Peak process RSS (MB): ${result.resources.process_peak_rss_mb ?? "--"}`,
      `% Peak GPU memory (MB): ${result.resources.peak_gpu_memory_mb ?? "--"}`,
    ];
    downloadFile("application-evaluation-table.tex", lines.join("\n"), "application/x-tex");
  };

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 sm:p-7">
        <div className="flex items-start gap-3">
          <div className="rounded-xl bg-cyan-500/10 p-3 text-cyan-300"><Table2 className="h-5 w-5" /></div>
          <div>
            <h2 className="text-xl font-semibold text-white">Application evaluation</h2>
            <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-400">Run annotated questions through the live indexes and compare retrieval methods, conventional dense RAG, and the full grounded pipeline. Add judged evidence IDs and reference answers to compute quality metrics.</p>
          </div>
        </div>

        <div className="mt-6 grid gap-5 lg:grid-cols-[minmax(0,1fr)_300px]">
          <div>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <label htmlFor="eval-jsonl" className="text-sm font-medium text-slate-200">Annotated query set (JSON or JSONL)</label>
              <div className="flex gap-2">
                <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:bg-slate-800">
                  <FileUp className="h-3.5 w-3.5" /> Import file<input type="file" accept=".json,.jsonl,.txt,application/json" onChange={loadFile} className="hidden" />
                </label>
                <button onClick={addExample} className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:bg-slate-800">Insert example</button>
              </div>
            </div>
            <textarea id="eval-jsonl" value={dataset} onChange={(event) => setDataset(event.target.value)} spellCheck={false}
              placeholder={EXAMPLE} className="min-h-56 w-full resize-y rounded-xl border border-slate-700 bg-slate-950 p-4 font-mono text-xs leading-5 text-slate-200 outline-none focus:border-cyan-500" />
            <p className="mt-2 text-xs leading-5 text-slate-500">Required per row: <code>query</code> and <code>relevant_chunk_ids</code>. Add <code>gold_answer</code> for normalized QA exact match/F1 and <code>gold_citation_ids</code> for citation ID precision/recall. Maximum 100 queries per run.</p>
          </div>

          <aside className="space-y-4 rounded-xl border border-slate-800 bg-slate-950/60 p-4">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-300">Generation provider</label>
              <select value={provider} onChange={(event) => setProvider(event.target.value)} disabled={!generateAnswers} className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200">
                <option value="fallback">Application default</option><option value="offline">Offline grounded</option><option value="ollama">Ollama</option><option value="openai">OpenAI</option><option value="gemini">Gemini</option><option value="groq">Groq</option><option value="openrouter">OpenRouter</option><option value="claude">Claude</option>
              </select>
            </div>
            <label className="flex items-start gap-2 text-xs leading-5 text-slate-300"><input type="checkbox" checked={generateAnswers} onChange={(event) => setGenerateAnswers(event.target.checked)} className="mt-1 accent-cyan-500" />Generate conventional and full-pipeline answers for QA and latency metrics</label>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-300">Context budgets (tokens)</label>
              <input type="text" value={contextBudgets} onChange={(event) => setContextBudgets(event.target.value)} placeholder="500,1000,1500" className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200" />
              <p className="mt-1 text-[11px] text-slate-500">Run the full pipeline at each length (up to 5).</p>
            </div>
            <button onClick={run} disabled={running} className="flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-600 px-4 py-3 text-sm font-semibold text-white hover:bg-cyan-500 disabled:cursor-wait disabled:opacity-60">
              <Play className="h-4 w-4" />{running ? "Evaluating queries…" : "Run evaluation"}
            </button>
            {running && <p className="text-xs leading-5 text-slate-400">The first run may take longer while retrieval and reranking models initialize.</p>}
          </aside>
        </div>

        {readyChunks.length > 0 && <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/50 p-4">
          <div className="mb-2 flex items-center justify-between gap-2"><p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Chunk IDs in selected document · {readyChunks.length}</p><span className="text-xs text-slate-500">Click to copy an ID</span></div>
          <div className="flex max-h-28 flex-wrap gap-2 overflow-y-auto">{readyChunks.slice(0, 120).map((chunk) => <button key={chunk.id} title={chunk.content.slice(0, 180)} onClick={() => navigator.clipboard?.writeText(chunk.id)} className="inline-flex items-center gap-1 rounded-md border border-slate-800 bg-slate-900 px-2 py-1 font-mono text-[10px] text-cyan-300 hover:border-cyan-700"><Clipboard className="h-3 w-3" />{chunk.id}</button>)}</div>
        </div>}
        {error && <div role="alert" className="mt-4 rounded-lg border border-rose-500/30 bg-rose-950/30 px-4 py-3 text-sm text-rose-300">{error}</div>}
      </section>

      {result && <>
        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[{ label: "Index size", value: `${result.resources.index_size_mb.toFixed(2)} MB` }, { label: "Process RAM", value: result.resources.process_rss_mb == null ? "Unavailable" : `${result.resources.process_rss_mb.toFixed(1)} MB` }, { label: "Peak process RAM", value: result.resources.process_peak_rss_mb == null ? "Unavailable" : `${result.resources.process_peak_rss_mb.toFixed(1)} MB` }, { label: "Peak GPU memory", value: result.resources.peak_gpu_memory_mb == null ? "Unavailable" : `${result.resources.peak_gpu_memory_mb.toFixed(1)} MB` }].map((metric) => <div key={metric.label} className="rounded-xl border border-slate-800 bg-slate-900/60 p-4"><p className="text-xs text-slate-400">{metric.label}</p><p className="mt-1 text-xl font-semibold text-white">{metric.value}</p></div>)}
        </section>
        <section className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/60">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 p-5"><div><h3 className="font-semibold text-white">Aggregate results</h3><p className="mt-1 text-xs text-slate-400">Null values mean the run lacked the required labels or hardware.</p></div><div className="flex gap-2"><button onClick={exportCsv} className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:bg-slate-800"><Download className="h-3.5 w-3.5" />Per-query CSV</button><button onClick={exportLatex} className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:bg-slate-800"><Download className="h-3.5 w-3.5" />IEEE LaTeX</button></div></div>
          <div className="overflow-x-auto"><table className="w-full min-w-[1100px] text-left text-xs"><thead className="bg-slate-950/70 text-slate-400"><tr>{["Method", "Recall@10", "MRR@10", "nDCG@10", "QA EM", "QA F1", "Citation P", "Citation R", "Median ms", "P95 ms", "Input tokens"].map((x) => <th key={x} className="px-3 py-3 font-medium">{x}</th>)}</tr></thead><tbody>{result.summaries.map((row) => <tr key={row.method} className={`border-t border-slate-800 ${row.method === "Full Pipeline" || row.method === "Conventional RAG" ? "bg-cyan-950/10" : ""}`}><td className="whitespace-nowrap px-3 py-3 font-medium text-slate-200">{row.method}</td>{["recall_at_10", "mrr_at_10", "ndcg_at_10", "qa_em", "qa_f1", "citation_precision", "citation_recall", "median_end_to_end_latency_ms", "p95_end_to_end_latency_ms", "mean_input_tokens"].map((key) => <td key={key} className="px-3 py-3 tabular-nums text-slate-300">{row[key] == null ? "—" : Number(row[key]).toFixed(3)}</td>)}</tr>)}</tbody></table></div>
        </section>
        {result.paired_comparisons.length > 0 && <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h3 className="font-semibold text-white">Paired query differences</h3><p className="mb-4 mt-1 text-xs text-slate-400">Positive differences favor the full pipeline. The CI is a paired bootstrap interval; p uses a two-sided paired sign-flip permutation test.</p><div className="overflow-x-auto"><table className="w-full min-w-[900px] text-left text-xs"><thead className="text-slate-400"><tr><th className="px-3 py-2">Metric</th><th className="px-3 py-2">Comparison</th><th className="px-3 py-2">Paired n</th><th className="px-3 py-2">Mean difference</th><th className="px-3 py-2">SD</th><th className="px-3 py-2">95% CI</th><th className="px-3 py-2">p</th></tr></thead><tbody>{result.paired_comparisons.map((row, i) => <tr key={`${row.metric}-${i}`} className="border-t border-slate-800"><td className="px-3 py-2 text-slate-200">{row.metric}</td><td className="px-3 py-2 text-slate-300">{row.comparison}</td><td className="px-3 py-2 text-slate-300">{row.n_paired_queries}</td><td className="px-3 py-2 text-slate-300">{Number(row.mean_difference).toFixed(3)}</td><td className="px-3 py-2 text-slate-300">{Number(row.std_difference).toFixed(3)}</td><td className="px-3 py-2 text-slate-300">[{Number(row.bootstrap_95ci_low).toFixed(3)}, {Number(row.bootstrap_95ci_high).toFixed(3)}]</td><td className="px-3 py-2 text-slate-300">{Number(row.permutation_p_two_sided).toFixed(4)}</td></tr>)}</tbody></table></div></section>}
        <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h3 className="font-semibold text-white">Per-query answers and retrieval</h3><p className="mt-1 text-xs text-slate-400">Inspect each system output and its answer or abstention.</p></div><select value={selectedQuery} onChange={(event) => setSelectedQuery(Number(event.target.value))} className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200">{Array.from(new Set(result.per_query.map((row) => row.query_id))).map((id, index) => <option key={`${id}-${index}`} value={index}>{String(id)}</option>)}</select></div><p className="mb-4 rounded-lg bg-slate-950 p-3 text-sm text-slate-200">{result.per_query[selectedQuery]?.query}</p><div className="grid gap-3 lg:grid-cols-2">{currentRows.map((row) => <article key={row.method} className="rounded-xl border border-slate-800 bg-slate-950/60 p-4"><div className="flex justify-between gap-2"><h4 className="text-sm font-semibold text-cyan-200">{row.method}</h4><span className="text-xs text-slate-500">{row.latency_ms == null ? "" : `${Number(row.latency_ms).toFixed(1)} ms`}</span></div><p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-300">{row.answer || "No answer generated for this retrieval-only run."}</p><p className="mt-3 text-[11px] text-slate-500">Retrieved chunks: {row.retrieved_chunk_ids?.join(", ") || "not available"}</p></article>)}</div></section>
        <p className="text-xs leading-5 text-slate-500">{result.notes.citations} {result.notes.qa} {result.notes.statistics}</p>
      </>}
    </div>
  );
}

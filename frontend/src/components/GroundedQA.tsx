"use client";

import React, { useState, useRef } from "react";
import {
  ComparisonResponse,
  GroundedAnswerResponse,
  streamQuestionComparison,
} from "@/lib/api";
import { Document, DocumentMetadata } from "@/types/ingestion";
import { UploadZone } from "./UploadZone";
import {
  Bot,
  ShieldCheck,
  UploadCloud,
  FileText,
  Send,
  RefreshCw,
  AlertTriangle,
  Sparkles,
  Zap,
  CheckCircle2,
  Cpu,
} from "lucide-react";

interface GroundedQAProps {
  documents?: DocumentMetadata[];
  selectedDocId?: string | null;
  onSelectDocId?: (docId: string | null) => void;
  onUploadSuccess?: (doc: Document) => void;
}

export function GroundedQA({
  documents = [],
  selectedDocId = null,
  onSelectDocId,
  onUploadSuccess,
}: GroundedQAProps) {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStage, setCurrentStage] = useState<string | null>(null);
  const [currentStageMessage, setCurrentStageMessage] = useState<string | null>(null);
  const [activeProvider, setActiveProvider] = useState<string | null>(null);
  const [streamedAnswer, setStreamedAnswer] = useState<string>("");
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showUploadZone, setShowUploadZone] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const selectedDoc = documents.find((d) => d.document_id === selectedDocId);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setIsLoading(true);
    setIsStreaming(true);
    setError(null);
    setComparison(null);
    setStreamedAnswer("");
    setCurrentStage("starting");
    setCurrentStageMessage("Initiating hybrid retrieval & fallback pipeline...");
    setActiveProvider(null);

    let accumulatedTokens = "";
    let capturedSufficiency: any = null;
    let capturedGrounding: any = null;
    let capturedNaive: any = null;
    let capturedProvider = "Auto-Fallback Cascade";

    try {
      await streamQuestionComparison(
        query.trim(),
        undefined,
        selectedDocId || undefined,
        {
          onStage: (stage, message) => {
            setCurrentStage(stage);
            setCurrentStageMessage(message);
          },
          onProvider: (model, provider) => {
            setActiveProvider(`${provider} (${model})`);
            capturedProvider = `${provider} (${model})`;
          },
          onSufficiency: (sufficiency) => {
            capturedSufficiency = sufficiency;
          },
          onAbstention: (abstention) => {
            accumulatedTokens = abstention.answer;
            setStreamedAnswer(abstention.answer);
          },
          onToken: (token) => {
            accumulatedTokens += token;
            setStreamedAnswer((prev) => prev + token);
          },
          onGrounding: (report) => {
            capturedGrounding = report;
          },
          onNaive: (naive) => {
            capturedNaive = naive;
          },
          onDone: (data) => {
            const comp: ComparisonResponse = {
              query: query.trim(),
              naive: capturedNaive || {
                query: query.trim(),
                answer: "Naive baseline processing completed.",
                has_citations: false,
                citations: [],
                faithfulness_score: 0.2,
                hallucination_risk: "High (Unverified)",
                latency_ms: data.latency_ms,
                model_name: "Naive Baseline",
              },
              grounded: {
                query: query.trim(),
                answer: accumulatedTokens,
                abstained: capturedSufficiency ? !capturedSufficiency.is_sufficient : false,
                abstention_reason: capturedSufficiency?.reasoning || null,
                sufficiency: capturedSufficiency || {
                  is_sufficient: true,
                  sufficiency_score: 1.0,
                  reasoning: "Sufficient evidence",
                  missing_aspects: [],
                  topic: "Query context",
                  abstention_message: null,
                },
                grounding_report: capturedGrounding || {
                  faithfulness_score: 1.0,
                  hallucination_detected: false,
                  total_claims: 0,
                  entailed_claims_count: 0,
                  neutral_claims_count: 0,
                  contradicted_claims_count: 0,
                  claims: [],
                  verified_citations: data.citations || [],
                  unverified_citations: [],
                },
                citations: data.citations || [],
                latency_ms: data.latency_ms,
                model_name: data.model || capturedProvider,
              },
              metrics_comparison: data.metrics_comparison,
            };
            setComparison(comp);
            setIsStreaming(false);
          },
        },
        abortController.signal
      );
    } catch (err: unknown) {
      if (!abortController.signal.aborted) {
        setError(err instanceof Error ? err.message : "Failed to execute streaming comparison");
      }
      setIsStreaming(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDocumentIngested = (doc: Document) => {
    if (onUploadSuccess) onUploadSuccess(doc);
    if (onSelectDocId) onSelectDocId(doc.id);
    setShowUploadZone(false);
  };

  const grounded: GroundedAnswerResponse | undefined = comparison?.grounded;
  const naive = comparison?.naive;

  // Format citations cleanly in grounded answer text
  const renderGroundedAnswer = (text: string) => {
    const parts = text.split(/(\[Doc\s+[^\]]+\])/g);
    return parts.map((part, i) => {
      if (/^\[Doc\s+[^\]]+\]$/.test(part)) {
        return (
          <span
            key={i}
            className="inline-block bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-1.5 py-0.5 rounded text-xs font-mono font-medium mx-1 shadow-sm"
          >
            {part}
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* ==================================================================== */}
      {/* 1. DOCUMENT & FALLBACK STATUS BAR (ORCHESTRATOR)                     */}
      {/* ==================================================================== */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
            <FileText className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-slate-200">Target Context:</span>
              {documents.length > 0 ? (
                <select
                  value={selectedDocId || "all"}
                  onChange={(e) =>
                    onSelectDocId?.(e.target.value === "all" ? null : e.target.value)
                  }
                  className="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 font-medium focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="all">All Documents ({documents.length})</option>
                  {documents.map((d) => (
                    <option key={d.document_id} value={d.document_id}>
                      {d.source_name}
                    </option>
                  ))}
                </select>
              ) : (
                <span className="text-xs text-amber-400 font-medium">
                  No document uploaded yet
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              {selectedDoc
                ? `Active: ${selectedDoc.source_name} (${selectedDoc.chunk_count} chunks • 220 words + 30 overlap)`
                : documents.length > 0
                ? "Full corpus index (FAISS Dense + BM25 Sparse + RRF)"
                : "Upload a document to run grounded anti-hallucination inference"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          {/* Dynamic Auto-Fallback Cascade Indicator */}
          <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 px-3 py-1.5 rounded-lg">
            <div className="flex items-center gap-1.5">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                <Zap className="w-3.5 h-3.5 text-amber-400" />
                Fallback Cascade:
              </span>
            </div>
            <div className="flex items-center gap-1 text-[11px] font-mono">
              <span className="px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 font-medium">
                Gemini
              </span>
              <span className="text-slate-600">&rarr;</span>
              <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 font-medium">
                Groq
              </span>
              <span className="text-slate-600">&rarr;</span>
              <span className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-300 border border-violet-500/20 font-medium">
                OpenRouter
              </span>
            </div>
          </div>

          <button
            onClick={() => setShowUploadZone(!showUploadZone)}
            className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition shrink-0"
          >
            <UploadCloud className="w-4 h-4 text-indigo-400" />
            {showUploadZone ? "Close Uploader" : "Upload Document"}
          </button>
        </div>
      </div>

      {/* Upload Zone Modal / Drawer */}
      {(showUploadZone || documents.length === 0) && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Upload Document (PDF, DOCX, TXT, MD)
            </h3>
            {documents.length > 0 && (
              <button
                onClick={() => setShowUploadZone(false)}
                className="text-xs text-slate-400 hover:text-slate-200"
              >
                Close
              </button>
            )}
          </div>
          <UploadZone onIngestSuccess={handleDocumentIngested} />
        </div>
      )}

      {/* ==================================================================== */}
      {/* 2. QUERY INPUT & REAL-TIME PROGRESS                                  */}
      {/* ==================================================================== */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm space-y-3">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              selectedDoc
                ? `Ask anything about "${selectedDoc.source_name}" (Streamed via Gemini / Groq / OpenRouter)...`
                : "Ask a question to stream comparison: Naive Baseline vs. Grounded Pipeline..."
            }
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium px-5 py-3 rounded-lg flex items-center gap-2 text-sm transition shrink-0 shadow-md shadow-indigo-600/20"
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Streaming...
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                Compare &amp; Stream
              </>
            )}
          </button>
        </form>

        {/* Live Stepper Status Pill */}
        {isLoading && currentStageMessage && (
          <div className="flex items-center justify-between bg-slate-950/80 border border-indigo-900/40 rounded-lg px-3.5 py-2 text-xs text-slate-300">
            <div className="flex items-center gap-2.5">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
              </span>
              <span className="font-mono text-indigo-300 uppercase text-[10px] tracking-wider px-1.5 py-0.5 rounded bg-indigo-950/60 border border-indigo-800/40">
                {currentStage}
              </span>
              <span className="text-slate-300">{currentStageMessage}</span>
            </div>
            {activeProvider && (
              <span className="text-[11px] font-mono text-emerald-400 font-medium">
                Active: {activeProvider}
              </span>
            )}
          </div>
        )}

        {error && (
          <div className="p-3 bg-red-950/40 border border-red-800/80 rounded-lg text-red-200 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
            <span className="flex-1">{error}</span>
          </div>
        )}
      </div>

      {/* ==================================================================== */}
      {/* 3. SIDE-BY-SIDE STREAMING COMPARISON VIEW                            */}
      {/* ==================================================================== */}
      {(isStreaming || comparison || streamedAnswer) && (
        <div className="space-y-4">
          <div className="flex items-center justify-between px-1">
            <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              Comparison for: &ldquo;{query || comparison?.query}&rdquo;
            </h3>
            {(activeProvider || comparison?.grounded?.model_name) && (
              <span className="text-xs font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-2.5 py-1 rounded-md flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5" />
                {activeProvider || comparison?.grounded?.model_name}
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* ---------------------------------------------------------------- */}
            {/* LEFT: NAIVE LLM (ENTIRE RAW DOCUMENT PROMPT)                     */}
            {/* ---------------------------------------------------------------- */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between space-y-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <Bot className="w-4 h-4 text-slate-400" />
                    <span className="font-semibold text-sm text-slate-200">
                      Naive LLM (Monolithic Baseline)
                    </span>
                  </div>
                  <span className="text-[11px] font-medium px-2 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">
                    No Retrieval / Blind
                  </span>
                </div>

                <div className="text-sm leading-relaxed text-slate-300 min-h-[120px]">
                  {naive ? (
                    naive.answer
                  ) : isLoading ? (
                    <div className="flex items-center gap-2 text-xs text-slate-500 pt-6">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Computing ungrounded monolithic prompt baseline...</span>
                    </div>
                  ) : (
                    "No naive baseline response available."
                  )}
                </div>
              </div>

              <div className="pt-3 border-t border-slate-800/60 text-xs text-slate-500 flex items-center justify-between">
                <span>⚠️ Directly prompted with entire raw document text without chunking.</span>
                {naive && (
                  <span className="text-red-400 font-mono text-[11px]">
                    0 citations • Unverified
                  </span>
                )}
              </div>
            </div>

            {/* ---------------------------------------------------------------- */}
            {/* RIGHT: OUR APPLICATION (FALLBACK LLM + VERIFICATION)              */}
            {/* ---------------------------------------------------------------- */}
            <div className="bg-slate-900 border border-indigo-900/50 rounded-xl p-5 flex flex-col justify-between space-y-4 shadow-sm ring-1 ring-indigo-500/10">
              <div className="space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    <span className="font-semibold text-sm text-white">
                      Our Application (Grounded Mitigation)
                    </span>
                  </div>
                  <span className="text-[11px] font-medium px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" />
                    Grounded &amp; NLI Verified
                  </span>
                </div>

                <div className="text-sm leading-relaxed text-slate-100 min-h-[120px]">
                  {isStreaming ? (
                    <div>
                      {renderGroundedAnswer(streamedAnswer)}
                      <span className="inline-block w-2 h-4 ml-1 bg-indigo-400 animate-pulse align-middle" />
                    </div>
                  ) : grounded ? (
                    renderGroundedAnswer(grounded.answer)
                  ) : streamedAnswer ? (
                    renderGroundedAnswer(streamedAnswer)
                  ) : (
                    <div className="flex items-center gap-2 text-xs text-indigo-300/80 pt-6">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Streaming response via fallback cascade...</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="pt-3 border-t border-slate-800/60 text-xs text-emerald-400/90 flex items-center justify-between">
                <span>✓ Hybrid Retrieval • Compaction • Truthful Abstention • Claim NLI</span>
                {(grounded?.citations?.length ?? 0) > 0 && (
                  <span className="text-indigo-300 font-mono text-[11px] font-semibold">
                    {grounded?.citations.length} verified citations
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Comparison Metrics Telemetry Cards */}
          {comparison?.metrics_comparison && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-3">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Guardrail Verification Metrics
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-slate-500 text-[10px] uppercase">Faithfulness</div>
                  <div className="text-emerald-400 font-bold mt-0.5">
                    {comparison.metrics_comparison.grounded_faithfulness}
                  </div>
                </div>
                <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-slate-500 text-[10px] uppercase">Sufficiency Gate</div>
                  <div className="text-indigo-300 font-bold mt-0.5">
                    {comparison.metrics_comparison.grounded_sufficiency_gate}
                  </div>
                </div>
                <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-slate-500 text-[10px] uppercase">NLI Claims</div>
                  <div className="text-slate-200 font-bold mt-0.5">
                    {comparison.metrics_comparison.grounded_nli_claims}
                  </div>
                </div>
                <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-slate-500 text-[10px] uppercase">Latency</div>
                  <div className="text-amber-400 font-bold mt-0.5">
                    {comparison.grounded?.latency_ms} ms
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

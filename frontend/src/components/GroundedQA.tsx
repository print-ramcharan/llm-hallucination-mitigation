"use client";

import React, { useEffect, useRef, useState } from "react";
import { Loader2, Send, UploadCloud } from "lucide-react";
import {
  CompactedEvidence,
  createSession,
  fetchSessions,
  NaiveGenerationResponse,
  streamQuestionComparison,
} from "@/lib/api";
import { UploadZone } from "@/components/UploadZone";
import { Document, DocumentMetadata } from "@/types/ingestion";

interface GroundedQAProps {
  documents: DocumentMetadata[];
  selectedDocId: string | null;
  onSelectDocId: (documentId: string | null) => void;
  onUploadSuccess: (document: Document) => void;
  onContext: (query: string, evidence: CompactedEvidence[]) => void;
}

export const GroundedQA: React.FC<GroundedQAProps> = ({
  documents,
  selectedDocId,
  onSelectDocId,
  onUploadSuccess,
  onContext,
}) => {
  const [query, setQuery] = useState("");
  const [sessionId, setSessionId] = useState<string>();
  const [naiveAnswer, setNaiveAnswer] = useState<NaiveGenerationResponse>();
  const [groundedAnswer, setGroundedAnswer] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isUploaderOpen, setIsUploaderOpen] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setIsUploaderOpen(documents.length === 0);
  }, [documents.length]);

  useEffect(() => {
    let cancelled = false;
    const initializeSession = async () => {
      try {
        const sessions = await fetchSessions();
        if (cancelled) return;
        if (sessions.length > 0) {
          setSessionId(sessions[0].session_id);
          return;
        }
        const session = await createSession("Default conversation");
        if (!cancelled) setSessionId(session.session_id);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not initialize a conversation.");
        }
      }
    };
    void initializeSession();
    return () => {
      cancelled = true;
      controllerRef.current?.abort();
    };
  }, []);

  const handleDocumentIngested = (document: Document) => {
    onUploadSuccess(document);
    setIsUploaderOpen(false);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const question = query.trim();
    if (!question || isLoading) return;

    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setIsLoading(true);
    setIsStreaming(true);
    setError("");
    setStatus("Preparing answers...");
    setNaiveAnswer(undefined);
    setGroundedAnswer("");
    onContext(question, []);

    try {
      await streamQuestionComparison(
        question,
        sessionId,
        selectedDocId ?? undefined,
        {
          onStage: (_stage, message) => setStatus(message),
          onNaive: setNaiveAnswer,
          onContext: (evidence) => onContext(question, evidence),
          onToken: (token) => setGroundedAnswer((current) => current + token),
          onAbstention: (abstention) => setGroundedAnswer(abstention.answer),
          onDone: () => setStatus(""),
        },
        controller.signal,
      );
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(err instanceof Error ? err.message : "The comparison could not be completed.");
      }
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setIsLoading(false);
        setIsStreaming(false);
        setStatus("");
      }
    }
  };

  return (
    <section className="space-y-5">
      <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-white">Documents</h2>
            <p className="mt-1 text-xs text-slate-400">
              Choose a document or ask across the full workspace.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setIsUploaderOpen((open) => !open)}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-xs font-medium text-slate-200 hover:bg-slate-800"
          >
            <UploadCloud className="h-4 w-4 text-indigo-400" />
            {isUploaderOpen ? "Close upload" : "Upload document"}
          </button>
        </div>

        {isUploaderOpen && (
          <div className="mt-4">
            <UploadZone onIngestSuccess={handleDocumentIngested} />
          </div>
        )}

        <label className="mt-4 block text-xs font-medium text-slate-300" htmlFor="qa-document">
          Search scope
        </label>
        <select
          id="qa-document"
          value={selectedDocId ?? ""}
          onChange={(event) => onSelectDocId(event.target.value || null)}
          className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100"
        >
          <option value="">All documents</option>
          {documents.map((document) => (
            <option key={document.document_id} value={document.document_id}>
              {document.source_name}
            </option>
          ))}
        </select>

        <form onSubmit={handleSubmit} className="mt-4 flex gap-2">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask a question about your documents"
            aria-label="Question"
            className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-indigo-600 px-4 py-3 text-sm font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            Ask
          </button>
        </form>

        {status && <p className="mt-3 text-xs text-slate-400">{status}</p>}
        {error && (
          <p role="alert" className="mt-3 rounded-lg border border-rose-900 bg-rose-950/40 p-3 text-sm text-rose-300">
            {error}
          </p>
        )}
      </div>

      {(naiveAnswer || groundedAnswer || isLoading) && (
        <div className="grid gap-4 md:grid-cols-2">
          <AnswerCard
            title="Direct LLM response"
            text={naiveAnswer?.answer ?? ""}
            loading={isLoading && !naiveAnswer}
          />
          <AnswerCard
            title="Hybrid retrieval response"
            text={groundedAnswer}
            loading={isLoading && !groundedAnswer}
            streaming={isStreaming && Boolean(groundedAnswer)}
          />
        </div>
      )}
    </section>
  );
};

function AnswerCard({
  title,
  text,
  loading,
  streaming = false,
}: {
  title: string;
  text: string;
  loading: boolean;
  streaming?: boolean;
}) {
  return (
    <article className="min-h-48 rounded-xl border border-slate-800 bg-slate-900 p-5">
      <h3 className="border-b border-slate-800 pb-3 text-sm font-semibold text-slate-100">
        {title}
      </h3>
      <div className="whitespace-pre-wrap pt-4 text-sm leading-6 text-slate-300">
        {text || (loading ? <span className="text-slate-500">Generating response…</span> : null)}
        {streaming && <span className="ml-1 inline-block h-4 w-1 animate-pulse bg-indigo-400 align-middle" />}
      </div>
    </article>
  );
}

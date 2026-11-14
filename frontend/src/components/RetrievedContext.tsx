"use client";

import React from "react";
import { CompactedEvidence } from "@/lib/api";

export function RetrievedContext({
  query,
  evidence,
}: {
  query: string;
  evidence: CompactedEvidence[];
}) {
  return (
    <section className="space-y-4">
      <header>
        <h2 className="text-lg font-semibold text-white">Chunks passed to the LLM</h2>
        <p className="mt-1 text-sm text-slate-400">
          The compacted evidence segments included in the hybrid response prompt.
        </p>
      </header>

      {!query ? (
        <p className="rounded-xl border border-dashed border-slate-700 p-8 text-center text-sm text-slate-400">
          Ask a question to see the chunks used for its response.
        </p>
      ) : (
        <>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Question</p>
            <p className="mt-2 text-sm text-slate-200">{query}</p>
          </div>

          {evidence.length === 0 ? (
            <p className="rounded-xl border border-slate-800 bg-slate-900 p-5 text-sm text-slate-400">
              No evidence chunks were passed to the LLM for this question.
            </p>
          ) : (
            <ol className="space-y-3">
              {evidence.map((item) => (
                <li
                  key={`${item.document_id}:${item.chunk_id}:${item.reorder_position}`}
                  className="rounded-xl border border-slate-800 bg-slate-900 p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-mono text-xs font-semibold text-indigo-300">
                      {item.citation_tag}
                    </span>
                    <span className="text-xs text-slate-500">
                      Prompt position {item.reorder_position} · {item.token_count} tokens
                    </span>
                  </div>
                  <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-200">
                    {item.extracted_text}
                  </p>
                </li>
              ))}
            </ol>
          )}
        </>
      )}
    </section>
  );
}

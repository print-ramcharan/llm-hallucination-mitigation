"use client";

import React, { useState } from "react";
import {
  FileText,
  Copy,
  Check,
  BookOpen,
  Boxes,
  FileDigit,
  Sparkles,
  Calendar,
  ChevronDown,
  ChevronUp,
  Tag,
  Code2,
} from "lucide-react";
import { Document } from "@/types/ingestion";

interface DocumentInspectorProps {
  document: Document | null;
  isLoading: boolean;
}

export const DocumentInspector: React.FC<DocumentInspectorProps> = ({
  document,
  isLoading,
}) => {
  const [activeTab, setActiveTab] = useState<"brief" | "chunks" | "full_text">("brief");
  const [copied, setCopied] = useState(false);
  const [copiedChunkId, setCopiedChunkId] = useState<string | null>(null);
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);

  if (isLoading) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-8 shadow-xl flex flex-col items-center justify-center min-h-[420px]">
        <div className="h-8 w-8 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin mb-3" />
        <p className="text-sm font-medium text-slate-300">Loading document details...</p>
      </div>
    );
  }

  if (!document) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-8 shadow-xl flex flex-col items-center justify-center min-h-[420px] text-center">
        <div className="h-14 w-14 rounded-2xl bg-slate-950 border border-slate-800 flex items-center justify-center text-slate-500 mb-4 shadow-inner">
          <BookOpen className="h-7 w-7" />
        </div>
        <h3 className="text-base font-semibold text-slate-200">No Document Selected</h3>
        <p className="text-xs text-slate-400 max-w-sm mt-1.5 leading-relaxed">
          Select a document from the list or upload a new file to view its brief, metadata, and chunk breakdown.
        </p>
      </div>
    );
  }

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyChunk = (chunkId: string, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedChunkId(chunkId);
    setTimeout(() => setCopiedChunkId(null), 2000);
  };

  const formatBytes = (bytes: number) => {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const formatDate = (isoString?: string) => {
    if (!isoString) return "Recently";
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return isoString;
    }
  };

  const chunks = document.chunks || [];
  const chunksCount = document.metadata.chunk_count || chunks.length;
  const pagesCount = document.metadata.page_count ?? 1;
  const wordCount = document.metadata.word_count || 0;

  // Extract key headings or topics for a clean overview
  const headings = Array.from(
    new Set(
      document.elements
        ?.filter((el) => el.element_type === "heading" && el.content?.trim())
        .map((el) => el.content.trim()) || []
    )
  ).slice(0, 6);

  // Generate a clean brief / excerpt from the start of the document content
  const fullText = document.content || "";
  const briefText =
    fullText.length > 600
      ? fullText.slice(0, 600).trim() + "..."
      : fullText || "No preview content available.";

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col space-y-6">
      {/* Header: Title, Format Badge & Quick Action */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="px-2.5 py-0.5 text-xs font-bold rounded uppercase tracking-wide bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              {document.metadata.file_type}
            </span>
            <h2
              className="text-lg font-bold text-white truncate max-w-lg"
              title={document.metadata.source_name}
            >
              {document.metadata.source_name}
            </h2>
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs text-slate-400">
            <Calendar className="h-3.5 w-3.5 text-slate-500" />
            <span>Added {formatDate(document.metadata.created_at)}</span>
            <span>•</span>
            <span className="font-mono text-slate-500 text-[11px] truncate max-w-[180px]">
              ID: {document.id}
            </span>
          </div>
        </div>

        <button
          onClick={() => handleCopy(fullText)}
          className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white px-3.5 py-2 rounded-xl bg-slate-950 border border-slate-800 hover:bg-slate-800/80 transition-colors shrink-0"
        >
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 text-emerald-400" />
              <span className="text-emerald-400 font-medium">Copied Text</span>
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5" />
              <span>Copy Full Text</span>
            </>
          )}
        </button>
      </div>

      {/* Realistic, Simple Metadata Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Pages</span>
            <FileDigit className="h-3.5 w-3.5 text-blue-400" />
          </div>
          <div className="mt-1.5 text-xl font-bold text-white">
            {pagesCount} {pagesCount === 1 ? "page" : "pages"}
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Word Count</span>
            <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
          </div>
          <div className="mt-1.5 text-xl font-bold text-white">
            {wordCount.toLocaleString()}
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>File Size</span>
            <FileText className="h-3.5 w-3.5 text-purple-400" />
          </div>
          <div className="mt-1.5 text-xl font-bold text-white">
            {formatBytes(document.metadata.file_size_bytes)}
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Chunks Created</span>
            <Boxes className="h-3.5 w-3.5 text-indigo-400" />
          </div>
          <div className="mt-1.5 text-xl font-bold text-white">
            {chunksCount}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab("brief")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            activeTab === "brief"
              ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <FileText className="h-3.5 w-3.5" />
          Document Brief
        </button>

        <button
          onClick={() => setActiveTab("chunks")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            activeTab === "chunks"
              ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <Boxes className="h-3.5 w-3.5" />
          Chunks ({chunksCount})
        </button>

        <button
          onClick={() => setActiveTab("full_text")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            activeTab === "full_text"
              ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <BookOpen className="h-3.5 w-3.5" />
          Full Document Text
        </button>
      </div>

      {/* TAB 1: DOCUMENT BRIEF */}
      {activeTab === "brief" && (
        <div className="space-y-4">
          {/* Brief Card */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
                Brief & Summary
              </h4>
              <span className="text-[11px] text-slate-500">Preview</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-line">
              {briefText}
            </p>
          </div>

          {/* Key Topics / Sections Detected */}
          {headings.length > 0 && (
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                <Tag className="h-3.5 w-3.5 text-purple-400" />
                Key Sections & Topics
              </h4>
              <div className="flex flex-wrap gap-2">
                {headings.map((heading, idx) => (
                  <span
                    key={idx}
                    className="px-2.5 py-1 rounded-lg text-xs bg-slate-900 border border-slate-800 text-slate-300"
                  >
                    {heading}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: CHUNKS (CLEAN & PRACTICAL) */}
      {activeTab === "chunks" && (
        <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-slate-800">
          {chunks.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl">
              No chunks generated for this document.
            </div>
          ) : (
            chunks.map((chunk, idx) => {
              const chunkId = chunk.metadata?.chunk_id || chunk.id;
              const pageNum =
                chunk.metadata?.page ??
                (chunk.page_numbers && chunk.page_numbers.length > 0
                  ? chunk.page_numbers[0]
                  : null);
              const sectionName =
                chunk.metadata?.section ||
                (chunk.section_titles && chunk.section_titles.length > 0
                  ? chunk.section_titles[0]
                  : null);
              const tokenCount =
                chunk.token_count ||
                chunk.metadata?.token_count ||
                Math.round(chunk.word_count * 1.3);
              const isExpanded = expandedChunkId === chunkId;

              return (
                <div
                  key={chunkId || idx}
                  className="p-4 rounded-xl bg-slate-950 border border-slate-800 hover:border-slate-700/80 transition-all space-y-2.5"
                >
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 font-mono font-bold border border-indigo-500/20 text-xs">
                        Chunk #{chunk.chunk_index + 1}
                      </span>
                      {pageNum !== null && (
                        <span className="px-2 py-0.5 rounded text-xs bg-slate-900 text-slate-300 border border-slate-800">
                          Page {pageNum}
                        </span>
                      )}
                      {sectionName && (
                        <span className="text-xs text-slate-400 font-medium truncate max-w-xs">
                          • {sectionName}
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-slate-500">
                        {tokenCount} tokens
                      </span>
                      <button
                        onClick={() => handleCopyChunk(chunkId, chunk.content)}
                        className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition-colors"
                        title="Copy chunk text"
                      >
                        {copiedChunkId === chunkId ? (
                          <Check className="h-3.5 w-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="h-3.5 w-3.5" />
                        )}
                      </button>
                    </div>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed">
                    {isExpanded
                      ? chunk.content
                      : chunk.content.length > 220
                      ? chunk.content.slice(0, 220) + "..."
                      : chunk.content}
                  </p>

                  {chunk.content.length > 220 && (
                    <button
                      onClick={() =>
                        setExpandedChunkId(isExpanded ? null : chunkId)
                      }
                      className="text-xs text-indigo-400 hover:text-indigo-300 font-medium flex items-center gap-1 transition-colors"
                    >
                      {isExpanded ? (
                        <>
                          <span>Show Less</span>
                          <ChevronUp className="h-3 w-3" />
                        </>
                      ) : (
                        <>
                          <span>Read Full Chunk</span>
                          <ChevronDown className="h-3 w-3" />
                        </>
                      )}
                    </button>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}

      {/* TAB 3: FULL DOCUMENT TEXT (CLEAN READER) */}
      {activeTab === "full_text" && (
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 max-h-[500px] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-800">
          <div className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap font-sans">
            {fullText || "No text content available."}
          </div>
        </div>
      )}

      {/* Subtle, Collapsed Developer Details (Out of the way) */}
      <details className="pt-2 text-xs text-slate-500 group">
        <summary className="cursor-pointer hover:text-slate-400 flex items-center gap-1.5 select-none text-[11px]">
          <Code2 className="h-3 w-3" />
          <span>Technical details & Raw JSON</span>
        </summary>
        <div className="mt-2.5 p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-[11px] text-slate-400 overflow-x-auto max-h-60 scrollbar-thin scrollbar-thumb-slate-800">
          <pre>{JSON.stringify(document, null, 2)}</pre>
        </div>
      </details>
    </div>
  );
};

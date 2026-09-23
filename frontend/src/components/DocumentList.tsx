"use client";

import React, { useState } from "react";
import {
  Trash2,
  RefreshCw,
  Search,
  Layers,
} from "lucide-react";
import { DocumentMetadata, DocumentType } from "@/types/ingestion";

interface DocumentListProps {
  documents: DocumentMetadata[];
  selectedDocId: string | null;
  onSelectDocument: (docId: string) => void;
  onDeleteDocument: (docId: string) => void;
  onRefresh: () => void;
  isLoading: boolean;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  selectedDocId,
  onSelectDocument,
  onDeleteDocument,
  onRefresh,
  isLoading,
}) => {
  const [searchTerm, setSearchTerm] = useState("");

  const filteredDocs = documents.filter((doc) =>
    doc.source_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.file_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.document_id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getFormatBadge = (fileType: DocumentType) => {
    switch (fileType) {
      case "pdf":
        return <span className="px-2 py-0.5 text-xs font-bold rounded bg-red-500/20 text-red-400 border border-red-500/30">PDF</span>;
      case "docx":
        return <span className="px-2 py-0.5 text-xs font-bold rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">DOCX</span>;
      case "md":
        return <span className="px-2 py-0.5 text-xs font-bold rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">MD</span>;
      case "html":
        return <span className="px-2 py-0.5 text-xs font-bold rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">HTML</span>;
      case "txt":
      default:
        return <span className="px-2 py-0.5 text-xs font-bold rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">TXT</span>;
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const formatDate = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 pb-3.5 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-indigo-400" />
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">
            Ingested Documents ({documents.length})
          </h3>
        </div>
        <button
          onClick={onRefresh}
          disabled={isLoading}
          title="Refresh repository"
          className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin text-indigo-400" : ""}`} />
        </button>
      </div>

      {/* Search filter */}
      <div className="my-3 relative">
        <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-500" />
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Filter by name, type, or ID..."
          className="w-full pl-8 pr-3 py-1.5 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
        />
      </div>

      {/* Document List */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1 max-h-[520px] scrollbar-thin scrollbar-thumb-slate-800">
        {filteredDocs.length === 0 ? (
          <div className="p-6 text-center text-slate-500 text-xs border border-dashed border-slate-800 rounded-xl">
            {searchTerm ? "No documents match your filter." : "No documents ingested yet. Upload a file above."}
          </div>
        ) : (
          filteredDocs.map((doc) => {
            const isSelected = selectedDocId === doc.document_id;
            const chunkCount = doc.chunk_count ?? 0;
            const pageCount = doc.page_count ?? 1;

            return (
              <div
                key={doc.document_id}
                onClick={() => onSelectDocument(doc.document_id)}
                className={`group cursor-pointer p-3.5 rounded-xl border transition-all ${
                  isSelected
                    ? "bg-indigo-950/40 border-indigo-500/60 shadow-md shadow-indigo-950/50"
                    : "bg-slate-950/50 border-slate-800/80 hover:bg-slate-800/40 hover:border-slate-700"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    {getFormatBadge(doc.file_type)}
                    <span
                      title={doc.source_name}
                      className="text-xs font-semibold text-slate-200 truncate"
                    >
                      {doc.source_name}
                    </span>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm(`Delete '${doc.source_name}'?`)) {
                        onDeleteDocument(doc.document_id);
                      }
                    }}
                    title="Delete document"
                    className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded transition-all"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>

                {/* Metadata Row: Chunks, Words, Pages */}
                <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-indigo-400">{chunkCount} chunks</span>
                    <span>•</span>
                    <span>{pageCount} {pageCount === 1 ? "page" : "pages"}</span>
                    {doc.word_count > 0 && (
                      <>
                        <span>•</span>
                        <span>{doc.word_count.toLocaleString()} words</span>
                      </>
                    )}
                  </div>
                  <span className="text-slate-500">{formatBytes(doc.file_size_bytes)}</span>
                </div>

                {/* Hash / ID footer */}
                <div className="mt-1.5 flex items-center justify-between text-[10px] text-slate-500 font-mono">
                  <span className="truncate max-w-[140px]" title={doc.document_id}>
                    {doc.document_id}
                  </span>
                  <span>{formatDate(doc.created_at)}</span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

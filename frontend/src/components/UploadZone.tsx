"use client";

import React, { useRef, useState } from "react";
import {
  UploadCloud,
  FileText,
  AlertCircle,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import { ingestRawText, uploadDocumentFile } from "@/lib/api";
import { Document } from "@/types/ingestion";

interface UploadZoneProps {
  onIngestSuccess: (doc: Document) => void;
}

const SUPPORTED_FORMATS = [
  { ext: "PDF", label: "Multi-page documents", color: "bg-red-500/10 text-red-400 border-red-500/20" },
  { ext: "DOCX", label: "Word documents", color: "bg-blue-500/10 text-blue-400 border-blue-500/20" },
  { ext: "TXT", label: "Plain text with encoding auto-detect", color: "bg-amber-500/10 text-amber-400 border-amber-500/20" },
  { ext: "MD", label: "Markdown with structure", color: "bg-purple-500/10 text-purple-400 border-purple-500/20" },
  { ext: "HTML", label: "Webpages stripped of scripts", color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" },
];

export const UploadZone: React.FC<UploadZoneProps> = ({ onIngestSuccess }) => {
  const [activeTab, setActiveTab] = useState<"file" | "text">("file");
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Raw text form state
  const [rawTitle, setRawTitle] = useState("");
  const [rawContent, setRawContent] = useState("");
  const [rawFormat, setRawFormat] = useState<"txt" | "md" | "html">("txt");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const processFile = async (file: File) => {
    setIsProcessing(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      const response = await uploadDocumentFile(file);
      setSuccessMsg(response.message);
      onIngestSuccess(response.document);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to process document";
      setErrorMsg(message);
    } finally {
      setIsProcessing(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const handleRawSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!rawContent.trim()) {
      setErrorMsg("Content cannot be empty");
      return;
    }

    setIsProcessing(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      const title = rawTitle.trim() || `direct_input_${Date.now()}.${rawFormat}`;
      const response = await ingestRawText(title, rawContent, rawFormat);
      setSuccessMsg(response.message);
      onIngestSuccess(response.document);
      setRawContent("");
      setRawTitle("");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to ingest text";
      setErrorMsg(message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      {/* Tab Switcher */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-5">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <UploadCloud className="h-5 w-5 text-indigo-400" />
            Upload Document
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Add documents or paste text to extract metadata and generate retrieval chunks.
          </p>
        </div>

        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab("file")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === "file"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            File Upload
          </button>
          <button
            onClick={() => setActiveTab("text")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === "text"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Raw Text / Markdown
          </button>
        </div>
      </div>

      {/* Notifications */}
      {errorMsg && (
        <div className="mb-4 flex items-center gap-2 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm animate-fade-in">
          <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
          <span>{errorMsg}</span>
        </div>
      )}

      {successMsg && (
        <div className="mb-4 flex items-center gap-2 p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-sm animate-fade-in">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
          <span>{successMsg}</span>
        </div>
      )}

      {/* Tab: File Upload */}
      {activeTab === "file" && (
        <div>
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center transition-all ${
              isDragging
                ? "border-indigo-500 bg-indigo-500/10"
                : "border-slate-700/80 hover:border-slate-600 bg-slate-950/40 hover:bg-slate-950/60"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.doc,.txt,.md,.markdown,.html,.htm"
              className="hidden"
              onChange={handleFileSelect}
            />

            {isProcessing ? (
              <div className="flex flex-col items-center gap-3 py-4">
                <Loader2 className="h-10 w-10 text-indigo-400 animate-spin" />
                <p className="text-sm font-medium text-slate-200">
                  Processing document and preparing chunks...
                </p>
                <p className="text-xs text-slate-400">
                  Extracting text, computing metadata, and indexing for retrieval
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3 py-2">
                <div className="h-14 w-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shadow-inner">
                  <UploadCloud className="h-7 w-7" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-200">
                    Drag and drop your document here, or{" "}
                    <span className="text-indigo-400 underline underline-offset-4">browse</span>
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Supports PDF, DOCX, TXT, Markdown, and HTML
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Supported format badges */}
          <div className="mt-4 flex flex-wrap gap-2 items-center justify-center">
            {SUPPORTED_FORMATS.map((fmt) => (
              <span
                key={fmt.ext}
                className={`text-xs px-2.5 py-1 rounded-md border font-semibold ${fmt.color}`}
              >
                .{fmt.ext.toLowerCase()}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Tab: Raw Text Input */}
      {activeTab === "text" && (
        <form onSubmit={handleRawSubmit} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="sm:col-span-2">
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Document Title / Identifier
              </label>
              <input
                type="text"
                value={rawTitle}
                onChange={(e) => setRawTitle(e.target.value)}
                placeholder="e.g. contextual_notes.md"
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Content Syntax
              </label>
              <select
                value={rawFormat}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
                  setRawFormat(e.target.value as "txt" | "md" | "html")
                }
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="txt">Plain Text (.txt)</option>
                <option value="md">Markdown (.md)</option>
                <option value="html">HTML (.html)</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">
              Raw Text Content
            </label>
            <textarea
              rows={6}
              value={rawContent}
              onChange={(e) => setRawContent(e.target.value)}
              placeholder="Paste raw text, markdown headings, code blocks, or HTML tags..."
              className="w-full p-3 bg-slate-950 border border-slate-800 rounded-lg text-sm font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 resize-y"
            />
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={isProcessing || !rawContent.trim()}
              className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-semibold rounded-xl flex items-center gap-2 shadow-lg shadow-indigo-600/30 transition-all"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Embedding & Indexing...
                </>
              ) : (
                <>
                  <FileText className="h-4 w-4" />
                  Ingest & Normalize
                </>
              )}
            </button>
          </div>
        </form>
      )}
    </div>
  );
};

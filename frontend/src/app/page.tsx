"use client";

import React, { useCallback, useEffect, useState } from "react";
import { CrossSessionMemory } from "@/components/CrossSessionMemory";
import { GroundedQA } from "@/components/GroundedQA";
import { Navbar } from "@/components/Navbar";
import { UploadZone } from "@/components/UploadZone";
import { DocumentList } from "@/components/DocumentList";
import { DocumentInspector } from "@/components/DocumentInspector";
import { Toast, ToastMessage } from "@/components/Toast";
import {
  checkHealth,
  deleteDocumentById,
  fetchDocumentById,
  fetchDocuments,
} from "@/lib/api";
import { Document, DocumentMetadata, HealthStatus } from "@/types/ingestion";
import { Brain, FileText, MessageSquareQuote, Sparkles } from "lucide-react";

export default function IngestionPage() {
  const [activeTab, setActiveTab] = useState<"ingestion" | "qa" | "memory">("qa");
  const [backendHealth, setBackendHealth] = useState<HealthStatus | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [isLoadingSelected, setIsLoadingSelected] = useState(false);
  const [toast, setToast] = useState<ToastMessage | null>(null);

  // Poll / check backend health & initial docs
  const loadHealthAndDocs = useCallback(async () => {
    try {
      const health = await checkHealth();
      setBackendHealth(health);
      setHealthError(false);
    } catch {
      setHealthError(true);
      setBackendHealth(null);
    }

    try {
      setIsLoadingDocs(true);
      const docs = await fetchDocuments();
      setDocuments(docs);
      if (docs.length > 0 && !selectedDocId) {
        setSelectedDocId(docs[0].document_id);
      }
    } catch {
      // Backend may not be reachable initially
    } finally {
      setIsLoadingDocs(false);
    }
  }, [selectedDocId]);

  useEffect(() => {
    loadHealthAndDocs();
    const interval = setInterval(() => {
      checkHealth()
        .then((h) => {
          setBackendHealth(h);
          setHealthError(false);
        })
        .catch(() => {
          setBackendHealth(null);
          setHealthError(true);
        });
    }, 10000);
    return () => clearInterval(interval);
  }, [loadHealthAndDocs]);

  // Fetch full document when selectedDocId changes
  useEffect(() => {
    if (!selectedDocId) {
      setSelectedDoc(null);
      return;
    }

    let isMounted = true;
    setIsLoadingSelected(true);
    fetchDocumentById(selectedDocId)
      .then((doc) => {
        if (isMounted) setSelectedDoc(doc);
      })
      .catch((err) => {
        console.error(err);
      })
      .finally(() => {
        if (isMounted) setIsLoadingSelected(false);
      });

    return () => {
      isMounted = false;
    };
  }, [selectedDocId]);

  const handleIngestSuccess = (doc: Document) => {
    setDocuments((prev) => [doc.metadata, ...prev.filter((d) => d.document_id !== doc.id)]);
    setSelectedDocId(doc.id);
    setSelectedDoc(doc);
    setToast({
      id: String(Date.now()),
      title: "Document Ingested",
      description: `Document processed successfully with ${doc.chunks?.length || 0} chunks ready for retrieval.`,
      type: "success",
    });
  };

  const handleDeleteDocument = async (docId: string) => {
    try {
      await deleteDocumentById(docId);
      setDocuments((prev) => prev.filter((d) => d.document_id !== docId));
      if (selectedDocId === docId) {
        const remaining = documents.filter((d) => d.document_id !== docId);
        if (remaining.length > 0) {
          setSelectedDocId(remaining[0].document_id);
        } else {
          setSelectedDocId(null);
          setSelectedDoc(null);
        }
      }
      setToast({
        id: String(Date.now()),
        title: "Document Removed",
        description: `The document has been removed from the repository.`,
        type: "info",
      });
    } catch (err) {
      alert("Failed to delete document: " + err);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30">
      {/* Transient Toast Notification */}
      <Toast toast={toast} onDismiss={() => setToast(null)} />

      {/* Top Navbar */}
      <Navbar backendHealth={backendHealth} healthError={healthError} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8 space-y-8">
        {/* Page Header & Tabs */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-slate-900 pb-5">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
              {activeTab === "qa"
                ? "LLM Hallucination Mitigation"
                : activeTab === "ingestion"
                ? "Document Repository"
                : "Cross-Session Memory"}
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              {activeTab === "qa"
                ? "Side-by-side comparison of Naive LLM vs. Our Grounded Application."
                : activeTab === "ingestion"
                ? "Upload and manage source documents, chunks, and metadata."
                : "Inspect cross-session memory synchronization and summaries."}
            </p>
          </div>

          <div className="flex items-center gap-2 bg-slate-900/90 border border-slate-800 p-1 rounded-xl">
            <button
              onClick={() => setActiveTab("qa")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition ${
                activeTab === "qa"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <MessageSquareQuote className="w-4 h-4" />
              Comparison
            </button>
            <button
              onClick={() => setActiveTab("ingestion")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition ${
                activeTab === "ingestion"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <FileText className="w-4 h-4" />
              Documents
            </button>
            <button
              onClick={() => setActiveTab("memory")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition ${
                activeTab === "memory"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Brain className="w-4 h-4" />
              Memory
            </button>
          </div>
        </div>

        {activeTab === "qa" ? (
          <GroundedQA
            documents={documents}
            selectedDocId={selectedDocId}
            onSelectDocId={setSelectedDocId}
            onUploadSuccess={handleIngestSuccess}
          />
        ) : activeTab === "memory" ? (
          <CrossSessionMemory />
        ) : (
          <>
            {/* Upload Zone */}
            <UploadZone onIngestSuccess={handleIngestSuccess} />

            {/* Repository & Deep Inspector Split View */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Document List Sidebar (4 cols) */}
              <div className="lg:col-span-4">
                <DocumentList
                  documents={documents}
                  selectedDocId={selectedDocId}
                  onSelectDocument={setSelectedDocId}
                  onDeleteDocument={handleDeleteDocument}
                  onRefresh={loadHealthAndDocs}
                  isLoading={isLoadingDocs}
                />
              </div>

              {/* Deep Document Inspector (8 cols) */}
              <div className="lg:col-span-8">
                <DocumentInspector
                  document={selectedDoc}
                  isLoading={isLoadingSelected}
                />
              </div>
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/60 py-5 text-center text-xs text-slate-500">
        <p>Mitigating Hallucination and Context Degradation in LLMs • Major Project</p>
      </footer>
    </div>
  );
}

"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Activity, Brain, FileText, MessageSquare, Search } from "lucide-react";
import { CrossSessionMemory } from "@/components/CrossSessionMemory";
import { DocumentInspector } from "@/components/DocumentInspector";
import { DocumentList } from "@/components/DocumentList";
import { EvaluationWorkbench } from "@/components/EvaluationWorkbench";
import { GroundedQA } from "@/components/GroundedQA";
import { Navbar } from "@/components/Navbar";
import { RetrievedContext } from "@/components/RetrievedContext";
import { Toast, ToastMessage } from "@/components/Toast";
import {
  checkHealth,
  CompactedEvidence,
  deleteDocumentById,
  fetchDocumentById,
  fetchDocuments,
} from "@/lib/api";
import { Document, DocumentMetadata, HealthStatus } from "@/types/ingestion";

type MainTab = "ask" | "tools" | "context";
type ToolsTab = "documents" | "memory" | "evaluate";

export default function IngestionPage() {
  const [activeTab, setActiveTab] = useState<MainTab>("ask");
  const [toolsTab, setToolsTab] = useState<ToolsTab>("documents");
  const [backendHealth, setBackendHealth] = useState<HealthStatus | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [isLoadingSelected, setIsLoadingSelected] = useState(false);
  const [contextQuery, setContextQuery] = useState("");
  const [contextEvidence, setContextEvidence] = useState<CompactedEvidence[]>([]);
  const [toast, setToast] = useState<ToastMessage | null>(null);

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
      if (docs.length > 0 && !selectedDocId) setSelectedDocId(docs[0].document_id);
    } catch {
      // The health indicator communicates backend availability.
    } finally {
      setIsLoadingDocs(false);
    }
  }, [selectedDocId]);

  useEffect(() => {
    void loadHealthAndDocs();
    const interval = setInterval(() => {
      checkHealth()
        .then((health) => {
          setBackendHealth(health);
          setHealthError(false);
        })
        .catch(() => {
          setBackendHealth(null);
          setHealthError(true);
        });
    }, 10000);
    return () => clearInterval(interval);
  }, [loadHealthAndDocs]);

  useEffect(() => {
    if (!selectedDocId) {
      setSelectedDoc(null);
      return;
    }
    let isMounted = true;
    setIsLoadingSelected(true);
    fetchDocumentById(selectedDocId)
      .then((document) => {
        if (isMounted) setSelectedDoc(document);
      })
      .catch((error) => console.error(error))
      .finally(() => {
        if (isMounted) setIsLoadingSelected(false);
      });
    return () => {
      isMounted = false;
    };
  }, [selectedDocId]);

  const handleIngestSuccess = (document: Document) => {
    setDocuments((current) => [
      document.metadata,
      ...current.filter((item) => item.document_id !== document.id),
    ]);
    setSelectedDocId(document.id);
    setSelectedDoc(document);
    setToast({
      id: String(Date.now()),
      title: "Document uploaded",
      description: `${document.metadata.source_name} is ready to search.`,
      type: "success",
    });
  };

  const handleDeleteDocument = async (documentId: string) => {
    try {
      await deleteDocumentById(documentId);
      const remaining = documents.filter((item) => item.document_id !== documentId);
      setDocuments(remaining);
      if (selectedDocId === documentId) {
        setSelectedDocId(remaining[0]?.document_id ?? null);
      }
    } catch (error) {
      setToast({
        id: String(Date.now()),
        title: "Could not remove document",
        description: error instanceof Error ? error.message : "Please try again.",
        type: "error",
      });
    }
  };

  const showContext = (query: string, evidence: CompactedEvidence[]) => {
    setContextQuery(query);
    setContextEvidence(evidence);
  };

  const mainTabs: { id: MainTab; label: string; icon: React.ReactNode }[] = [
    { id: "ask", label: "Ask", icon: <MessageSquare className="h-4 w-4" /> },
    { id: "tools", label: "Documents & tools", icon: <FileText className="h-4 w-4" /> },
    { id: "context", label: "Retrieved chunks", icon: <Search className="h-4 w-4" /> },
  ];

  const toolTabs: { id: ToolsTab; label: string; icon: React.ReactNode }[] = [
    { id: "documents", label: "Documents", icon: <FileText className="h-4 w-4" /> },
    { id: "memory", label: "Memory", icon: <Brain className="h-4 w-4" /> },
    { id: "evaluate", label: "Evaluate", icon: <Activity className="h-4 w-4" /> },
  ];

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 font-sans text-slate-100">
      <Toast toast={toast} onDismiss={() => setToast(null)} />
      <Navbar backendHealth={backendHealth} healthError={healthError} />

      <main className="mx-auto w-full max-w-7xl flex-1 space-y-6 px-4 py-6 sm:px-6">
        <nav aria-label="Main navigation" className="flex gap-2 border-b border-slate-800">
          {mainTabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              aria-current={activeTab === tab.id ? "page" : undefined}
              className={`inline-flex items-center gap-2 border-b-2 px-4 py-3 text-sm transition ${
                activeTab === tab.id
                  ? "border-indigo-500 text-white"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </nav>

        {activeTab === "ask" && (
          <GroundedQA
            documents={documents}
            selectedDocId={selectedDocId}
            onSelectDocId={setSelectedDocId}
            onUploadSuccess={handleIngestSuccess}
            onContext={showContext}
          />
        )}

        {activeTab === "tools" && (
          <section className="space-y-5">
            <nav aria-label="Documents and tools" className="flex gap-2">
              {toolTabs.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setToolsTab(tab.id)}
                  aria-current={toolsTab === tab.id ? "page" : undefined}
                  className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
                    toolsTab === tab.id
                      ? "bg-slate-800 text-white"
                      : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </nav>

            {toolsTab === "documents" && (
              <div className="grid items-start gap-5 lg:grid-cols-12">
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
                <div className="lg:col-span-8">
                  <DocumentInspector document={selectedDoc} isLoading={isLoadingSelected} />
                </div>
              </div>
            )}
            {toolsTab === "memory" && <CrossSessionMemory />}
            {toolsTab === "evaluate" && (
              <EvaluationWorkbench document={selectedDoc} documentId={selectedDocId} />
            )}
          </section>
        )}

        {activeTab === "context" && (
          <RetrievedContext query={contextQuery} evidence={contextEvidence} />
        )}
      </main>
    </div>
  );
}

import { Document, DocumentMetadata, HealthStatus, IngestApiResponse } from "@/types/ingestion";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function checkHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE_URL}/api/health`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`API health check failed with status ${res.status}`);
  }
  return res.json();
}

export async function uploadDocumentFile(file: File): Promise<IngestApiResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/api/ingest/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Failed to upload and parse document");
  }

  return res.json();
}

export async function ingestRawText(
  title: string,
  content: string,
  format: "txt" | "md" | "html"
): Promise<IngestApiResponse> {
  const res = await fetch(`${API_BASE_URL}/api/ingest/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, content, format }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Failed to ingest text content");
  }

  return res.json();
}

export async function fetchDocuments(): Promise<DocumentMetadata[]> {
  const res = await fetch(`${API_BASE_URL}/api/ingest/documents`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error("Failed to fetch documents list");
  }
  return res.json();
}

export async function fetchDocumentById(docId: string): Promise<Document> {
  const res = await fetch(`${API_BASE_URL}/api/ingest/documents/${docId}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch document ${docId}`);
  }
  return res.json();
}

export async function deleteDocumentById(docId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/ingest/documents/${docId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`Failed to delete document ${docId}`);
  }
}

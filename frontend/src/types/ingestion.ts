export type DocumentType = "pdf" | "docx" | "txt" | "md" | "html" | "other";

export interface DocumentElement {
  id: string;
  element_type: string;
  content: string;
  page_number?: number | null;
  section_title?: string | null;
  char_count: number;
  word_count: number;
  metadata: Record<string, unknown>;
}

export interface ChunkMetadata {
  chunk_id: string;
  source: string;
  page?: number | null;
  section: string;
  document_type: string;
  created_at: string;
  token_count?: number;
  char_count?: number;
  word_count?: number;
  overlap_token_count?: number;
  has_table?: boolean;
  has_code?: boolean;
  prev_chunk_id?: string | null;
  next_chunk_id?: string | null;
  [key: string]: unknown;
}

export interface DocumentChunk {
  id: string;
  document_id: string;
  chunk_index: number;
  content: string;
  char_count: number;
  word_count: number;
  token_count?: number;
  page_numbers: number[];
  section_titles: string[];
  metadata: ChunkMetadata;
}

export interface DocumentMetadata {
  document_id: string;
  source_name: string;
  file_type: DocumentType;
  file_size_bytes: number;
  content_hash: string;
  char_count: number;
  word_count: number;
  element_count: number;
  chunk_count: number;
  page_count?: number | null;
  created_at: string;
  extra: Record<string, unknown>;
}

export interface Document {
  id: string;
  content: string;
  metadata: DocumentMetadata;
  elements: DocumentElement[];
  chunks: DocumentChunk[];
}

export interface IngestApiResponse {
  success: boolean;
  document: Document;
  message: string;
}

export interface HealthStatus {
  status: string;
  module: string;
  supported_extensions: string[];
  version: string;
}

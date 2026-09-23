"""FastAPI REST routes for document ingestion and management."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.indexing.manager import default_index_manager
from src.ingestion.models import Document, DocumentChunk, DocumentMetadata
from src.ingestion.registry import UnsupportedFormatError, default_registry
from src.ingestion.store import default_store

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])


class RawTextInput(BaseModel):
    """Schema for direct text/markdown ingestion."""
    title: str = Field(default="untitled.txt", description="Document title or pseudo-filename")
    content: str = Field(..., description="Raw text or markdown content")
    format: str = Field(default="txt", description="Format: 'txt', 'md', or 'html'")
    document_type: str | None = Field(
        default=None, description="Optional explicit document type category (e.g. HR_POLICY, TECHNICAL_SPEC)"
    )


class IngestResponse(BaseModel):
    """API response after successful ingestion."""
    success: bool = True
    document: Document
    message: str = "Document parsed and stored successfully"
    index_stats: dict | None = None


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="Document file to parse (PDF, DOCX, TXT, MD, HTML)"),
    document_type: str | None = Form(None, description="Optional explicit document type (e.g. HR_POLICY)"),
) -> IngestResponse:
    """Upload and parse any supported document into the standardized Document format."""
    filename = file.filename or "uploaded_document"

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        doc = default_registry.parse_bytes(
            content=content_bytes,
            filename=filename,
            mime_type=file.content_type,
            file_size_bytes=len(content_bytes),
            document_type=document_type,
        )
        default_store.add(doc)

        # Index chunks: Sentence-Transformers L2 embeddings -> FAISS + BM25 + Metadata Store
        idx_res = default_index_manager.index_document(doc)

        return IngestResponse(
            success=True,
            document=doc,
            message=f"Successfully parsed '{filename}' as {doc.metadata.file_type.value.upper()}",
            index_stats=idx_res,
        )
    except UnsupportedFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse document '{filename}': {exc!s}",
        )


@router.post("/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_text(payload: RawTextInput) -> IngestResponse:
    """Directly ingest raw text, Markdown, or HTML without a file upload."""
    if not payload.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content cannot be empty.",
        )

    ext = payload.format.lower().lstrip(".")
    filename = payload.title
    if not filename.endswith(f".{ext}"):
        filename = f"{filename}.{ext}"

    content_bytes = payload.content.encode("utf-8")

    try:
        doc = default_registry.parse_bytes(
            content=content_bytes,
            filename=filename,
            file_size_bytes=len(content_bytes),
            document_type=payload.document_type,
        )
        default_store.add(doc)

        # Index chunks: Sentence-Transformers L2 embeddings -> FAISS + BM25 + Metadata Store
        idx_res = default_index_manager.index_document(doc)

        return IngestResponse(
            success=True,
            document=doc,
            message=f"Successfully ingested raw text as {doc.metadata.file_type.value.upper()}",
            index_stats=idx_res,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest text: {exc!s}",
        )


@router.get("/documents", response_model=list[DocumentMetadata])
def list_documents() -> list[DocumentMetadata]:
    """Retrieve metadata for all ingested documents."""
    return default_store.list_all()


@router.get("/documents/{doc_id}", response_model=Document)
def get_document(doc_id: str) -> Document:
    """Retrieve a specific document by its unique ID."""
    doc = default_store.get(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return doc


@router.get("/documents/{doc_id}/chunks", response_model=list[DocumentChunk])
def get_document_chunks(doc_id: str) -> list[DocumentChunk]:
    """Retrieve all semantic chunks and tagged metadata for a specific document."""
    doc = default_store.get(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return doc.chunks


@router.get("/indexes/stats")
def get_index_stats() -> dict:
    """Retrieve current status and telemetry of FAISS, BM25, and Metadata indexes."""
    return default_index_manager.get_stats()


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    """Delete a document by its unique ID and remove from all indexes."""
    deleted = default_store.delete(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    # Remove from FAISS, BM25, and Metadata Store
    default_index_manager.remove_document(doc_id)
    return {"success": True, "message": f"Document '{doc_id}' deleted and unindexed successfully."}

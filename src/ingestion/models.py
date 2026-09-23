"""Data models for the Ingestion Module.

Defines the common Document object schema, metadata, and constituent elements.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    OTHER = "other"

    @classmethod
    def from_extension(cls, ext: str) -> DocumentType:
        """Map file extension to DocumentType."""
        clean_ext = ext.lower().lstrip(".")
        mapping = {
            "pdf": cls.PDF,
            "docx": cls.DOCX,
            "doc": cls.DOCX,
            "txt": cls.TXT,
            "text": cls.TXT,
            "md": cls.MD,
            "markdown": cls.MD,
            "html": cls.HTML,
            "htm": cls.HTML,
        }
        return mapping.get(clean_ext, cls.OTHER)


class DocumentElement(BaseModel):
    """A granular block or segment within a document (page, paragraph, heading, table)."""

    id: str = Field(..., description="Unique element identifier (e.g. doc_id_el_0)")
    element_type: str = Field(
        default="paragraph",
        description="Type of element: 'page', 'paragraph', 'heading', 'table', 'code', etc.",
    )
    content: str = Field(..., description="Normalized text content of this element")
    page_number: int | None = Field(
        default=None, description="1-indexed page number if applicable"
    )
    section_title: str | None = Field(
        default=None, description="Title of the enclosing section or heading"
    )
    char_count: int = Field(default=0, description="Character count of content")
    word_count: int = Field(default=0, description="Word count of content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Format-specific metadata for this element"
    )

    def model_post_init(self, __context: Any) -> None:
        """Automatically populate char_count and word_count if zero."""
        if not self.char_count and self.content:
            object.__setattr__(self, "char_count", len(self.content))
        if not self.word_count and self.content:
            object.__setattr__(self, "word_count", len(self.content.split()))


class DocumentChunk(BaseModel):
    """A retrievable text chunk derived from document elements or text segments."""

    id: str = Field(..., description="Unique chunk identifier (e.g. doc_id_chunk_0)")
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(default=0, description="0-indexed sequence of this chunk")
    content: str = Field(..., description="Text content of the chunk")
    char_count: int = Field(default=0, description="Character count of chunk")
    word_count: int = Field(default=0, description="Word count of chunk")
    token_count: int = Field(default=0, description="Token count of chunk")
    page_numbers: list[int] = Field(
        default_factory=list, description="Pages spanned by this chunk"
    )
    section_titles: list[str] = Field(
        default_factory=list, description="Sections/headings relevant to this chunk"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata and provenance for this chunk"
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count and self.content:
            object.__setattr__(self, "char_count", len(self.content))
        if not self.word_count and self.content:
            object.__setattr__(self, "word_count", len(self.content.split()))
        if not self.token_count and self.content:
            from src.chunking.tokenizer import count_tokens
            object.__setattr__(self, "token_count", count_tokens(self.content))


class DocumentMetadata(BaseModel):
    """Metadata describing a parsed document."""

    document_id: str = Field(..., description="Unique deterministic or generated identifier")
    source_name: str = Field(..., description="Original filename or document name")
    file_type: DocumentType = Field(..., description="Type of the original document")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    content_hash: str = Field(..., description="SHA-256 hash of extracted content")
    char_count: int = Field(default=0, description="Total characters in extracted content")
    word_count: int = Field(default=0, description="Total words in extracted content")
    element_count: int = Field(default=0, description="Total number of structured elements/blocks")
    chunk_count: int = Field(default=0, description="Total number of chunks produced")
    page_count: int | None = Field(
        default=None, description="Total page count (for paginated documents like PDF)"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 creation timestamp",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Extra metadata extracted from file header/tags"
    )


class Document(BaseModel):
    """The canonical Document object produced by all ingestion parsers.

    This serves as the single standard schema across downstream retrieval,
    reranking, and context degradation mitigation steps.
    """

    id: str = Field(..., description="Unique document ID (matching metadata.document_id)")
    content: str = Field(..., description="Full consolidated normalized text content")
    metadata: DocumentMetadata = Field(..., description="Document-level metadata")
    elements: list[DocumentElement] = Field(
        default_factory=list,
        description="List of structured elements (pages, headings, paragraphs)",
    )
    chunks: list[DocumentChunk] = Field(
        default_factory=list,
        description="List of retrievable text chunks generated from elements",
    )

    @classmethod
    def create(
        cls,
        source_name: str,
        file_type: DocumentType,
        elements: list[DocumentElement],
        file_size_bytes: int = 0,
        extra_metadata: dict[str, Any] | None = None,
        raw_content: str | None = None,
        page_count: int | None = None,
    ) -> Document:
        """Factory method to construct a canonical Document with cleaning and semantic chunking."""
        from src.chunking.semantic import default_semantic_chunker
        from src.cleaning.cleaner import default_cleaner

        # 1. Clean elements (removes duplicate headers, page numbers, broken line breaks, irrelevant symbols, repeated footers)
        cleaned_elements = default_cleaner.clean_elements(elements)

        # 2. Consolidate and clean text content
        if raw_content is not None:
            full_text = default_cleaner.clean_text(raw_content)
        else:
            full_text = "\n\n".join(
                el.content.strip() for el in cleaned_elements if el.content.strip()
            ).strip()

        # Compute deterministic content hash
        content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        doc_id = f"doc_{content_hash[:16]}"

        # Fix element IDs to reference doc_id if not already done
        updated_elements = []
        for idx, el in enumerate(cleaned_elements):
            el_id = el.id if el.id and not el.id.startswith("temp_") else f"{doc_id}_el_{idx}"
            updated_elements.append(
                DocumentElement(
                    id=el_id,
                    element_type=el.element_type,
                    content=el.content,
                    page_number=el.page_number,
                    section_title=el.section_title,
                    char_count=len(el.content),
                    word_count=len(el.content.split()),
                    metadata=el.metadata,
                )
            )

        # 3. Generate semantic chunks with 50-token overlap and rich metadata
        explicit_type = extra_metadata.get("document_type") if extra_metadata else None
        doc_chunks = default_semantic_chunker.chunk_elements(
            doc_id=doc_id,
            elements=updated_elements,
            source_name=source_name,
            document_type=explicit_type,
            raw_text=full_text,
            extra_metadata=extra_metadata,
        )

        char_count = len(full_text)
        word_count = len(full_text.split())
        metadata = DocumentMetadata(
            document_id=doc_id,
            source_name=source_name,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            content_hash=content_hash,
            char_count=char_count,
            word_count=word_count,
            element_count=len(updated_elements),
            chunk_count=len(doc_chunks),
            page_count=page_count,
            extra=extra_metadata or {},
        )

        return cls(
            id=doc_id,
            content=full_text,
            metadata=metadata,
            elements=updated_elements,
            chunks=doc_chunks,
        )

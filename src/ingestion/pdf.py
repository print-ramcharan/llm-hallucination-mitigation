"""PDF document parser using PyMuPDF (fitz)."""

from __future__ import annotations

from typing import Any

import fitz  # PyMuPDF

from src.ingestion.base import BaseParser
from src.ingestion.models import Document, DocumentElement, DocumentType


class PDFParser(BaseParser):
    """Extracts text, pages, and metadata from PDF files."""

    @property
    def supported_types(self) -> list[DocumentType]:
        return [DocumentType.PDF]

    def can_parse(self, extension: str, mime_type: str | None = None) -> bool:
        ext = extension.lower().lstrip(".")
        if ext == "pdf":
            return True
        if mime_type and "pdf" in mime_type.lower():
            return True
        return False

    def parse_bytes(
        self,
        content: bytes,
        filename: str = "document.pdf",
        file_size_bytes: int = 0,
        **kwargs,
    ) -> Document:
        """Parse PDF bytes into a standardized Document object."""
        if not content:
            raise ValueError("Cannot parse empty PDF byte stream")

        doc = fitz.open(stream=content, filetype="pdf")
        elements: list[DocumentElement] = []
        page_count = len(doc)
        full_text_parts: list[str] = []

        # Extract document-level metadata from PDF
        pdf_metadata: dict[str, Any] = {}
        try:
            raw_meta = doc.metadata or {}
            for k in ("title", "author", "subject", "keywords", "creator", "producer", "format"):
                if val := raw_meta.get(k):
                    pdf_metadata[k] = val
        except Exception:
            pass

        if kwargs.get("document_type"):
            pdf_metadata["document_type"] = kwargs["document_type"]
        if "extra_metadata" in kwargs and isinstance(kwargs["extra_metadata"], dict):
            pdf_metadata.update(kwargs["extra_metadata"])

        el_idx = 0
        for page_idx in range(page_count):
            page = doc[page_idx]
            page_num = page_idx + 1

            # Extract structured text blocks (preserves layout and paragraphs)
            blocks = page.get_text("blocks")
            page_text_parts = []

            for block in blocks:
                # fitz block format: (x0, y0, x1, y1, text, block_no, block_type)
                # block_type 0 is text
                if len(block) >= 5:
                    block_text = block[4].strip()
                    if block_text:
                        page_text_parts.append(block_text)
                        elements.append(
                            DocumentElement(
                                id=f"temp_el_{el_idx}",
                                element_type="paragraph",
                                content=block_text,
                                page_number=page_num,
                                metadata={
                                    "bbox": block[:4] if len(block) >= 4 else None,
                                    "block_number": block[5] if len(block) >= 6 else None,
                                },
                            )
                        )
                        el_idx += 1

            # If blocks yielded no text, fallback to page.get_text()
            if not page_text_parts:
                raw_page_text = page.get_text().strip()
                if raw_page_text:
                    page_text_parts.append(raw_page_text)
                    elements.append(
                        DocumentElement(
                            id=f"temp_el_{el_idx}",
                            element_type="page",
                            content=raw_page_text,
                            page_number=page_num,
                            metadata={"fallback": True},
                        )
                    )
                    el_idx += 1

            if page_text_parts:
                full_text_parts.append(f"--- Page {page_num} ---\n" + "\n\n".join(page_text_parts))

        doc.close()

        full_content = "\n\n".join(full_text_parts) if full_text_parts else ""

        return Document.create(
            source_name=filename,
            file_type=DocumentType.PDF,
            elements=elements,
            file_size_bytes=file_size_bytes or len(content),
            extra_metadata=pdf_metadata,
            raw_content=full_content,
            page_count=page_count,
        )

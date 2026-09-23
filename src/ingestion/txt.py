"""Plain text parser with multi-encoding fallback."""

from __future__ import annotations

from src.ingestion.base import BaseParser
from src.ingestion.models import Document, DocumentElement, DocumentType


class TxtParser(BaseParser):
    """Parses plain text (.txt) files with encoding auto-detection."""

    @property
    def supported_types(self) -> list[DocumentType]:
        return [DocumentType.TXT]

    def can_parse(self, extension: str, mime_type: str | None = None) -> bool:
        ext = extension.lower().lstrip(".")
        if ext in ("txt", "text", "log", "csv", "tsv"):
            return True
        if mime_type and "text/plain" in mime_type.lower():
            return True
        return False

    def parse_bytes(
        self,
        content: bytes,
        filename: str = "document.txt",
        file_size_bytes: int = 0,
        **kwargs,
    ) -> Document:
        """Parse raw text bytes, attempting common encodings."""
        if not content:
            raise ValueError("Cannot parse empty text byte stream")

        # Try common text encodings
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "utf-16"]
        decoded_text: str | None = None
        used_encoding: str = "unknown"

        for enc in encodings:
            try:
                decoded_text = content.decode(enc)
                used_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if decoded_text is None:
            # Fallback with replacement characters
            decoded_text = content.decode("utf-8", errors="replace")
            used_encoding = "utf-8-replace"

        # Normalize line endings
        normalized = decoded_text.replace("\r\n", "\n").replace("\r", "\n")

        # Split into paragraphs by blank lines
        raw_paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]

        elements: list[DocumentElement] = []
        for idx, para in enumerate(raw_paragraphs):
            elements.append(
                DocumentElement(
                    id=f"temp_el_{idx}",
                    element_type="paragraph",
                    content=para,
                    metadata={"paragraph_index": idx},
                )
            )

        extra = {"encoding": used_encoding}
        if kwargs.get("document_type"):
            extra["document_type"] = kwargs["document_type"]
        if "extra_metadata" in kwargs and isinstance(kwargs["extra_metadata"], dict):
            extra.update(kwargs["extra_metadata"])

        return Document.create(
            source_name=filename,
            file_type=DocumentType.TXT,
            elements=elements,
            file_size_bytes=file_size_bytes or len(content),
            extra_metadata=extra,
            raw_content=normalized.strip(),
        )

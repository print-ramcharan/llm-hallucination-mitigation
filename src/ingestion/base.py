"""Base parser interface for document ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.ingestion.models import Document, DocumentType


class BaseParser(ABC):
    """Abstract Base Class for all document parsers."""

    @property
    @abstractmethod
    def supported_types(self) -> list[DocumentType]:
        """List of DocumentType values supported by this parser."""
        ...

    @abstractmethod
    def can_parse(self, extension: str, mime_type: str | None = None) -> bool:
        """Check if this parser can handle the given file extension or MIME type."""
        ...

    @abstractmethod
    def parse_bytes(
        self,
        content: bytes,
        filename: str = "document",
        file_size_bytes: int = 0,
        **kwargs,
    ) -> Document:
        """Parse raw file bytes into a canonical Document object."""
        ...

    def parse_file(self, file_path: str | Path, **kwargs) -> Document:
        """Read and parse a file from disk into a canonical Document object."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        file_bytes = path.read_bytes()
        file_size = path.stat().st_size
        return self.parse_bytes(
            content=file_bytes,
            filename=path.name,
            file_size_bytes=file_size,
            **kwargs,
        )

"""Parser registry and factory for document ingestion."""

from __future__ import annotations

import os
from pathlib import Path

from src.ingestion.base import BaseParser
from src.ingestion.docx import DocxParser
from src.ingestion.html import HtmlParser
from src.ingestion.markdown import MarkdownParser
from src.ingestion.models import Document
from src.ingestion.pdf import PDFParser
from src.ingestion.txt import TxtParser


class UnsupportedFormatError(Exception):
    """Raised when an uploaded file cannot be parsed by any registered parser."""


class ParserRegistry:
    """Extensible registry and dispatcher for document parsers."""

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = []
        self._register_default_parsers()

    def _register_default_parsers(self) -> None:
        """Register built-in default parsers."""
        self.register(PDFParser())
        self.register(DocxParser())
        self.register(MarkdownParser())
        self.register(HtmlParser())
        self.register(TxtParser())  # TxtParser registered last as general text fallback

    def register(self, parser: BaseParser) -> None:
        """Register a new parser instance."""
        self._parsers.insert(0, parser)  # Prepend so newer/specialized parsers take precedence

    @property
    def registered_parsers(self) -> list[BaseParser]:
        return list(self._parsers)

    def get_supported_extensions(self) -> list[str]:
        """Return list of standard supported extensions."""
        return [".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".html", ".htm"]

    def find_parser(
        self, filename: str, mime_type: str | None = None
    ) -> BaseParser | None:
        """Find the matching parser for a given filename and/or MIME type."""
        _, ext = os.path.splitext(filename)
        for parser in self._parsers:
            if parser.can_parse(extension=ext, mime_type=mime_type):
                return parser
        return None

    def parse_bytes(
        self,
        content: bytes,
        filename: str,
        mime_type: str | None = None,
        file_size_bytes: int = 0,
        **kwargs,
    ) -> Document:
        """Dispatch parsing to the appropriate parser for raw bytes."""
        parser = self.find_parser(filename=filename, mime_type=mime_type)
        if not parser:
            raise UnsupportedFormatError(
                f"No parser available for filename '{filename}' (MIME: {mime_type}). "
                f"Supported formats: {', '.join(self.get_supported_extensions())}"
            )
        return parser.parse_bytes(
            content=content,
            filename=filename,
            file_size_bytes=file_size_bytes or len(content),
            **kwargs,
        )

    def parse_file(
        self,
        file_path: str | Path,
        mime_type: str | None = None,
        **kwargs,
    ) -> Document:
        """Dispatch parsing to the appropriate parser for a file on disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        parser = self.find_parser(filename=path.name, mime_type=mime_type)
        if not parser:
            raise UnsupportedFormatError(
                f"No parser available for file '{path.name}'. "
                f"Supported formats: {', '.join(self.get_supported_extensions())}"
            )
        return parser.parse_file(path, **kwargs)


# Global default registry instance
default_registry = ParserRegistry()

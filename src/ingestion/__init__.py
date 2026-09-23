"""Ingestion module for heterogeneous document processing and unified representation."""

from src.ingestion.base import BaseParser
from src.ingestion.chunker import ElementChunker, default_chunker
from src.ingestion.docx import DocxParser
from src.ingestion.html import HtmlParser
from src.ingestion.markdown import MarkdownParser
from src.ingestion.models import (
    Document,
    DocumentChunk,
    DocumentElement,
    DocumentMetadata,
    DocumentType,
)
from src.ingestion.pdf import PDFParser
from src.ingestion.registry import (
    ParserRegistry,
    UnsupportedFormatError,
    default_registry,
)
from src.ingestion.store import DocumentStore, default_store
from src.ingestion.txt import TxtParser

__all__ = [
    "BaseParser",
    "Document",
    "DocumentChunk",
    "DocumentElement",
    "DocumentMetadata",
    "DocumentStore",
    "DocumentType",
    "DocxParser",
    "ElementChunker",
    "HtmlParser",
    "MarkdownParser",
    "PDFParser",
    "ParserRegistry",
    "TxtParser",
    "UnsupportedFormatError",
    "default_chunker",
    "default_registry",
    "default_store",
]

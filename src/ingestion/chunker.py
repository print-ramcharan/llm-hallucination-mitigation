"""Chunking engine for breaking down document elements into standardized chunks using Semantic Chunking."""

from __future__ import annotations

from typing import Any

from src.ingestion.models import DocumentChunk, DocumentElement


class ElementChunker:
    """Semantic chunking engine that groups and segments document elements into cohesive retrieval chunks.

    Chunks documents along semantic boundaries (sections, headings, tables, paragraphs)
    and maintains token overlap (default: 50 tokens) between consecutive chunks to
    mitigate context degradation at chunk seams.
    """

    def __init__(
        self,
        target_tokens: int = 350,
        overlap_tokens: int = 50,
        target_words: int | None = None,
        overlap_words: int | None = None,
        target_chars: int | None = None,
        overlap_chars: int | None = None,
    ) -> None:
        if target_words is not None:
            # Approximate 1.3 tokens per word
            target_tokens = max(1, int(target_words * 1.3))
        elif target_chars is not None:
            # Approximate 4 characters per token
            target_tokens = max(1, int(target_chars / 4))

        if overlap_words is not None:
            overlap_tokens = max(0, int(overlap_words * 1.3))
        elif overlap_chars is not None:
            overlap_tokens = max(0, int(overlap_chars / 4))

        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self._chunker: Any | None = None

    def _get_chunker(self) -> Any:
        if self._chunker is None:
            from src.chunking.semantic import SemanticChunker

            self._chunker = SemanticChunker(
                target_tokens=self.target_tokens, overlap_tokens=self.overlap_tokens
            )
        return self._chunker

    def chunk_elements(
        self,
        doc_id: str,
        elements: list[DocumentElement],
        source_name: str = "document.txt",
        document_type: str | None = None,
        raw_text: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[DocumentChunk]:
        """Convert document elements into standardized DocumentChunk objects using semantic chunking."""
        return self._get_chunker().chunk_elements(
            doc_id=doc_id,
            elements=elements,
            source_name=source_name,
            document_type=document_type,
            raw_text=raw_text,
            extra_metadata=extra_metadata,
        )


class _DefaultChunkerProxy:
    """Proxy object delegating to default_semantic_chunker to avoid circular import."""

    def __getattr__(self, name: str) -> Any:
        from src.chunking.semantic import default_semantic_chunker

        return getattr(default_semantic_chunker, name)

    def chunk_elements(self, *args: Any, **kwargs: Any) -> list[DocumentChunk]:
        from src.chunking.semantic import default_semantic_chunker

        return default_semantic_chunker.chunk_elements(*args, **kwargs)


default_chunker: Any = _DefaultChunkerProxy()

"""Chunking engine for breaking down document elements into standardized chunks."""

from __future__ import annotations

from src.ingestion.models import DocumentChunk, DocumentElement


class ElementChunker:
    """Chunker that groups and segments document elements into cohesive retrieval chunks.

    Defaults to word-based chunking with 220 words target size and 30 words overlap,
    matching the system architecture specification: Word based chunking (220 size + 30 overlap).
    """

    def __init__(
        self,
        target_words: int = 220,
        overlap_words: int = 30,
        target_chars: int | None = None,
        overlap_chars: int | None = None,
    ) -> None:
        if target_words < 1:
            raise ValueError("target_words must be at least 1")
        if overlap_words < 0:
            raise ValueError("overlap_words cannot be negative")
        self.target_words = target_words
        self.overlap_words = min(overlap_words, target_words - 1)
        # Approximate 6 characters per word as fallback
        self.target_chars = target_chars or (target_words * 6)
        self.overlap_chars = overlap_chars or (self.overlap_words * 6)

    def chunk_elements(
        self,
        doc_id: str,
        elements: list[DocumentElement],
        raw_text: str | None = None,
    ) -> list[DocumentChunk]:
        """Convert document elements into standardized DocumentChunk objects."""
        if not elements:
            if not raw_text or not raw_text.strip():
                return []
            # Fallback for plain raw text without elements
            return self._chunk_plain_text(doc_id, raw_text)

        chunks: list[DocumentChunk] = []
        current_words: list[str] = []
        current_pages: set[int] = set()
        current_sections: set[str] = set()
        chunk_idx = 0

        def flush() -> None:
            nonlocal chunk_idx, current_words, current_pages, current_sections
            if not current_words:
                return
            chunk_text = " ".join(current_words).strip()
            chunks.append(
                DocumentChunk(
                    id=f"{doc_id}_chunk_{chunk_idx}",
                    document_id=doc_id,
                    chunk_index=chunk_idx,
                    content=chunk_text,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    page_numbers=sorted(current_pages),
                    section_titles=sorted(current_sections),
                )
            )
            chunk_idx += 1
            overlap = current_words[-self.overlap_words :] if self.overlap_words else []
            current_words = overlap
            current_pages = set()
            current_sections = set()

        for el in elements:
            text = el.content.strip()
            if not text:
                continue
            words = text.split()
            offset = 0
            while offset < len(words):
                available = self.target_words - len(current_words)
                if available <= 0:
                    flush()
                    available = self.target_words - len(current_words)
                take = min(available, len(words) - offset)
                current_words.extend(words[offset : offset + take])
                offset += take
                if el.page_number is not None:
                    current_pages.add(el.page_number)
                if el.section_title:
                    current_sections.add(el.section_title)
                if len(current_words) >= self.target_words and offset < len(words):
                    flush()
            # Preserve an element boundary when the next element would overflow.
            if len(current_words) >= self.target_words:
                flush()

        # Flush final chunk
        if current_words:
            # Do not emit a trailing chunk consisting only of the prior overlap.
            if not chunks or len(current_words) > self.overlap_words:
                flush()

        return chunks

    def _chunk_plain_text(self, doc_id: str, text: str) -> list[DocumentChunk]:
        """Word-based sliding window chunker with 220 words target size and 30 words overlap."""
        words = text.split()
        if not words:
            return []

        chunks: list[DocumentChunk] = []
        chunk_idx = 0
        start = 0
        step = max(1, self.target_words - self.overlap_words)

        while start < len(words):
            chunk_words = words[start : start + self.target_words]
            chunk_str = " ".join(chunk_words).strip()
            if chunk_str:
                chunks.append(
                    DocumentChunk(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        content=chunk_str,
                        char_count=len(chunk_str),
                        word_count=len(chunk_words),
                    )
                )
                chunk_idx += 1
            start += step

        return chunks


default_chunker = ElementChunker()

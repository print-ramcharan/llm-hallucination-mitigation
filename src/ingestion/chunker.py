"""Chunking engine for breaking down document elements into standardized chunks."""

from __future__ import annotations

from src.ingestion.models import DocumentChunk, DocumentElement


class ElementChunker:
    """Chunker that groups and segments document elements into cohesive retrieval chunks."""

    def __init__(self, target_chars: int = 800, overlap_chars: int = 100) -> None:
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

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
        current_texts: list[str] = []
        current_pages: set[int] = set()
        current_sections: set[str] = set()
        current_len = 0
        chunk_idx = 0

        for el in elements:
            text = el.content.strip()
            if not text:
                continue

            if el.page_number is not None:
                current_pages.add(el.page_number)
            if el.section_title:
                current_sections.add(el.section_title)

            # If adding this element exceeds target and we already have content, flush current
            if current_len > 0 and (current_len + len(text) > self.target_chars):
                chunk_text = "\n\n".join(current_texts).strip()
                chunks.append(
                    DocumentChunk(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        char_count=len(chunk_text),
                        word_count=len(chunk_text.split()),
                        page_numbers=sorted(list(current_pages)),
                        section_titles=sorted(list(current_sections)),
                    )
                )
                chunk_idx += 1

                # Reset accumulator
                current_texts = [text]
                current_len = len(text)
                current_pages = {el.page_number} if el.page_number is not None else set()
                current_sections = {el.section_title} if el.section_title else set()
            else:
                current_texts.append(text)
                current_len += len(text)

        # Flush final chunk
        if current_texts:
            chunk_text = "\n\n".join(current_texts).strip()
            chunks.append(
                DocumentChunk(
                    id=f"{doc_id}_chunk_{chunk_idx}",
                    document_id=doc_id,
                    chunk_index=chunk_idx,
                    content=chunk_text,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    page_numbers=sorted(list(current_pages)),
                    section_titles=sorted(list(current_sections)),
                )
            )

        return chunks

    def _chunk_plain_text(self, doc_id: str, text: str) -> list[DocumentChunk]:
        """Simple sliding window chunker for raw unformatted text."""
        chunks: list[DocumentChunk] = []
        start = 0
        text_len = len(text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.target_chars, text_len)
            chunk_str = text[start:end].strip()
            if chunk_str:
                chunks.append(
                    DocumentChunk(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        content=chunk_str,
                        char_count=len(chunk_str),
                        word_count=len(chunk_str.split()),
                    )
                )
                chunk_idx += 1
            start += max(1, self.target_chars - self.overlap_chars)

        return chunks

default_chunker = ElementChunker()

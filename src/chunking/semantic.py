"""Semantic chunking engine with 50 tokens overlap and boundary awareness."""

from __future__ import annotations

import re
from typing import Any

from src.chunking.metadata import MetadataTagger
from src.chunking.tokenizer import count_tokens, get_trailing_token_overlap
from src.ingestion.models import DocumentChunk, DocumentElement


class SemanticChunker:
    """Chunks documents along semantic boundaries (sections, headings, tables, paragraphs)

    maintains an exact or tailored token overlap (default: 50 tokens) between
    consecutive chunks to mitigate context degradation at chunk seams.
    """

    def __init__(self, target_tokens: int = 350, overlap_tokens: int = 50) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_elements(
        self,
        doc_id: str,
        elements: list[DocumentElement],
        source_name: str = "document.txt",
        document_type: str | None = None,
        raw_text: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Convert a list of cleaned DocumentElements into semantic DocumentChunks."""
        if not elements:
            if not raw_text or not raw_text.strip():
                return []
            return self._chunk_plain_text(
                doc_id=doc_id,
                text=raw_text,
                source_name=source_name,
                document_type=document_type,
                extra_metadata=extra_metadata,
            )

        # Detect or infer document_type if not provided
        sample_text = " ".join(el.content[:200] for el in elements[:5])
        inferred_type = MetadataTagger.infer_document_type(
            source_name=source_name,
            text_sample=sample_text,
            explicit_type=document_type,
        )

        chunks: list[DocumentChunk] = []
        current_texts: list[str] = []
        current_pages: set[int] = set()
        current_sections: set[str] = set()
        current_token_count = 0
        seq_num = 1
        previous_overlap_text = ""
        current_section_title = "General"

        for el in elements:
            text = el.content.strip()
            if not text:
                continue

            # Track page numbers and section titles
            if el.page_number is not None:
                current_pages.add(el.page_number)
            if el.section_title:
                current_sections.add(el.section_title)
                current_section_title = el.section_title
            elif el.element_type == "heading":
                current_section_title = text.lstrip("#").strip()
                current_sections.add(current_section_title)

            el_tokens = count_tokens(text)

            # If element is huge (e.g. giant paragraph > target_tokens), split it by sentences
            if el_tokens > self.target_tokens and el.element_type not in ("table", "code"):
                sub_sentences = self._split_into_sentences(text)
                for sentence in sub_sentences:
                    s_tokens = count_tokens(sentence)
                    if current_token_count > 0 and (current_token_count + s_tokens > self.target_tokens):
                        chunk = self._finalize_chunk(
                            doc_id=doc_id,
                            source_name=source_name,
                            inferred_type=inferred_type,
                            texts=current_texts,
                            pages=current_pages,
                            sections=current_sections,
                            section_title=current_section_title,
                            seq_num=seq_num,
                            overlap_tokens_count=count_tokens(previous_overlap_text),
                            extra_metadata=extra_metadata,
                        )
                        chunks.append(chunk)
                        seq_num += 1

                        # Generate 50-token overlap for the next chunk
                        previous_overlap_text = get_trailing_token_overlap(
                            chunk.content, self.overlap_tokens
                        )
                        current_texts = [previous_overlap_text, sentence] if previous_overlap_text else [sentence]
                        current_token_count = count_tokens("\n\n".join(current_texts))
                        current_pages = {el.page_number} if el.page_number is not None else set()
                        current_sections = {current_section_title}
                    else:
                        current_texts.append(sentence)
                        current_token_count += s_tokens
                continue

            # Check if adding this element breaches target size
            if current_token_count > 0 and (current_token_count + el_tokens > self.target_tokens):
                chunk = self._finalize_chunk(
                    doc_id=doc_id,
                    source_name=source_name,
                    inferred_type=inferred_type,
                    texts=current_texts,
                    pages=current_pages,
                    sections=current_sections,
                    section_title=current_section_title,
                    seq_num=seq_num,
                    overlap_tokens_count=count_tokens(previous_overlap_text),
                    extra_metadata=extra_metadata,
                )
                chunks.append(chunk)
                seq_num += 1

                # Generate 50-token overlap from the finalized chunk
                previous_overlap_text = get_trailing_token_overlap(
                    chunk.content, self.overlap_tokens
                )
                current_texts = [previous_overlap_text, text] if previous_overlap_text else [text]
                current_token_count = count_tokens("\n\n".join(current_texts))
                current_pages = {el.page_number} if el.page_number is not None else set()
                current_sections = {current_section_title}
            else:
                current_texts.append(text)
                current_token_count += el_tokens

        # Flush final remaining accumulator
        if current_texts:
            chunk = self._finalize_chunk(
                doc_id=doc_id,
                source_name=source_name,
                inferred_type=inferred_type,
                texts=current_texts,
                pages=current_pages,
                sections=current_sections,
                section_title=current_section_title,
                seq_num=seq_num,
                overlap_tokens_count=count_tokens(previous_overlap_text),
                extra_metadata=extra_metadata,
            )
            chunks.append(chunk)

        # Wire sequential pointers between chunks
        self._wire_chunk_pointers(chunks)

        return chunks

    def _finalize_chunk(
        self,
        doc_id: str,
        source_name: str,
        inferred_type: str,
        texts: list[str],
        pages: set[int],
        sections: set[str],
        section_title: str,
        seq_num: int,
        overlap_tokens_count: int = 0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> DocumentChunk:
        """Assemble a single DocumentChunk with complete metadata."""
        content = "\n\n".join(t for t in texts if t.strip()).strip()
        tokens = count_tokens(content)
        primary_page = sorted(list(pages))[0] if pages else None
        primary_section = sorted(list(sections))[0] if sections else section_title

        meta = MetadataTagger.build_chunk_metadata(
            source_name=source_name,
            page=primary_page,
            section=primary_section,
            document_type=inferred_type,
            sequence_num=seq_num,
            content=content,
            token_count=tokens,
            overlap_token_count=overlap_tokens_count,
            extra=extra_metadata,
        )

        return DocumentChunk(
            id=meta["chunk_id"],
            document_id=doc_id,
            chunk_index=seq_num - 1,
            content=content,
            char_count=len(content),
            word_count=len(content.split()),
            page_numbers=sorted(list(pages)),
            section_titles=sorted(list(sections)),
            metadata=meta,
        )

    def _chunk_plain_text(
        self,
        doc_id: str,
        text: str,
        source_name: str,
        document_type: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Fallback semantic chunker for raw plain text documents."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        inferred_type = MetadataTagger.infer_document_type(
            source_name=source_name,
            text_sample=text[:500],
            explicit_type=document_type,
        )

        chunks: list[DocumentChunk] = []
        current_texts: list[str] = []
        current_tokens = 0
        seq_num = 1
        previous_overlap = ""

        for p in paragraphs:
            p_tokens = count_tokens(p)
            if current_tokens > 0 and (current_tokens + p_tokens > self.target_tokens):
                content = "\n\n".join(current_texts).strip()
                tokens = count_tokens(content)
                meta = MetadataTagger.build_chunk_metadata(
                    source_name=source_name,
                    page=None,
                    section="General",
                    document_type=inferred_type,
                    sequence_num=seq_num,
                    content=content,
                    token_count=tokens,
                    overlap_token_count=count_tokens(previous_overlap),
                    extra=extra_metadata,
                )
                chunk = DocumentChunk(
                    id=meta["chunk_id"],
                    document_id=doc_id,
                    chunk_index=seq_num - 1,
                    content=content,
                    char_count=len(content),
                    word_count=len(content.split()),
                    metadata=meta,
                )
                chunks.append(chunk)
                seq_num += 1

                previous_overlap = get_trailing_token_overlap(content, self.overlap_tokens)
                current_texts = [previous_overlap, p] if previous_overlap else [p]
                current_tokens = count_tokens("\n\n".join(current_texts))
            else:
                current_texts.append(p)
                current_tokens += p_tokens

        if current_texts:
            content = "\n\n".join(current_texts).strip()
            tokens = count_tokens(content)
            meta = MetadataTagger.build_chunk_metadata(
                source_name=source_name,
                page=None,
                section="General",
                document_type=inferred_type,
                sequence_num=seq_num,
                content=content,
                token_count=tokens,
                overlap_token_count=count_tokens(previous_overlap),
                extra=extra_metadata,
            )
            chunk = DocumentChunk(
                id=meta["chunk_id"],
                document_id=doc_id,
                chunk_index=seq_num - 1,
                content=content,
                char_count=len(content),
                word_count=len(content.split()),
                metadata=meta,
            )
            chunks.append(chunk)

        self._wire_chunk_pointers(chunks)
        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split a long block of text into sentences."""
        # Split on period, question mark, or exclamation mark followed by whitespace
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _wire_chunk_pointers(self, chunks: list[DocumentChunk]) -> None:
        """Inject previous_chunk_id and next_chunk_id for traversal."""
        for i, chunk in enumerate(chunks):
            prev_id = chunks[i - 1].id if i > 0 else None
            next_id = chunks[i + 1].id if i < len(chunks) - 1 else None
            chunk.metadata["prev_chunk_id"] = prev_id
            chunk.metadata["next_chunk_id"] = next_id


default_semantic_chunker = SemanticChunker(target_tokens=350, overlap_tokens=50)

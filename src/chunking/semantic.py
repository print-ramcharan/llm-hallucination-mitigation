"""Semantic chunking engine with boundary awareness, embedding-based semantic breakpoints, and 50 tokens overlap."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from src.chunking.metadata import MetadataTagger
from src.chunking.tokenizer import count_tokens, get_trailing_token_overlap
from src.ingestion.models import DocumentChunk, DocumentElement

_PROTECTED_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Inc.",
    "Ltd.",
    "vs.",
    "Fig.",
    "etc.",
)

_COMMON_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "about",
    "by",
    "of",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "we",
    "our",
}

_ALPHANUM_REGEX = re.compile(r"[a-zA-Z0-9]+")
_DECIMAL_REGEX = re.compile(r"(\d+)\.(\d+)")
_LIST_DOT_REGEX = re.compile(r"(?<=\b\d)\.(?=\s+)")
_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
_STRUCTURE_REGEX = re.compile(
    r"^(?:Module\s+Schedule|Schedule|Modules|Topics|Overview|Agenda|•|-|\*|\d+[\.\)])",
    re.IGNORECASE,
)
_SUFFIXES = ("ing", "tion", "ment", "able", "ness", "ity", "ed", "ly", "es", "s")


def _stem_word(w: str) -> str:
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) > len(suf) + 3:
            return w[: -len(suf)]
    return w


class SemanticChunker:
    """Chunks documents along semantic boundaries using semantic similarity and breakpoint detection,

    while maintaining exact or tailored token overlap (default: 50 tokens) between consecutive chunks
    to mitigate context degradation at chunk seams.
    """

    def __init__(
        self,
        target_tokens: int = 350,
        overlap_tokens: int = 50,
        min_tokens: int | None = None,
        similarity_threshold: float = 0.05,
        embedding_engine: Any | None = None,
        use_embeddings: bool = False,
    ) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = (
            min_tokens
            if min_tokens is not None
            else min(40, max(10, target_tokens // 2))
        )
        self.similarity_threshold = similarity_threshold
        self._embedding_engine = embedding_engine
        self.use_embeddings = use_embeddings
        self._sentence_embedding_cache: dict[str, Any] = {}
        self._term_vector_cache: dict[str, tuple[Counter[str], float]] = {}

    def _get_embedding_engine(self) -> Any | None:
        if self._embedding_engine is not None:
            return self._embedding_engine
        if self.use_embeddings:
            try:
                from src.indexing.embeddings import default_embedding_engine

                self._embedding_engine = default_embedding_engine
            except Exception:
                self._embedding_engine = None
        return self._embedding_engine

    def compute_sentence_similarity(self, s1: str, s2: str) -> float:
        """Compute semantic similarity between two sentences in [0.0, 1.0].

        Uses neural sentence embeddings with caching when use_embeddings=True,
        or high-performance lexical-semantic term vector cosine similarity as default.
        """
        s1_strip = s1.strip()
        s2_strip = s2.strip()
        if not s1_strip or not s2_strip:
            return 0.0

        if self.use_embeddings:
            engine = self._get_embedding_engine()
            if engine is not None:
                try:
                    import numpy as np

                    v1 = self._sentence_embedding_cache.get(s1_strip)
                    v2 = self._sentence_embedding_cache.get(s2_strip)
                    missing: list[str] = []
                    if v1 is None:
                        missing.append(s1_strip)
                    if v2 is None and s2_strip != s1_strip:
                        missing.append(s2_strip)
                    if missing:
                        vecs = engine.embed_texts(missing)
                        if len(self._sentence_embedding_cache) >= 10000:
                            self._sentence_embedding_cache.clear()
                        for m_text, m_vec in zip(missing, vecs):
                            self._sentence_embedding_cache[m_text] = m_vec
                        if v1 is None:
                            v1 = self._sentence_embedding_cache[s1_strip]
                        if v2 is None:
                            v2 = self._sentence_embedding_cache[s2_strip]
                    sim = float(np.dot(v1, v2))
                    return max(0.0, min(1.0, sim))
                except Exception:
                    pass

        # Fast lexical-semantic cosine similarity with stop-word filtering & stemming
        v1, norm1 = self._get_term_vector_and_norm(s1_strip)
        v2, norm2 = self._get_term_vector_and_norm(s2_strip)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        if len(v1) < len(v2):
            dot_product = sum(cnt * v2.get(k, 0) for k, cnt in v1.items())
        else:
            dot_product = sum(cnt * v1.get(k, 0) for k, cnt in v2.items())

        return max(0.0, min(1.0, dot_product / (norm1 * norm2)))

    def _get_term_vector_and_norm(self, text: str) -> tuple[Counter[str], float]:
        cached = self._term_vector_cache.get(text)
        if cached is not None:
            return cached

        raw_tokens = _ALPHANUM_REGEX.findall(text.lower())
        meaningful = [
            _stem_word(tok)
            for tok in raw_tokens
            if len(tok) > 1 and tok not in _COMMON_STOP_WORDS
        ]
        v = Counter(meaningful)
        norm = math.sqrt(sum(c * c for c in v.values())) if v else 0.0
        if len(self._term_vector_cache) >= 15000:
            self._term_vector_cache.clear()
        self._term_vector_cache[text] = (v, norm)
        return v, norm

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
        last_sentence: str | None = None

        for el in elements:
            text = el.content.strip()
            if not text:
                continue

            el_page = el.page_number
            el_section = el.section_title
            is_heading = el.element_type == "heading" or text.startswith("#")
            if not el_section and is_heading:
                el_section = text.lstrip("#").strip()

            # Hard structural semantic boundary on new heading
            if is_heading:
                heading_title = text.lstrip("#").strip()
                if current_token_count >= self.min_tokens:
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

                    previous_overlap_text = get_trailing_token_overlap(
                        chunk.content, self.overlap_tokens
                    )
                    current_texts = (
                        [previous_overlap_text] if previous_overlap_text else []
                    )
                    current_token_count = count_tokens("\n\n".join(current_texts))
                    current_pages = set()
                    current_sections = {heading_title}

                current_section_title = heading_title
                current_sections.add(heading_title)
                heading_line = (
                    f"## {heading_title}" if not text.startswith("#") else text
                )
                current_texts.append(heading_line)
                current_token_count += count_tokens(heading_line)
                if el_page is not None:
                    current_pages.add(el_page)
                last_sentence = heading_title
                continue

            # Atomic elements (tables, code blocks)
            if el.element_type in ("table", "code"):
                el_tokens = count_tokens(text)
                if current_token_count > 0 and (
                    current_token_count + el_tokens > self.target_tokens
                ):
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

                    previous_overlap_text = get_trailing_token_overlap(
                        chunk.content, self.overlap_tokens
                    )
                    current_texts = (
                        [previous_overlap_text, text]
                        if previous_overlap_text
                        else [text]
                    )
                    current_token_count = count_tokens("\n\n".join(current_texts))
                    current_pages = {el_page} if el_page is not None else set()
                    current_sections = (
                        {el_section}
                        if el_section
                        else (
                            {current_section_title} if current_section_title else set()
                        )
                    )
                else:
                    current_texts.append(text)
                    current_token_count += el_tokens
                    if el_page is not None:
                        current_pages.add(el_page)
                    if el_section:
                        current_sections.add(el_section)
                        current_section_title = el_section
                last_sentence = None
                continue

            # Paragraph element: split into sentences for semantic breakpoint analysis
            sub_sentences = self._split_into_sentences(text)
            for sentence in sub_sentences:
                s_tokens = count_tokens(sentence)

                # Check semantic breakpoint if we have accumulated enough tokens
                semantic_break = False
                if last_sentence is not None and current_token_count >= self.min_tokens:
                    is_structure = bool(_STRUCTURE_REGEX.search(sentence.strip()))
                    if not is_structure:
                        sim = self.compute_sentence_similarity(last_sentence, sentence)
                        if sim < self.similarity_threshold:
                            semantic_break = True

                size_overflow = current_token_count > 0 and (
                    current_token_count + s_tokens > self.target_tokens
                )

                if semantic_break or size_overflow:
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

                    # Generate overlap tokens for seam mitigation
                    previous_overlap_text = get_trailing_token_overlap(
                        chunk.content, self.overlap_tokens
                    )
                    current_texts = (
                        [previous_overlap_text, sentence]
                        if previous_overlap_text
                        else [sentence]
                    )
                    current_token_count = count_tokens("\n\n".join(current_texts))
                    current_pages = {el_page} if el_page is not None else set()
                    current_sections = (
                        {el_section}
                        if el_section
                        else (
                            {current_section_title} if current_section_title else set()
                        )
                    )
                    if el_section:
                        current_section_title = el_section
                else:
                    current_texts.append(sentence)
                    current_token_count += s_tokens
                    if el_page is not None:
                        current_pages.add(el_page)
                    if el_section:
                        current_sections.add(el_section)
                        current_section_title = el_section

                last_sentence = sentence

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
        primary_page = sorted(pages)[0] if pages else None
        primary_section = sorted(sections)[0] if sections else section_title

        # Mitigate context degradation: inject section context if not already present in chunk content
        if (
            primary_section
            and primary_section != "General"
            and primary_section not in content
        ):
            content = f"## {primary_section}\n\n{content}"

        tokens = count_tokens(content)

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
            document_id=doc_id,
        )

        return DocumentChunk(
            id=meta["chunk_id"],
            document_id=doc_id,
            chunk_index=seq_num - 1,
            content=content,
            char_count=len(content),
            word_count=len(content.split()),
            token_count=tokens,
            page_numbers=sorted(pages),
            section_titles=sorted(sections),
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
        """Semantic chunker for raw plain text documents."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        inferred_type = MetadataTagger.infer_document_type(
            source_name=source_name,
            text_sample=text[:500],
            explicit_type=document_type,
        )

        # Break paragraphs into individual sentences
        units: list[str] = []
        for p in paragraphs:
            sub_sentences = self._split_into_sentences(p)
            if sub_sentences:
                units.extend(sub_sentences)
            else:
                units.append(p)

        chunks: list[DocumentChunk] = []
        current_texts: list[str] = []
        current_tokens = 0
        seq_num = 1
        previous_overlap = ""
        last_sentence: str | None = None

        for unit in units:
            unit_tokens = count_tokens(unit)

            # Check semantic breakpoint
            semantic_break = False
            if last_sentence is not None and current_tokens >= self.min_tokens:
                sim = self.compute_sentence_similarity(last_sentence, unit)
                if sim < self.similarity_threshold:
                    semantic_break = True

            size_overflow = current_tokens > 0 and (
                current_tokens + unit_tokens > self.target_tokens
            )

            if semantic_break or size_overflow:
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
                    document_id=doc_id,
                )
                chunk = DocumentChunk(
                    id=meta["chunk_id"],
                    document_id=doc_id,
                    chunk_index=seq_num - 1,
                    content=content,
                    char_count=len(content),
                    word_count=len(content.split()),
                    token_count=tokens,
                    metadata=meta,
                )
                chunks.append(chunk)
                seq_num += 1

                previous_overlap = get_trailing_token_overlap(
                    content, self.overlap_tokens
                )
                current_texts = [previous_overlap, unit] if previous_overlap else [unit]
                current_tokens = count_tokens("\n\n".join(current_texts))
            else:
                current_texts.append(unit)
                current_tokens += unit_tokens

            last_sentence = unit

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
                document_id=doc_id,
            )
            chunk = DocumentChunk(
                id=meta["chunk_id"],
                document_id=doc_id,
                chunk_index=seq_num - 1,
                content=content,
                char_count=len(content),
                word_count=len(content.split()),
                token_count=tokens,
                metadata=meta,
            )
            chunks.append(chunk)

        self._wire_chunk_pointers(chunks)
        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split a block of text into sentences, protecting abbreviations and decimal numbers."""
        if not text.strip():
            return []

        # Fast path: if no sentence punctuation, return as single sentence
        if "." not in text and "?" not in text and "!" not in text:
            return [text.strip()]

        protected = text
        for idx, abbr in enumerate(_PROTECTED_ABBREVIATIONS):
            if abbr in protected:
                protected = protected.replace(abbr, f"__ABBR_{idx}__")

        # Protect decimal numbers e.g. 3.14 -> 3__DOT__14
        if "." in protected:
            protected = _DECIMAL_REGEX.sub(r"\1__DOT__\2", protected)
            # Protect numbered list items e.g. 1. Item -> 1__LISTDOT__ Item
            protected = _LIST_DOT_REGEX.sub("__LISTDOT__", protected)

        # Split on sentence boundaries: period, question mark, or exclamation mark followed by whitespace
        parts = _SENTENCE_SPLIT_REGEX.split(protected)
        results: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if "__ABBR_" in part:
                for idx, abbr in enumerate(_PROTECTED_ABBREVIATIONS):
                    part = part.replace(f"__ABBR_{idx}__", abbr)
            if "__DOT__" in part or "__LISTDOT__" in part:
                part = part.replace("__DOT__", ".").replace("__LISTDOT__", ".")
            results.append(part)

        return results

    def _wire_chunk_pointers(self, chunks: list[DocumentChunk]) -> None:
        """Inject previous_chunk_id and next_chunk_id for traversal."""
        for i, chunk in enumerate(chunks):
            prev_id = chunks[i - 1].id if i > 0 else None
            next_id = chunks[i + 1].id if i < len(chunks) - 1 else None
            chunk.metadata["prev_chunk_id"] = prev_id
            chunk.metadata["next_chunk_id"] = next_id

    def create_small_to_big_chunks(
        self,
        parent_chunks: list[DocumentChunk],
        child_target_tokens: int = 100,
        child_overlap_tokens: int = 20,
    ) -> list[DocumentChunk]:
        """Decompose parent chunks into granular child chunks for Small-to-Big retrieval.

        Child chunks provide sharp, focused embeddings with high semantic proximity,
        while carrying pointers to their full parent context to avoid context degradation.
        """
        all_child_chunks: list[DocumentChunk] = []

        for parent in parent_chunks:
            sentences = self._split_into_sentences(parent.content)
            if not sentences or count_tokens(parent.content) <= child_target_tokens:
                child = parent.model_copy(deep=True)
                child.parent_id = parent.id
                child.parent_content = parent.content
                child.is_child = True
                child.metadata["parent_id"] = parent.id
                all_child_chunks.append(child)
                continue

            current_sentences: list[str] = []
            curr_tokens = 0
            sub_idx = 0

            for sent in sentences:
                s_tok = count_tokens(sent)
                if current_sentences and (curr_tokens + s_tok > child_target_tokens):
                    child_text = " ".join(current_sentences)
                    child_meta = dict(parent.metadata)
                    child_id = f"{parent.id}_c{sub_idx}"
                    child_meta["chunk_id"] = child_id
                    child_meta["parent_id"] = parent.id
                    child_meta["is_child"] = True

                    child_chunk = DocumentChunk(
                        id=child_id,
                        document_id=parent.document_id,
                        chunk_index=parent.chunk_index,
                        content=child_text,
                        char_count=len(child_text),
                        word_count=len(child_text.split()),
                        token_count=curr_tokens,
                        page_numbers=list(parent.page_numbers),
                        section_titles=list(parent.section_titles),
                        metadata=child_meta,
                        parent_id=parent.id,
                        parent_content=parent.content,
                        is_child=True,
                    )
                    all_child_chunks.append(child_chunk)
                    sub_idx += 1

                    overlap_text = get_trailing_token_overlap(
                        child_text, child_overlap_tokens
                    )
                    current_sentences = [overlap_text, sent] if overlap_text else [sent]
                    curr_tokens = count_tokens(" ".join(current_sentences))
                else:
                    current_sentences.append(sent)
                    curr_tokens += s_tok

            if current_sentences:
                child_text = " ".join(current_sentences)
                child_meta = dict(parent.metadata)
                child_id = f"{parent.id}_c{sub_idx}"
                child_meta["chunk_id"] = child_id
                child_meta["parent_id"] = parent.id
                child_meta["is_child"] = True

                child_chunk = DocumentChunk(
                    id=child_id,
                    document_id=parent.document_id,
                    chunk_index=parent.chunk_index,
                    content=child_text,
                    char_count=len(child_text),
                    word_count=len(child_text.split()),
                    token_count=curr_tokens,
                    page_numbers=list(parent.page_numbers),
                    section_titles=list(parent.section_titles),
                    metadata=child_meta,
                    parent_id=parent.id,
                    parent_content=parent.content,
                    is_child=True,
                )
                all_child_chunks.append(child_chunk)

        return all_child_chunks


default_semantic_chunker = SemanticChunker(
    target_tokens=350, overlap_tokens=50, use_embeddings=False
)

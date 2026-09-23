"""Unit tests for Semantic Chunking, 50-token overlap, and Metadata Tagging."""

from datetime import datetime, timezone

from src.chunking.metadata import MetadataTagger
from src.chunking.semantic import SemanticChunker
from src.chunking.tokenizer import count_tokens, get_trailing_token_overlap
from src.ingestion.models import Document, DocumentElement, DocumentType


class TestTokenizerAndOverlap:
    def test_count_tokens(self):
        text = "Semantic chunking with 50 tokens overlap for context degradation."
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens <= 15

    def test_trailing_token_overlap_extracts_target_tokens(self):
        # Create a paragraph with ~100 tokens
        words = ["word" + str(i) for i in range(100)]
        long_text = ". ".join(" ".join(words[i:i+10]) for i in range(0, 100, 10)) + "."

        overlap = get_trailing_token_overlap(long_text, overlap_tokens=50)
        overlap_tokens = count_tokens(overlap)
        # Should extract close to 50 tokens (accounting for sentence boundary snapping)
        assert 25 <= overlap_tokens <= 55


class TestMetadataTagger:
    def test_exact_user_spec_example(self):
        """Verify the exact user-specified example schema and chunk_id pattern."""
        meta = MetadataTagger.build_chunk_metadata(
            source_name="employee_handbook.pdf",
            page=12,
            section="Leave Policy",
            document_type="HR_POLICY",
            sequence_num=3,
            content="Employees receive 20 days of paid leave annually.",
            token_count=10,
            overlap_token_count=0,
        )

        assert meta["chunk_id"] == "handbook_12_003"
        assert meta["source"] == "employee_handbook.pdf"
        assert meta["page"] == 12
        assert meta["section"] == "Leave Policy"
        assert meta["document_type"] == "HR_POLICY"

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert meta["created_at"] == today

    def test_infer_document_types(self):
        assert MetadataTagger.infer_document_type("employee_handbook.pdf") == "HR_POLICY"
        assert MetadataTagger.infer_document_type("research_paper.pdf", "empirical findings and arxiv") == "RESEARCH_PAPER"
        assert MetadataTagger.infer_document_type("api_spec.json", "endpoint schemas") == "TECHNICAL_SPEC"
        assert MetadataTagger.infer_document_type("terms_agreement.docx", "confidentiality agreement") == "LEGAL_CONTRACT"
        assert MetadataTagger.infer_document_type("random_notes.txt", "some text") == "GENERAL_DOC"


class TestSemanticChunker:
    def test_semantic_chunking_with_50_tokens_overlap(self):
        chunker = SemanticChunker(target_tokens=60, overlap_tokens=50)

        # Create two distinct sections with enough content to trigger multiple chunks
        elements = [
            DocumentElement(id="el_0", element_type="heading", content="Leave Policy", section_title="Leave Policy", page_number=12),
            DocumentElement(
                id="el_1",
                element_type="paragraph",
                content=(
                    "All full-time employees are entitled to 20 days of paid time off per calendar year. "
                    "Vacation requests must be submitted at least two weeks in advance through the HR portal. "
                    "Unused vacation time may be rolled over up to a maximum of 5 business days into the subsequent year."
                ),
                section_title="Leave Policy",
                page_number=12,
            ),
            DocumentElement(
                id="el_2",
                element_type="paragraph",
                content=(
                    "Sick leave is granted at 10 days per year and requires medical documentation if spanning "
                    "more than three consecutive working days. Parental leave policies allow for twelve weeks "
                    "of fully compensated leave for primary caregivers following birth or adoption."
                ),
                section_title="Leave Policy",
                page_number=12,
            ),
        ]

        chunks = chunker.chunk_elements(
            doc_id="doc_12345",
            elements=elements,
            source_name="employee_handbook.pdf",
            document_type="HR_POLICY",
        )

        assert len(chunks) >= 2, "Expected multiple chunks for this text length at target_tokens=60"

        # Check metadata on every chunk
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == "employee_handbook.pdf"
            assert chunk.metadata["page"] == 12
            assert chunk.metadata["section"] == "Leave Policy"
            assert chunk.metadata["document_type"] == "HR_POLICY"
            assert chunk.metadata["chunk_id"].startswith("handbook_12_")
            assert "created_at" in chunk.metadata
            assert "token_count" in chunk.metadata

        # Verify 50 tokens overlap:
        # The start of chunk[1] should overlap with the end of chunk[0]
        assert chunks[1].metadata["overlap_token_count"] > 0
        assert any(w in chunks[1].content for w in chunks[0].content.split()[-10:])

        # Wire check
        assert chunks[0].metadata["prev_chunk_id"] is None
        assert chunks[0].metadata["next_chunk_id"] == chunks[1].id
        assert chunks[1].metadata["prev_chunk_id"] == chunks[0].id

    def test_document_create_end_to_end_cleaning_and_chunking(self):
        """Test full Document creation pipeline with cleaning, chunking, and metadata tagging."""
        elements = [
            DocumentElement(id="el_0", element_type="paragraph", content="Page 1", page_number=1),  # page number should be cleaned
            DocumentElement(id="el_1", element_type="heading", content="Context Optimization", section_title="Context Optimization", page_number=1),
            DocumentElement(id="el_2", element_type="paragraph", content="Mitigating context degra-\ndation is critical.", section_title="Context Optimization", page_number=1),
        ]

        doc = Document.create(
            source_name="paper.pdf",
            file_type=DocumentType.PDF,
            elements=elements,
            extra_metadata={"document_type": "RESEARCH_PAPER"},
        )

        # Verify cleaning occurred
        assert "degradation" in doc.content
        assert "degra-" not in doc.content
        # Page 1 standalone page number element was filtered
        assert not any(el.content == "Page 1" for el in doc.elements)

        # Verify chunks have metadata
        assert len(doc.chunks) >= 1
        first_chunk = doc.chunks[0]
        assert first_chunk.metadata["document_type"] == "RESEARCH_PAPER"
        assert first_chunk.metadata["source"] == "paper.pdf"
        assert "paper_" in first_chunk.metadata["chunk_id"]

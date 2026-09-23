"""Metadata tagging engine for document chunks."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MetadataTagger:
    """Tags document chunks with standardized metadata schemas and provenance."""

    @staticmethod
    def extract_source_slug(source_name: str) -> str:
        """Derive a clean, human-readable slug for chunk ID generation.

        e.g. 'employee_handbook.pdf' -> 'handbook' (or 'employee_handbook')
        """
        stem = Path(source_name).stem.lower()
        # Clean non-alphanumeric chars
        cleaned = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")

        # If composed of underscores, pick either significant keyword or first 2 words
        parts = [p for p in cleaned.split("_") if p]
        if not parts:
            return "doc"

        # If contains common domain suffixes like 'handbook', 'manual', 'policy', 'paper'
        for keyword in ("handbook", "policy", "spec", "report", "paper", "guide", "manual", "contract"):
            if keyword in parts:
                return keyword

        # Default to first 2 words or whole stem up to 16 chars
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"[:16]
        return parts[0][:16]

    @staticmethod
    def infer_document_type(
        source_name: str,
        text_sample: str = "",
        explicit_type: str | None = None,
    ) -> str:
        """Infer or format document_type (e.g. 'HR_POLICY', 'RESEARCH_PAPER', 'TECHNICAL_SPEC')."""
        if explicit_type and explicit_type.strip():
            # Format explicit type as UPPER_SNAKE_CASE
            formatted = re.sub(r"[^a-zA-Z0-9]+", "_", explicit_type.strip()).upper().strip("_")
            return formatted

        haystack = f"{source_name} {text_sample}".lower()

        # HR / Policy patterns
        if any(
            k in haystack
            for k in (
                "handbook",
                "leave policy",
                "hr policy",
                "employee",
                "benefits",
                "vacation",
                "sick leave",
                "workplace conduct",
                "human resources",
            )
        ):
            return "HR_POLICY"

        # Research / Academic patterns
        if any(
            k in haystack
            for k in (
                "arxiv",
                "abstract",
                "methodology",
                "experiments",
                "empirical findings",
                "related work",
                "context degradation",
                "hallucination",
                "rerank",
            )
        ):
            return "RESEARCH_PAPER"

        # Technical / API / Architecture patterns
        if any(
            k in haystack
            for k in (
                "architecture",
                "api reference",
                "specification",
                "endpoint",
                "schema",
                "sdk",
                "fastapi",
                "docker",
                "deployment",
            )
        ):
            return "TECHNICAL_SPEC"

        # Legal / Contract patterns
        if any(
            k in haystack
            for k in (
                "agreement",
                "terms and conditions",
                "non-disclosure",
                "confidentiality agreement",
                "indemnification",
                "governing law",
            )
        ):
            return "LEGAL_CONTRACT"

        # User guide / Manual
        if any(k in haystack for k in ("user guide", "instruction manual", "quickstart", "how to")):
            return "USER_MANUAL"

        # Financial
        if any(k in haystack for k in ("balance sheet", "quarterly earnings", "revenue", "fiscal year", "10-k")):
            return "FINANCIAL_REPORT"

        return "GENERAL_DOC"

    @classmethod
    def generate_chunk_id(
        cls,
        source_name: str,
        page: int | None,
        section: str | None,
        sequence_num: int,
    ) -> str:
        """Generate a clean semantic chunk ID adhering to the format: {slug}_{page}_{seq:03d}.

        e.g. 'handbook_12_003'
        """
        slug = cls.extract_source_slug(source_name)
        page_part = str(page) if page is not None else "0"
        return f"{slug}_{page_part}_{sequence_num:03d}"

    @classmethod
    def build_chunk_metadata(
        cls,
        source_name: str,
        page: int | None,
        section: str | None,
        document_type: str,
        sequence_num: int,
        content: str,
        token_count: int,
        overlap_token_count: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construct the standardized metadata payload for a chunk.

        Matches exact user specification:
        {
          "chunk_id": "handbook_12_003",
          "source": "employee_handbook.pdf",
          "page": 12,
          "section": "Leave Policy",
          "document_type": "HR_POLICY",
          "created_at": "2026-09-19"
        }
        """
        chunk_id = cls.generate_chunk_id(source_name, page, section, sequence_num)
        created_at_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        meta: dict[str, Any] = {
            "chunk_id": chunk_id,
            "source": source_name,
            "page": page,
            "section": section or "General",
            "document_type": document_type,
            "created_at": created_at_date,
            # Provenance & downstream retrieval indicators
            "token_count": token_count,
            "char_count": len(content),
            "word_count": len(content.split()),
            "overlap_token_count": overlap_token_count,
            "has_table": "|" in content and re.search(r"\|.*\|", content) is not None,
            "has_code": "```" in content,
        }

        if extra:
            for k, v in extra.items():
                if k not in meta:
                    meta[k] = v

        return meta

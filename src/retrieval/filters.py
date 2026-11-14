"""Unified metadata filtering utilities for dense and sparse retrieval."""

from __future__ import annotations

import re
from typing import Any


def matches_metadata_filters(record: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Check whether a chunk metadata record satisfies all filter criteria.

    Supports:
      - Key aliases ('document_id' vs 'doc_id', 'source' vs 'source_name', etc.)
      - Enums (e.g. DocumentType.PDF -> 'pdf')
      - Collections in filter values (e.g. filters={'document_id': ['doc1', 'doc2']})
      - Collections in record values (e.g. page_numbers=[1, 2] with target_val=1)
      - Exact and case-insensitive string matching
      - Numeric equivalence (e.g. 2024 == '2024')
      - Semantic regex fallback for 'year', 'department', 'document_type', and 'section'
    """
    if not filters:
        return True

    chunk_meta = record.get("metadata", {}) or {}
    content = record.get("content", "") or ""
    source = str(
        chunk_meta.get("source", "")
        or chunk_meta.get("source_name", "")
        or record.get("source_name", "")
        or ""
    )
    section = str(
        chunk_meta.get("section", "")
        or chunk_meta.get("section_title", "")
        or ""
    )

    for key, target_val in filters.items():
        if target_val is None:
            continue

        # Extract Enum values (e.g. DocumentType.PDF -> "pdf")
        if hasattr(target_val, "value"):
            target_val = target_val.value

        norm_key = key.lower().strip()
        val = chunk_meta.get(key)
        if val is None:
            val = chunk_meta.get(norm_key)
        if val is None:
            val = record.get(key)
        if val is None:
            val = record.get(norm_key)

        # Alias resolution
        if norm_key in ("doc_id", "document_id"):
            val = (
                val
                or chunk_meta.get("document_id")
                or chunk_meta.get("doc_id")
                or record.get("document_id")
                or record.get("doc_id")
            )
        elif norm_key in ("source", "source_name", "file_name"):
            val = val or source
        elif norm_key in ("section", "section_title"):
            val = val or section or chunk_meta.get("section_titles")
        elif norm_key in ("page", "page_number", "page_numbers"):
            val = (
                val
                or chunk_meta.get("page")
                or chunk_meta.get("page_number")
                or chunk_meta.get("page_numbers")
            )
        elif norm_key in ("document_type", "doc_type", "file_type"):
            val = (
                val
                or chunk_meta.get("document_type")
                or chunk_meta.get("doc_type")
                or chunk_meta.get("file_type")
                or record.get("file_type")
            )

        # Handle list/collection of target values (e.g. filters={"document_id": ["doc1", "doc2"]})
        if isinstance(target_val, (list, tuple, set)):
            target_strs = {str(t).strip().lower() for t in target_val if t is not None}
            if not target_strs:
                continue
            if isinstance(val, (list, tuple, set)):
                if any(str(v).strip().lower() in target_strs for v in val):
                    continue
            elif val is not None and str(val).strip().lower() in target_strs:
                continue
            return False

        target_str = str(target_val).strip().lower()

        # Handle list/collection in record value (e.g. page_numbers=[1, 2] with target_val=1)
        if isinstance(val, (list, tuple, set)):
            if any(str(v).strip().lower() == target_str for v in val):
                continue
            try:
                target_num = float(target_val)
                if any(
                    float(v) == target_num
                    for v in val
                    if isinstance(v, (int, float, str)) and str(v).replace(".", "", 1).isdigit()
                ):
                    continue
            except (ValueError, TypeError):
                pass
            return False

        # Exact or case-insensitive string match
        if val is not None:
            if str(val).strip().lower() == target_str:
                continue
            try:
                if float(val) == float(target_val):
                    continue
            except (ValueError, TypeError):
                pass

        # Multi-attribute fallback resolution for semantic filter dimensions
        if norm_key == "year":
            year_pat = rf"\b{re.escape(target_str)}\b"
            if (
                re.search(year_pat, source.lower())
                or re.search(year_pat, content.lower())
                or re.search(year_pat, str(chunk_meta.get("created_at", "")))
            ):
                continue
            return False

        if norm_key == "department":
            dept_pat = rf"\b{re.escape(target_str)}\b"
            if (
                re.search(dept_pat, source.lower())
                or re.search(dept_pat, section.lower())
                or re.search(dept_pat, content.lower())
            ):
                continue
            return False

        if norm_key in ("document_type", "doc_type", "file_type"):
            norm_doc_type = str(chunk_meta.get("document_type", "")).strip().lower()
            norm_file_type = str(
                record.get("file_type", "") or chunk_meta.get("file_type", "")
            ).strip().lower()
            if (
                target_str in norm_doc_type
                or norm_doc_type in target_str
                or target_str in norm_file_type
                or norm_file_type in target_str
            ):
                continue
            return False

        if norm_key in ("section", "section_title"):
            sec_pat = rf"\b{re.escape(target_str)}\b"
            if re.search(sec_pat, section.lower()) or re.search(sec_pat, content.lower()):
                continue
            return False

        if val is None:
            return False

        if str(val).strip().lower() != target_str:
            return False

    return True

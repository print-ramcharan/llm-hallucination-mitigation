"""Filter extraction engine for structured metadata filtering."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple


class FilterExtractor:
    """Extracts structured metadata filters (year, department, document_type) from user queries

    and produces a distilled, cleaned search query.
    """

    KNOWN_DEPARTMENTS = {
        "engineering": "engineering",
        "tech": "engineering",
        "software": "engineering",
        "hr": "hr",
        "human resources": "hr",
        "finance": "finance",
        "accounting": "finance",
        "marketing": "marketing",
        "legal": "legal",
        "operations": "operations",
        "ops": "operations",
        "sales": "sales",
        "product": "product",
        "security": "security",
        "it": "it",
    }

    KNOWN_DOC_TYPES = {
        "handbook": "HANDBOOK",
        "policy": "POLICY",
        "guidelines": "GUIDELINES",
        "contract": "LEGAL_CONTRACT",
        "specification": "TECHNICAL_SPEC",
        "spec": "TECHNICAL_SPEC",
        "report": "RESEARCH_PAPER",
    }

    def __init__(self) -> None:
        self.year_pattern = re.compile(r"\b(19\d{2}|20\d{2})\b")
        # Build department regex pattern sorted by length descending so "human resources" matches before "hr"
        sorted_depts = sorted(self.KNOWN_DEPARTMENTS.keys(), key=len, reverse=True)
        self.dept_pattern = re.compile(r"\b(" + "|".join(re.escape(d) for d in sorted_depts) + r")\b", re.IGNORECASE)

    def extract(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Extract metadata filters and return (cleaned_search_query, filters)."""
        raw_q = query.strip()
        filters: Dict[str, Any] = {}

        # 1. Extract Year
        year_match = self.year_pattern.search(raw_q)
        if year_match:
            filters["year"] = int(year_match.group(1))

        # 2. Extract Department
        dept_match = self.dept_pattern.search(raw_q)
        detected_dept = None
        if dept_match:
            matched_dept_str = dept_match.group(1).lower()
            detected_dept = self.KNOWN_DEPARTMENTS[matched_dept_str]
            filters["department"] = detected_dept

        # 3. Clean search query
        cleaned_query = self._clean_search_query(raw_q, filters)

        return cleaned_query, filters

    def _clean_search_query(self, query: str, filters: Dict[str, Any]) -> str:
        """Derive an optimal search query representation based on extracted entities."""
        q = query.strip()

        # Specific pattern: "What did the [YEAR] [DEPT] policy say about [TOPIC]?"
        # -> "[DEPT] [TOPIC] policy"
        m1 = re.match(
            r"^what\s+did\s+(?:the\s+)?(?:\d{4}\s+)?([a-z0-9_\-\s]+?)\s+(?:policy|handbook|guidelines)\s+say\s+about\s+(.+?)[?!.,;:]*$",
            q,
            re.IGNORECASE,
        )
        if m1:
            dept_term = m1.group(1).strip()
            topic_term = m1.group(2).strip()
            return f"{dept_term} {topic_term} policy"

        # Specific pattern: "What does the [DEPT] handbook/policy say about [TOPIC]?"
        # -> "[TOPIC]"
        m2 = re.match(
            r"^what\s+does\s+(?:the\s+)?([a-z0-9_\-\s]+?)\s+(?:policy|handbook|guidelines)\s+say\s+about\s+(.+?)[?!.,;:]*$",
            q,
            re.IGNORECASE,
        )
        if m2:
            return m2.group(2).strip()

        # Specific pattern: "What was the [YEAR] [TOPIC]?"
        # -> "[TOPIC]"
        m3 = re.match(
            r"^what\s+(?:was|is)\s+(?:the\s+)?(?:\d{4}\s+)?(.+?)[?!.,;:]*$",
            q,
            re.IGNORECASE,
        )
        if m3:
            return m3.group(1).strip()

        # Remove years from search query if extracted
        if "year" in filters:
            q = re.sub(r"\b" + str(filters["year"]) + r"\b", "", q)

        # Remove question conversational prefixes
        q = re.sub(r"^(?:what\s+(?:is|was|are|were)|tell\s+me\s+about|how\s+does|how\s+do)\s+(?:our\s+|the\s+)?", "", q, flags=re.IGNORECASE)
        q = re.sub(r"^what\s+does\s+the\s+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\s+say\s+about\s+", " ", q, flags=re.IGNORECASE)
        q = re.sub(r"[?!.,;:]+$", "", q).strip()

        # Normalize whitespace
        tokens = [t for t in q.split() if t]
        return " ".join(tokens)

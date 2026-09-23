"""Intent detection engine using rule-based classification."""

from __future__ import annotations

import re

from src.query_processing.models import QueryIntent


class IntentDetector:
    """Classifies user query intent using fast pattern matching and lexical heuristics."""

    COMPARISON_PATTERNS = [
        r"\b(?:compare|comparison|contrasting|contrasted)\b",
        r"\b(?:difference|differences)\s+between\b",
        r"\b(?:vs\.?|versus)\b",
        r"\b(?:how\s+does\s+.+?\s+compare\s+to\b)",
        r"\b(?:better\s+than|worse\s+than|different\s+from)\b",
    ]

    SUMMARY_PATTERNS = [
        r"\b(?:summarize|summarise|summary|summarization)\b",
        r"\b(?:give\s+me\s+(?:a\s+)?(?:brief\s+)?overview)\b",
        r"\b(?:tl;?dr|brief\s+summary|wrap\s+up|main\s+takeaways)\b",
        r"\b(?:high[- ]level\s+overview)\b",
    ]

    EXPLORATORY_PATTERNS = [
        r"\b(?:what\s+topics|what\s+areas|what\s+sections)\s+(?:are\s+covered|exist)\b",
        r"\b(?:what\s+(?:policies|documents|guidelines)\s+(?:do\s+we\s+have|are\s+available))\b",
        r"\b(?:explore|browse|overview\s+of\s+all)\b",
        r"\b(?:list\s+all\s+(?:the\s+)?(?:available\s+)?(?:policies|options|benefits|categories))\b",
    ]

    def __init__(self) -> None:
        self._compiled_comparison = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self._compiled_summary = [re.compile(p, re.IGNORECASE) for p in self.SUMMARY_PATTERNS]
        self._compiled_exploratory = [re.compile(p, re.IGNORECASE) for p in self.EXPLORATORY_PATTERNS]

    def detect(self, query: str) -> QueryIntent:
        """Detect intent from the query string."""
        q = query.strip()
        if not q:
            return QueryIntent.INFORMATIONAL

        # 1. Comparison check
        if any(pat.search(q) for pat in self._compiled_comparison):
            return QueryIntent.COMPARISON

        # 2. Summary check
        if any(pat.search(q) for pat in self._compiled_summary):
            return QueryIntent.SUMMARY

        # 3. Exploratory check
        if any(pat.search(q) for pat in self._compiled_exploratory):
            return QueryIntent.EXPLORATORY

        # 4. Keyword lookup check (few words, no question starter or punctuation)
        words = q.split()
        is_question = any(q.lower().startswith(w) for w in ["what", "how", "when", "where", "why", "who", "which", "can", "is", "are"])
        if len(words) <= 3 and not is_question and not q.endswith("?"):
            return QueryIntent.KEYWORD_LOOKUP

        # 5. Default is Informational
        return QueryIntent.INFORMATIONAL

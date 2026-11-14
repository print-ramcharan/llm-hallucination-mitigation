"""Key Evidence Extractor & Sentence-Level Pruner."""

from __future__ import annotations

import re

# Common stop words and document-framing meta words to exclude
_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "doing",
    "that",
    "this",
    "these",
    "those",
    "one",
    "two",
    "regarding",
    "regard",
    "regards",
    "considered",
    "considering",
    "consider",
    "considers",
    "include",
    "includes",
    "including",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "them",
    "their",
    "what",
    "which",
    # Document framing / meta terms
    "document",
    "documents",
    "doc",
    "docs",
    "file",
    "files",
    "text",
    "texts",
    "article",
    "articles",
    "paper",
    "papers",
    "passage",
    "passages",
    "content",
    "contents",
    "context",
    "contexts",
    "mention",
    "mentions",
    "mentioned",
    "mentioning",
    "give",
    "given",
    "provide",
    "provided",
    "state",
    "stated",
    "states",
    "tell",
    "told",
    "say",
    "said",
    "says",
    "list",
    "listed",
    "lists",
    "show",
    "shows",
    "describe",
    "describes",
    "explain",
    "explains",
}

_SEMANTIC_SYNONYMS: dict[str, set[str]] = {
    "skill": {
        "qualification",
        "qualifications",
        "requirement",
        "requirements",
        "competency",
        "competencies",
        "proficiency",
        "proficiencies",
        "ability",
        "abilities",
        "knowledge",
        "fundamentals",
        "expertise",
        "experience",
        "programming",
        "technology",
        "technologies",
        "tools",
        "python",
        "sql",
        "engineering",
        "responsibilities",
        "duties",
    },
    "skills": {
        "qualification",
        "qualifications",
        "requirement",
        "requirements",
        "competency",
        "competencies",
        "proficiency",
        "proficiencies",
        "ability",
        "abilities",
        "knowledge",
        "fundamentals",
        "expertise",
        "experience",
        "programming",
        "technology",
        "technologies",
        "tools",
        "python",
        "sql",
        "engineering",
        "responsibilities",
        "duties",
    },
    "qualification": {
        "skill",
        "skills",
        "requirement",
        "requirements",
        "fundamentals",
        "experience",
        "education",
    },
    "qualifications": {
        "skill",
        "skills",
        "requirement",
        "requirements",
        "fundamentals",
        "experience",
        "education",
    },
    "responsibility": {
        "duties",
        "tasks",
        "role",
        "work",
        "responsibilities",
        "solutions",
        "develop",
        "build",
    },
    "responsibilities": {
        "duties",
        "tasks",
        "role",
        "work",
        "responsibility",
        "solutions",
        "develop",
        "build",
    },
    "salary": {
        "compensation",
        "pay",
        "rate",
        "wage",
        "wages",
        "stipend",
        "remuneration",
        "benefits",
    },
    "leave": {"vacation", "pto", "holiday", "holidays", "absence", "time off"},
    "vacation": {"leave", "pto", "holiday", "holidays", "absence", "time off"},
    "benefit": {"perk", "perks", "insurance", "401k", "pension", "health", "dental"},
    "benefits": {"perk", "perks", "insurance", "401k", "pension", "health", "dental"},
}

_SENTENCE_SPLIT_REGEX = re.compile(
    r"(?<!\b\d)(?<!\b[A-Za-z])(?<=[.!?])\s+(?=[A-Z0-9\"\'\(\[])"
)
_WORD_REGEX = re.compile(r"\b[a-zA-Z0-9_\-\$]+\b")
_FORMULA_MARKER_REGEX = re.compile(r"(?:[=≡≤≥×÷±∑∏<>]|\b(?:mod|gcd)\b)", re.IGNORECASE)
_SUFFIXES = (
    "ing",
    "tion",
    "ment",
    "able",
    "ness",
    "ity",
    "ed",
    "ly",
    "es",
    "al",
    "ure",
    "ic",
    "s",
)


class SentencePruner:
    """Extracts salient sentence boundaries matching query intent and prunes filler prose."""

    def __init__(self, default_min_salience: float = 0.15) -> None:
        self.default_min_salience = default_min_salience

    @staticmethod
    def _stem_word(w: str) -> str:
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) > len(suf) + 3:
                return w[: -len(suf)]
        return w

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split text into individual sentences while preserving sentence content and numbered lists."""
        if not text or not text.strip():
            return []

        # Rejoin solitary list markers on their own line with the following text line
        text = re.sub(
            r"(?m)^(\s*(?:\d+[\.\)]|\(\d+\)|\[\d+\]|[a-zA-Z][\.\)]|\([a-zA-Z]\)|[•\-\*]))\s*\n+\s*([A-Za-z0-9\"\'\(\[])",
            r"\1 \2",
            text,
        )

        # Protect numbered list markers (e.g. "1. ", "12. ") from erroneous sentence splitting
        protected = re.sub(r"\b(\d+)\.(\s+)", r"\1__LISTDOT__\2", text)

        # First split on line breaks to preserve structural paragraphs
        lines = [line.strip() for line in protected.splitlines() if line.strip()]

        # Second pass to merge any remaining dangling list markers with the next line
        merged_lines: list[str] = []
        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            cleaned_line = line.replace("__LISTDOT__", ".").strip()
            if i + 1 < len(lines) and re.match(
                r"^(\d+[\.\)]|\(\d+\)|\[\d+\]|[a-zA-Z][\.\)]|\([a-zA-Z]\)|[•\-\*])$",
                cleaned_line,
            ):
                next_line = lines[i + 1]
                merged_lines.append(f"{cleaned_line}__LISTDOT__ {next_line}")
                skip_next = True
            else:
                merged_lines.append(line)

        sentences: list[str] = []
        for line in merged_lines:
            parts = _SENTENCE_SPLIT_REGEX.split(line)
            for part in parts:
                cleaned = part.replace("__LISTDOT__", ".").strip()
                if cleaned:
                    sentences.append(cleaned)

        return sentences

    @staticmethod
    def _extract_terms(text: str) -> set[str]:
        """Extract lowercase content terms excluding stop words."""
        words = _WORD_REGEX.findall(text.lower())
        return {w for w in words if w not in _STOP_WORDS and len(w) > 1}

    @staticmethod
    def _is_formula_sentence(sentence: str) -> bool:
        """Identify standalone mathematical expressions that have few natural-language terms."""
        return bool(_FORMULA_MARKER_REGEX.search(sentence))

    @staticmethod
    def _is_list_or_schedule_item(sentence: str) -> bool:
        """Identify bullet points, numbered items, module schedule lines, and table rows."""
        s = sentence.strip()
        if not s:
            return False
        if s.startswith(("•", "-", "*", ">", "–", "—", "+")):
            return True
        if re.match(r"^(\d+[\.\)]|\(\d+\)|\[\d+\]|[a-zA-Z][\.\)]|\([a-zA-Z]\))\s+", s):
            return True
        if re.search(r"\b(?:Admin|Dev|SME|Module|Superbadge|Project|Demo Link)\b$", s):
            return True
        return False

    def compute_salience(self, query_terms: set[str], sentence: str) -> float:
        """Compute the informational salience score of a sentence relative to the query.

        Combines:
          1. Query term and synonym overlap ratio
          2. Specific factual token density (numbers, currencies, percentages, dates)
        """
        if not query_terms or not sentence.strip():
            return 0.5  # Neutral default when query is empty

        sentence_terms = self._extract_terms(sentence)
        if not sentence_terms:
            return 0.0

        # Direct term overlap
        shared_terms = query_terms.intersection(sentence_terms)

        # Semantic synonym & stem expansion check for remaining query terms
        unmatched_query_terms = query_terms - shared_terms
        for qt in unmatched_query_terms:
            syns = _SEMANTIC_SYNONYMS.get(qt, set())
            if syns.intersection(sentence_terms):
                shared_terms.add(qt)
                continue

            qt_stem = self._stem_word(qt)
            syns_stem = _SEMANTIC_SYNONYMS.get(qt_stem, set())
            if syns_stem.intersection(sentence_terms):
                shared_terms.add(qt)
                continue

            for st in sentence_terms:
                st_stem = self._stem_word(st)
                if qt_stem == st_stem:
                    shared_terms.add(qt)
                    break
                min_len = min(len(qt), len(st))
                if min_len >= 5 and (qt[: min_len - 1] == st[: min_len - 1]):
                    shared_terms.add(qt)
                    break

        term_overlap = len(shared_terms) / max(len(query_terms), 1)

        # Factual density boost (digits, currency, percentages, e.g. "20 days", "$500")
        density_boost = 0.0
        if shared_terms:
            has_digits = bool(re.search(r"\b\d+\b", sentence))
            has_currency = bool(re.search(r"[\$\€\£\¥%]", sentence))
            if has_digits or has_currency:
                density_boost = 0.15

        # Jaccard overlap component
        union_len = len(query_terms.union(sentence_terms))
        jaccard = len(shared_terms) / union_len if union_len > 0 else 0.0

        salience = (0.7 * term_overlap) + (0.15 * jaccard) + density_boost
        return min(max(salience, 0.0), 1.0)

    def prune_chunk(
        self,
        query: str,
        text: str,
        min_salience: float | None = None,
    ) -> tuple[str, list[str]]:
        """Filter out non-relevant sentences from a chunk, preserving key evidence.

        Args:
            query: User search query.
            text: Raw chunk passage text.
            min_salience: Minimum threshold required to keep a sentence.

        Returns:
            Tuple of (compacted_text, list_of_retained_sentences).
        """
        threshold = self.default_min_salience if min_salience is None else min_salience
        sentences = self.split_sentences(text)

        if not sentences:
            return "", []

        # If chunk is very short (1-2 sentences), avoid over-pruning
        if len(sentences) <= 2:
            return text.strip(), sentences

        query_terms = self._extract_terms(query)

        scored_sentences: list[tuple[str, float]] = []
        for s in sentences:
            score = self.compute_salience(query_terms, s)
            scored_sentences.append((s, score))

        # Preserve equations and structured lists/schedules when other evidence in this chunk is query-relevant.
        has_relevant_context = any(
            score >= threshold and not self._is_formula_sentence(sentence)
            for sentence, score in scored_sentences
        )
        has_schedule_context = any(
            bool(
                re.search(
                    r"\b(?:schedule|modules?|curriculum|topics?|agenda|syllabus)\b",
                    sentence,
                    re.IGNORECASE,
                )
            )
            and score >= threshold
            for sentence, score in scored_sentences
        )

        is_in_relevant_list = False
        in_item_body = False
        kept: list[str] = []

        for sentence, score in scored_sentences:
            is_list_item = self._is_list_or_schedule_item(sentence)

            is_list_lead_in = bool(
                re.search(
                    r"(?:following|below|include|includes|considers?|questions?|factors?|steps?|ways?):\s*$",
                    sentence.strip(),
                    re.IGNORECASE,
                )
                or (sentence.strip().endswith(":") and score >= threshold)
            )

            if is_list_lead_in and score >= threshold:
                is_in_relevant_list = True
                in_item_body = False

            if is_list_item:
                in_item_body = has_relevant_context or is_in_relevant_list

            # If we hit a new structural section or heading, reset list state
            if sentence.startswith(("#", "---")):
                is_in_relevant_list = False
                in_item_body = False

            if (
                score >= threshold
                or (has_relevant_context and self._is_formula_sentence(sentence))
                or (has_schedule_context and is_list_item)
                or (has_relevant_context and is_list_item)
                or (is_in_relevant_list and is_list_item)
                or (in_item_body and (has_relevant_context or is_in_relevant_list))
            ):
                kept.append(sentence)

        # Fallback preservation: If no sentence met threshold
        if not kept:
            # First preference: sentences with positive salience
            positive_sentences = [s for s, score in scored_sentences if score > 0]
            if positive_sentences:
                kept = positive_sentences
            else:
                # If all scored 0 (e.g. semantic conceptual query), retain full sentences
                # so crucial domain context is never discarded
                kept = sentences

        compacted = " ".join(kept)
        return compacted, kept

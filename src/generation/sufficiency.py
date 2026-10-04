"""Evidence Sufficiency Classifier & Truthful Abstention Protocol."""

from __future__ import annotations

import re

from src.context.models import CompactedEvidence, OptimizedContext
from src.generation.models import SufficiencyAssessment

# Common stop words and query filler tokens to ignore when evaluating factual coverage
# Common stop words, query filler tokens, and document-framing meta words
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "about",
    "by", "of", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "what", "where", "when", "why", "how", "who", "which", "whose", "whom",
    "can", "could", "should", "would", "do", "does", "did", "please", "tell",
    "me", "explain", "describe", "detail", "give", "show", "find", "say", "according",
    "into", "onto", "unto", "within", "without", "through", "over", "under", "above",
    "below", "between", "among", "during", "before", "after", "across", "behind",
    "per", "via", "it", "its", "they", "them", "their", "this", "that", "these", "those",
    # Document framing / meta terms
    "document", "documents", "doc", "docs", "file", "files", "pdf", "docx", "txt",
    "text", "texts", "article", "articles", "paper", "papers", "report", "reports",
    "passage", "passages", "content", "contents", "context", "contexts", "page", "pages",
    "mention", "mentions", "mentioned", "mentioning",
    "state", "states", "stated", "stating", "list", "listed", "lists", "listing",
}

_SEMANTIC_SYNONYMS: dict[str, set[str]] = {
    "skill": {
        "qualification", "qualifications", "requirement", "requirements",
        "competency", "competencies", "proficiency", "proficiencies",
        "ability", "abilities", "knowledge", "fundamentals", "expertise",
        "experience", "programming", "technology", "technologies", "tools",
        "python", "sql", "engineering", "responsibilities", "duties", "internship",
    },
    "skills": {
        "qualification", "qualifications", "requirement", "requirements",
        "competency", "competencies", "proficiency", "proficiencies",
        "ability", "abilities", "knowledge", "fundamentals", "expertise",
        "experience", "programming", "technology", "technologies", "tools",
        "python", "sql", "engineering", "responsibilities", "duties", "internship",
    },
    "qualification": {"skill", "skills", "requirement", "requirements", "fundamentals", "experience", "education", "degree"},
    "qualifications": {"skill", "skills", "requirement", "requirements", "fundamentals", "experience", "education", "degree"},
    "responsibility": {"duties", "tasks", "role", "work", "responsibilities", "solutions", "develop", "build"},
    "responsibilities": {"duties", "tasks", "role", "work", "responsibility", "solutions", "develop", "build"},
    "salary": {"compensation", "pay", "rate", "wage", "wages", "stipend", "remuneration", "benefits"},
    "leave": {"vacation", "pto", "holiday", "holidays", "absence", "time off"},
    "vacation": {"leave", "pto", "holiday", "holidays", "absence", "time off"},
    "benefit": {"perk", "perks", "insurance", "401k", "pension", "health", "dental"},
    "benefits": {"perk", "perks", "insurance", "401k", "pension", "health", "dental"},
}


class EvidenceSufficiencyClassifier:
    """Evaluates whether retrieved evidence context adequately answers the query.

    If the quantified sufficiency score is below theta (default 0.40), triggers the
    Truthful Abstention Protocol to prevent hallucinated generation on out-of-domain,
    unanswerable, or missing-evidence queries.
    """

    def __init__(self, default_threshold: float = 0.40) -> None:
        self.default_threshold = default_threshold

    @staticmethod
    def extract_informative_terms(text: str) -> list[str]:
        """Extract meaningful factual query terms, entities, and keywords."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        return [t for t in tokens if t not in _STOP_WORDS]

    @staticmethod
    def extract_topic(query: str) -> str:
        """Extract focal topic or target question phrase from the query."""
        cleaned = re.sub(
            r"^(can you tell me about|please explain|what are the|what is the|what is|what are|where is|where are|who is|who are|how many|how much|what|where|when|why|how|who|is the|are the)\s+",
            "",
            query.strip(),
            flags=re.IGNORECASE,
        ).rstrip("?., ")
        # Also clean trailing "mentioned in the document / text / file"
        cleaned = re.sub(
            r"\s+(?:mentioned\s+in|stated\s+in|listed\s+in|given\s+in|described\s+in|found\s+in)\s+(?:the\s+)?(?:document|doc|file|text|passage|paper|pdf|docx)s?$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s+in\s+(?:the\s+)?(?:document|doc|file|text|passage|paper|pdf|docx)s?$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip() if cleaned.strip() else query.strip()

    def assess_sufficiency(
        self,
        query: str,
        context: OptimizedContext | None = None,
        evidence_items: list[CompactedEvidence] | None = None,
        threshold: float | None = None,
    ) -> SufficiencyAssessment:
        """Evaluate evidence coverage against query information requirements.

        Args:
            query: The user query string.
            context: Optional OptimizedContext from Module 5.
            evidence_items: Optional direct list of CompactedEvidence items.
            threshold: Optional custom sufficiency cutoff threshold.

        Returns:
            SufficiencyAssessment detailing sufficiency status, score, topic, and reasoning.
        """
        tau = threshold if threshold is not None else self.default_threshold
        topic = self.extract_topic(query)

        # Collect evidence passages
        items: list[CompactedEvidence] = []
        if context is not None and context.evidence_items:
            items = context.evidence_items
        elif evidence_items:
            items = evidence_items

        if not items:
            abstention_msg = (
                f"The available documents do not contain sufficient evidence to verify {topic}."
            )
            return SufficiencyAssessment(
                is_sufficient=False,
                sufficiency_score=0.0,
                threshold=tau,
                topic=topic,
                abstention_message=abstention_msg,
                reasoning="No evidence passages were retrieved or retained in the context window.",
                matched_aspects=[],
                missing_aspects=self.extract_informative_terms(query),
            )

        # Aggregate evidence text
        combined_evidence = " ".join(
            f"{item.extracted_text} {item.original_text}".lower() for item in items
        )

        query_terms = self.extract_informative_terms(query)
        if not query_terms:
            # Query is purely generic/conversational (e.g. "Hello there")
            return SufficiencyAssessment(
                is_sufficient=True,
                sufficiency_score=1.0,
                threshold=tau,
                topic=topic,
                abstention_message=None,
                reasoning="Query contains conversational greeting without specific factual constraint.",
                matched_aspects=[],
                missing_aspects=[],
            )

        # Analyze token coverage and aspect matching with stem & semantic synonym expansion
        matched_aspects: list[str] = []
        missing_aspects: list[str] = []

        for term in query_terms:
            stem = term.rstrip("s").rstrip("es")
            syns = _SEMANTIC_SYNONYMS.get(term, set()) | _SEMANTIC_SYNONYMS.get(stem, set())

            # 1. Exact term match or stem match in evidence text
            if term in combined_evidence or (len(stem) >= 3 and stem in combined_evidence):
                matched_aspects.append(term)
            # 2. Semantic synonym match in evidence text
            elif any(syn in combined_evidence for syn in syns):
                matched_aspects.append(term)
            # 3. Substring match for longer words
            elif len(stem) >= 5 and any(stem[:4] in word for word in combined_evidence.split()):
                matched_aspects.append(term)
            else:
                missing_aspects.append(term)

        term_coverage = len(matched_aspects) / len(query_terms) if query_terms else 0.0

        # Query intent & entity constraints check
        numeric_need = bool(re.search(r"\b(how many|how much|days|hours|percentage|cost|amount|date|year)\b", query, re.I))
        numeric_found = bool(re.search(r"\b\d+(\.\d+)?\b", combined_evidence))
        numeric_alignment = 1.0 if (not numeric_need or numeric_found) else 0.4

        # Evidence quality multiplier (average salience or rerank score)
        salience_scores = [
            float(item.metadata.get("salience_score", item.metadata.get("rerank_score", 0.5)))
            for item in items
        ]
        avg_salience = sum(salience_scores) / len(salience_scores) if salience_scores else 0.5
        clamped_salience = max(0.0, min(1.0, avg_salience))

        # Sufficiency score calculation:
        # Factual and semantic coverage across concepts
        if term_coverage >= 0.40:
            sufficiency_score = (
                0.60 * term_coverage
                + 0.20 * numeric_alignment
                + 0.20 * clamped_salience
            )
        elif term_coverage > 0:
            sufficiency_score = (
                0.45 * term_coverage
                + 0.25 * numeric_alignment
                + 0.30 * clamped_salience
            )
        else:
            sufficiency_score = 0.25 * clamped_salience

        sufficiency_score = round(max(0.0, min(1.0, sufficiency_score)), 4)
        is_sufficient = sufficiency_score >= tau

        if not is_sufficient:
            missing_desc = (
                f" Specifically, verified details on {', '.join(missing_aspects[:4])} were not found in the indexed sources."
                if missing_aspects
                else ""
            )
            abstention_msg = (
                f"The available documents do not contain sufficient evidence to verify {topic}.{missing_desc}"
            )
            reasoning = (
                f"Evidence sufficiency score ({sufficiency_score:.2f}) falls below threshold ({tau:.2f}). "
                f"Missing critical factual aspects: {', '.join(missing_aspects) if missing_aspects else 'None'}."
            )
        else:
            abstention_msg = None
            reasoning = (
                f"Evidence sufficiency score ({sufficiency_score:.2f}) meets or exceeds threshold ({tau:.2f}). "
                f"Matched aspects: {', '.join(matched_aspects)}."
            )

        return SufficiencyAssessment(
            is_sufficient=is_sufficient,
            sufficiency_score=sufficiency_score,
            threshold=tau,
            topic=topic,
            abstention_message=abstention_msg,
            reasoning=reasoning,
            matched_aspects=matched_aspects,
            missing_aspects=missing_aspects,
        )


default_sufficiency_classifier = EvidenceSufficiencyClassifier()

"""Conversational query rewriting and search query distillation."""

from __future__ import annotations

import re

from src.query_processing.models import ConversationTurn


class QueryRewriter:
    """Rewrites queries by resolving conversational coreferences and ellipsis,

    and distilling raw queries into focused search queries.
    """

    CONVERSATIONAL_PREFIXES = [
        r"^(?:can you|could you|please)\s+(?:tell me|explain|describe|show me)\s+(?:about\s+)?",
        r"^(?:what (?:is|are|was|were)|tell me about|how does|how do|explain to me)\s+(?:our\s+|the\s+)?",
        r"^(?:what did (?:the\s+)?)(?:.*?)(?:say about\s+)",
        r"^(?:i would like to know|i want to know)\s+(?:about\s+)?",
        r"^(?:do we have\s+(?:a\s+)?(?:policy on|rules for)\s+)",
    ]

    PRONOUNS = {
        r"\bit\b",
        r"\bits\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bthose\b",
        r"\bthey\b",
        r"\bthem\b",
    }

    def __init__(self) -> None:
        self._compiled_prefixes = [
            re.compile(p, re.IGNORECASE) for p in self.CONVERSATIONAL_PREFIXES
        ]

    def extract_main_subject(self, text: str) -> str | None:
        """Extract dominant topic or subject phrase from a conversational turn."""
        cleaned = text.strip()
        for pat in self._compiled_prefixes:
            cleaned = pat.sub("", cleaned)

        # Strip trailing punctuation
        cleaned = re.sub(r"[?!.,;:]+$", "", cleaned).strip()

        # Check for common policy / topic nouns
        match = re.search(
            r"([a-z0-9_\-\s]+(?:policy|handbook|guidelines|process|rules|leave|work|deployment))",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        words = cleaned.split()
        return " ".join(words[:4]) if words else None

    def rewrite(self, query: str, history: list[ConversationTurn] | None = None) -> str:
        """Enrich a conversational query by expanding ellipsis and pronoun references."""
        raw_query = query.strip()
        if not history:
            return raw_query

        # Find last user question and last assistant response
        last_user_turn = next(
            (t.content for t in reversed(history) if t.role == "user"), None
        )
        last_asst_turn = next(
            (t.content for t in reversed(history) if t.role == "assistant"), None
        )

        subject = None
        if last_user_turn:
            subject = self.extract_main_subject(last_user_turn)

        # 1. Ellipsis pattern: "What about <X>?" / "How about <X>?" / "And <X>?"
        ellipsis_match = re.match(
            r"^(?:what|how)\s+about\s+(.+?)[?!.,;:]*$", raw_query, re.IGNORECASE
        )
        if not ellipsis_match:
            ellipsis_match = re.match(
                r"^and\s+(?:what\s+about\s+)?(.+?)[?!.,;:]*$", raw_query, re.IGNORECASE
            )

        if ellipsis_match:
            target = ellipsis_match.group(1).strip()
            # If target is e.g. "sick leave" and subject is "leave policy"
            if subject and "leave" in subject.lower() and "leave" in target.lower():
                # Formulate "What is the employee sick leave policy?"
                prefix = (
                    "employee "
                    if "employee" in (last_user_turn or "").lower()
                    or (last_asst_turn and "employee" in last_asst_turn.lower())
                    else ""
                )
                return f"What is the {prefix}{target} policy?"
            if subject:
                # Merge target with context subject
                return f"What is the {target} regarding {subject}?"
            return f"What is the policy for {target}?"

        # 2. Pronoun reference resolution: "How many days does it allow?"
        has_pronoun = any(re.search(p, raw_query, re.IGNORECASE) for p in self.PRONOUNS)
        if has_pronoun and subject:
            # Replace pronoun with subject
            resolved = raw_query
            for p in self.PRONOUNS:
                resolved = re.sub(p, f"the {subject}", resolved, flags=re.IGNORECASE)
            return resolved

        return raw_query

    def distill_search_query(self, query: str) -> str:
        """Strip conversational scaffolding to yield concise retrieval tokens."""
        q = query.strip()

        # Handle pattern: "What did the [YEAR] [DEPT] policy say about [TOPIC]?"
        # -> "[DEPT] [TOPIC] policy"
        pattern_did_say = re.match(
            r"^what\s+did\s+(?:the\s+)?(?:\d{4}\s+)?([a-z0-9_\-\s]+?)\s+(?:policy|handbook|guide)\s+say\s+about\s+(.+?)[?!.,;:]*$",
            q,
            re.IGNORECASE,
        )
        if pattern_did_say:
            dept = pattern_did_say.group(1).strip()
            topic = pattern_did_say.group(2).strip()
            return f"{dept} {topic} policy"

        # General prefix stripping
        for pat in self._compiled_prefixes:
            q = pat.sub("", q)

        # Remove polite trailing phrases
        q = re.sub(r"(?:please\??|thanks\??|thank you\??)$", "", q, flags=re.IGNORECASE)
        q = re.sub(r"[?!.,;:]+$", "", q).strip()

        # Normalize redundant spaces
        tokens = [t for t in q.split() if t]
        return " ".join(tokens)

    def distill(self, query: str) -> str:
        """Alias for distill_search_query."""
        return self.distill_search_query(query)

    def generate_hyde_passage(self, query: str, intent: str | None = None) -> str:
        """Synthesize a hypothetical declarative document passage (HyDE).

        Converts questions into declarative document assertions that align
        closely with the semantic vector space of index documents.
        """
        distilled = self.distill_search_query(query)
        if not distilled:
            return query

        # Formulate affirmative declarative context
        if intent and intent.lower() in ("factual", "informational"):
            return (
                f"The official documentation states regarding {distilled} that all guidelines, "
                f"requirements, standards, and procedures are defined and maintained for compliance."
            )
        if intent and intent.lower() == "comparison":
            return (
                f"A detailed comparison regarding {distilled} outlines the key differences, "
                f"tradeoffs, performance metrics, and recommended usage patterns across both options."
            )
        return (
            f"Regarding {distilled}, the document provides specific instructions, rules, "
            f"operational parameters, and essential reference details."
        )

    def generate_query_expansions(self, query: str) -> list[str]:
        """Generate alternative keyword and semantic query formulations to boost recall."""
        distilled = self.distill_search_query(query)
        expansions = [query]
        if distilled and distilled != query:
            expansions.append(distilled)

        # Common domain synonym expansions
        synonyms = {
            "vacation": "annual leave paid time off",
            "leave": "time off absence policy",
            "bonus": "compensation incentive payment",
            "salary": "compensation wage rate",
            "authentication": "auth bearer token credentials",
            "deploy": "deployment release pipeline",
        }
        for word, syn in synonyms.items():
            if re.search(rf"\b{word}\b", distilled, re.IGNORECASE):
                expanded = re.sub(
                    rf"\b{word}\b", f"{word} {syn}", distilled, flags=re.IGNORECASE
                )
                expansions.append(expanded)
                break

        return expansions

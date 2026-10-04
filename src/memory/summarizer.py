"""Hierarchical Persistent Summarizer condensing completed sessions into structured memory."""

from __future__ import annotations

import re
import uuid

from src.memory.models import EpisodicMemoryItem, HierarchicalSummary, Session


class HierarchicalSummarizer:
    """Condenses multi-turn sessions into dense, fact-preserving hierarchical summaries.

    Solves context degradation and token bloat by ensuring that older conversational turns
    do not have to be injected verbatim into future prompts. Instead, extracted entities,
    decisions, and verified factual conclusions are organized hierarchically.
    """

    def __init__(self, turns_per_summary: int = 5) -> None:
        self.turns_per_summary = turns_per_summary

    def extract_key_entities(self, memories: list[EpisodicMemoryItem]) -> list[str]:
        """Extract focal entities, document references, and domain concepts from memories."""
        entities: set[str] = set()

        for m in memories:
            # Include explicit referenced document IDs
            for doc in m.referenced_doc_ids:
                if doc:
                    entities.add(doc)

            # Heuristic entity extraction: Capitalized multi-word phrases and tags
            for tag in m.tags:
                if tag:
                    entities.add(tag)

            # Regex for potential capitalized entities / proper nouns
            text = f"{m.query} {m.answer}"
            matches = re.findall(r"\b[A-Z][a-zA-Z0-9_\-\.]{2,}(?:\s+[A-Z][a-zA-Z0-9_\-\.]{2,})*\b", text)
            for match in matches:
                # Filter out standard sentence starters
                if match not in {"What", "How", "Why", "When", "Where", "Which", "Can", "Could", "Should"}:
                    entities.add(match)

        # Return sorted list of unique entities, capped at 15
        return sorted(entities)[:15]

    def summarize_session(
        self,
        session: Session,
        memories: list[EpisodicMemoryItem],
        level: int = 1,
    ) -> HierarchicalSummary:
        """Condense a session's episodic memories into a persistent HierarchicalSummary."""
        if not memories:
            return HierarchicalSummary(
                summary_id=f"sum_{uuid.uuid4().hex[:12]}",
                session_id=session.session_id,
                level=level,
                title=f"Summary for {session.title}",
                summary_text="No interactions recorded in this session.",
                key_entities=[],
                turn_count=0,
            )

        key_entities = self.extract_key_entities(memories)

        # Collect distinct verified facts across turns
        all_verified_facts: list[str] = []
        for m in memories:
            for fact in m.verified_facts:
                if fact and fact not in all_verified_facts:
                    all_verified_facts.append(fact)

        # Build bulleted interaction summary
        turn_points = []
        for m in memories[-self.turns_per_summary:]:
            q_clean = m.query.strip().rstrip("?")
            ans_snippet = m.answer.strip().split(".")[0] if m.answer else "Concluded."
            turn_points.append(f"- User queried '{q_clean}': Established that {ans_snippet}.")

        summary_body = (
            f"Session '{session.title}' contains {len(memories)} interactions across {len(key_entities)} key entities.\n"
            + "\n".join(turn_points)
        )

        if all_verified_facts:
            facts_list = "; ".join(all_verified_facts[:5])
            summary_body += f"\nVerified core facts: {facts_list}."

        title = f"Session Brief: {session.title} ({len(memories)} turns)"

        return HierarchicalSummary(
            summary_id=f"sum_{uuid.uuid4().hex[:12]}",
            session_id=session.session_id,
            level=level,
            title=title,
            summary_text=summary_body.strip(),
            key_entities=key_entities,
            turn_count=len(memories),
        )


default_hierarchical_summarizer = HierarchicalSummarizer()

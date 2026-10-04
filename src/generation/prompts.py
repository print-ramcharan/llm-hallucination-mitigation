"""Grounded Prompt Synthesizer for factual LLM generation with mandatory citations."""

from __future__ import annotations

from src.context.models import CompactedEvidence, OptimizedContext
from src.query_processing.models import ConversationTurn


class GroundedPromptSynthesizer:
    """Constructs prompt templates enforcing strict evidence grounding and bracketed citations.

    Guarantees:
      1. Zero external speculation: restricts the LLM to only facts present in context passages.
      2. Mandatory bracketed citations: every claim must explicitly reference [Doc X, Chunk Y].
      3. Attribution preservation: links claims directly to source chunk tags for downstream verification.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an evidence-grounded AI question answering assistant.\n"
        "Your task is to answer the user's question accurately and objectively using ONLY the facts "
        "provided in the EVIDENCE CONTEXT below.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Strictly Factual & Grounded: Do NOT fabricate, extrapolate, or assume any information. "
        "Every single factual assertion you make must be directly backed by the provided evidence.\n"
        "2. Mandatory Bracketed Citations: Every statement, number, policy, or claim MUST be immediately "
        "followed by its source citation tag in brackets, exactly as given in the context headers "
        "(for example: 'Employees receive 20 days of paid vacation leave [Doc 1, Chunk 0].').\n"
        "3. Combined Citations: If multiple passages support a claim, combine their citations "
        "(for example: '[Doc 1, Chunk 0][Doc 2, Chunk 1]').\n"
        "4. No External Knowledge: If the context contains partial information, state what is known "
        "with citations, and state clearly what cannot be determined from the evidence.\n"
        "5. Concise & Objective: Keep your response direct, structured, and free of conversational filler."
    )

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def build_evidence_block(
        self,
        context: OptimizedContext | None = None,
        evidence_items: list[CompactedEvidence] | None = None,
    ) -> str:
        """Render context passages with clear citation identifier headers."""
        if context is not None and context.formatted_prompt_context:
            return context.formatted_prompt_context

        items = []
        if context is not None and context.evidence_items:
            items = context.evidence_items
        elif evidence_items:
            items = evidence_items

        if not items:
            return "NO EVIDENCE PROVIDED."

        blocks = []
        for item in items:
            tag = item.citation_tag
            text = item.extracted_text or item.original_text
            source_meta = item.metadata.get("source", item.document_id)
            blocks.append(f"--- EVIDENCE {tag} (Source: {source_meta}) ---\n{text.strip()}")

        return "\n\n".join(blocks)

    def format_conversation_history(
        self, conversation_history: list[ConversationTurn] | None = None
    ) -> str:
        """Format prior conversational turns for conversational context injection."""
        if not conversation_history:
            return ""

        lines = ["CONVERSATION HISTORY:"]
        for turn in conversation_history:
            role_label = "User" if turn.role.lower() == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content.strip()}")

        return "\n".join(lines)

    def synthesize_prompt(
        self,
        query: str,
        context: OptimizedContext | None = None,
        evidence_items: list[CompactedEvidence] | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        memory_context: str | None = None,
    ) -> str:
        """Synthesize the complete grounded user prompt."""
        evidence_block = self.build_evidence_block(context=context, evidence_items=evidence_items)
        history_block = self.format_conversation_history(conversation_history)

        parts = []
        if memory_context and memory_context.strip():
            parts.append("RELEVANT PAST MEMORY & USER CONTEXT:\n" + memory_context.strip())

        if history_block:
            parts.append(history_block)

        parts.append("EVIDENCE CONTEXT:\n" + evidence_block)
        parts.append(
            f"USER QUERY:\n{query.strip()}\n\n"
            "GROUNDED ANSWER (Include bracketed citations like [Doc X, Chunk Y] after each factual claim):"
        )

        return "\n\n".join(parts)


default_prompt_synthesizer = GroundedPromptSynthesizer()

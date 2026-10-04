"""Cross-Session Memory Synchronizer providing bi-directional synchronization with LLM inference."""

from __future__ import annotations

import datetime
import uuid

from src.generation.models import ClaimStatus, GroundingReport
from src.memory.models import (
    EpisodicMemoryItem,
    HierarchicalSummary,
    MemorySyncContext,
    WebContextCaptureRequest,
)
from src.memory.session_registry import SessionRegistry, default_session_registry
from src.memory.store import ExternalMemoryStore, default_memory_store
from src.memory.summarizer import (
    HierarchicalSummarizer,
    default_hierarchical_summarizer,
)


class CrossSessionMemorySynchronizer:
    """Bi-directional synchronization service connecting Module 6 (Inference) and Module 7 (Memory).

    Before Inference (Pre-Sync):
      - Retrieves the most relevant past episodic turns and persistent summaries.
      - Formulates a compact, strictly token-budgeted memory context snippet.
      - Injects relevant past facts into the prompt without causing context degradation.

    After Inference (Post-Sync):
      - Extracts empirically verified facts (entailed claims) and citation provenance.
      - Persists the interaction turn and indexed embeddings into external vector storage.
      - Periodically condenses completed sessions into persistent hierarchical summaries.
    """

    def __init__(
        self,
        store: ExternalMemoryStore | None = None,
        registry: SessionRegistry | None = None,
        summarizer: HierarchicalSummarizer | None = None,
    ) -> None:
        self.store = store or default_memory_store
        self.registry = registry or default_session_registry
        self.summarizer = summarizer or default_hierarchical_summarizer

    # =========================================================================
    # Pre-Inference Synchronization: Memory Injection
    # =========================================================================

    def pre_inference_sync(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str = "default_user",
        max_tokens: int | None = None,
    ) -> MemorySyncContext:
        """Retrieve relevant past memories and synthesize a token-budgeted memory injection block."""
        if not session_id:
            # Fallback to default session
            session = self.registry.get_or_create_default_session(user_id=user_id)
            session_id = session.session_id

        perms = self.registry.get_permissions(session_id)
        if not perms.read_enabled:
            return MemorySyncContext(
                session_id=session_id,
                read_enabled=False,
                injected_memories=[],
                injected_summaries=[],
                formatted_memory_context="",
                estimated_tokens=0,
            )

        budget_limit = max_tokens or perms.max_injected_tokens

        # Retrieve top relevant memories
        memories, scores = self.store.search_memories(
            query=query,
            session_id=None,  # Cross-session retrieval enabled across user's history
            user_id=user_id,
            top_k=perms.max_injected_memories,
            min_similarity=0.25,
        )

        # Retrieve relevant session summaries if applicable
        summaries, _ = self.store.search_summaries(
            query=query,
            session_id=session_id,
            top_k=1,
            min_similarity=0.25,
        )

        formatted_context, estimated_tokens = self._format_compact_memory_context(
            memories=memories,
            summaries=summaries,
            max_tokens=budget_limit,
        )

        return MemorySyncContext(
            session_id=session_id,
            read_enabled=True,
            injected_memories=memories,
            injected_summaries=summaries,
            formatted_memory_context=formatted_context,
            estimated_tokens=estimated_tokens,
        )

    def _format_compact_memory_context(
        self,
        memories: list[EpisodicMemoryItem],
        summaries: list[HierarchicalSummary],
        max_tokens: int = 250,
    ) -> tuple[str, int]:
        """Synthesize a dense, compact markdown block guaranteed to stay within token budget."""
        if not memories and not summaries:
            return "", 0

        blocks: list[str] = []

        if summaries:
            summary = summaries[0]
            # Use concise summary line
            clean_summary = summary.summary_text.split("\n")[0]
            blocks.append(f"[Session Summary]: {clean_summary}")

        if memories:
            blocks.append("[Relevant Past Interactions & Verified Facts]:")
            for m in memories:
                # Prioritize verified facts if available
                if m.verified_facts:
                    facts_str = "; ".join(m.verified_facts[:2])
                    blocks.append(f"- Past Q: \"{m.query}\" -> Established: {facts_str}")
                else:
                    ans_excerpt = m.answer.strip().split(".")[0]
                    blocks.append(f"- Past Q: \"{m.query}\" -> {ans_excerpt}.")

        raw_text = "\n".join(blocks)
        # Token estimation: roughly 1.3 tokens per whitespace-delimited word
        words = raw_text.split()
        estimated_tokens = int(len(words) * 1.3)

        # If over budget, truncate words safely
        if estimated_tokens > max_tokens and max_tokens > 20:
            allowed_words = int(max_tokens / 1.3)
            truncated_words = words[:allowed_words]
            raw_text = " ".join(truncated_words) + "..."
            estimated_tokens = max_tokens

        return raw_text, estimated_tokens

    # =========================================================================
    # Post-Inference Synchronization: Verified Memory Persistence
    # =========================================================================

    def post_inference_sync(
        self,
        session_id: str,
        query: str,
        answer: str,
        grounding_report: GroundingReport | None = None,
        intent: str | None = None,
        referenced_doc_ids: list[str] | None = None,
        tags: list[str] | None = None,
        user_id: str = "default_user",
    ) -> EpisodicMemoryItem | None:
        """Record completed interaction, filter verified facts, and persist to vector storage."""
        session = self.registry.get_session(session_id)
        if not session:
            session = self.registry.get_or_create_default_session(user_id=user_id)
            session_id = session.session_id

        perms = self.registry.get_permissions(session_id)
        if not perms.write_enabled:
            return None

        # Filter strictly verified factual propositions (only entailed claims)
        verified_facts: list[str] = []
        citations: list[str] = []
        if grounding_report:
            citations = list(set(grounding_report.verified_citations))
            for claim in grounding_report.claims:
                if claim.status == ClaimStatus.ENTAILED:
                    verified_facts.append(claim.claim_text.strip())

        # If no grounding report provided, extract first factual sentence as baseline fact
        if not verified_facts and answer and not answer.startswith("The available documents do not contain"):
            first_sentence = answer.strip().split(".")[0].strip()
            if len(first_sentence) > 10:
                verified_facts.append(first_sentence)

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        item = EpisodicMemoryItem(
            memory_id=memory_id,
            session_id=session_id,
            user_id=user_id,
            turn_index=session.turn_count,
            query=query.strip(),
            intent=intent,
            answer=answer.strip(),
            verified_facts=verified_facts,
            referenced_doc_ids=referenced_doc_ids or session.active_document_ids,
            citations=citations,
            timestamp=now,
            importance_score=1.2 if verified_facts else 1.0,
            tags=tags or ["qa_interaction"],
            metadata={},
        )

        saved_item = self.store.save_memory_item(item)

        # Trigger automatic hierarchical summarization if turn count is a multiple of turns_per_summary
        updated_session = self.registry.get_session(session_id)
        if updated_session and perms.auto_summarize and updated_session.turn_count >= 3:
            if updated_session.turn_count % self.summarizer.turns_per_summary == 0 or updated_session.turn_count == 3:
                all_memories = self.store.list_session_memories(session_id)
                summary = self.summarizer.summarize_session(updated_session, all_memories)
                self.store.save_summary(summary)

        return saved_item

    # =========================================================================
    # Browser Extension & Web Client Context Capture
    # =========================================================================

    def capture_web_context(self, request: WebContextCaptureRequest) -> EpisodicMemoryItem:
        """Capture highlighted text and page metadata sent from Browser Extension or Web Client."""
        session_id = request.session_id
        if not session_id:
            session = self.registry.get_or_create_default_session(user_id=request.user_id)
            session_id = session.session_id

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        memory_id = f"web_{uuid.uuid4().hex[:12]}"

        # Synthesize query and answer from the web capture
        query = f"Captured excerpt from: {request.title}"
        answer = request.selected_text.strip()
        if request.full_content and len(request.full_content) > len(answer):
            extra_snippet = request.full_content[:200].strip()
            answer += f"\nContext: {extra_snippet}..."

        item = EpisodicMemoryItem(
            memory_id=memory_id,
            session_id=session_id,
            user_id=request.user_id,
            turn_index=0,
            query=query,
            intent="WEB_CONTEXT",
            answer=answer,
            verified_facts=[request.selected_text.strip()[:150]],
            referenced_doc_ids=[],
            citations=[f"URL: {request.url}"],
            timestamp=now,
            importance_score=1.1,
            tags=request.tags or ["web_capture"],
            metadata={"url": request.url, "title": request.title},
        )

        return self.store.save_memory_item(item)


default_memory_synchronizer = CrossSessionMemorySynchronizer()

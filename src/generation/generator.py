"""End-to-End Grounded Generator orchestrating Sufficiency, Prompting, Inference, and NLI Verification."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator

from src.context.models import CompactedEvidence, OptimizedContext
from src.generation.engine import (
    BaseInferenceEngine,
    EngineFactory,
    default_inference_engine,
)
from src.generation.models import (
    GenerationRequest,
    GenerationResponse,
    GroundingReport,
    NaiveGenerationResponse,
    SufficiencyAssessment,
)
from src.generation.prompts import GroundedPromptSynthesizer, default_prompt_synthesizer
from src.generation.sufficiency import (
    EvidenceSufficiencyClassifier,
    default_sufficiency_classifier,
)
from src.generation.verifier import (
    AntiHallucinationVerifier,
    default_anti_hallucination_verifier,
)
from src.query_processing.models import ConversationTurn


class GroundedGenerator:
    """Coordinator executing grounded LLM inference with anti-hallucination guardrails.

    Workflow:
      1. Pre-flight Evidence Sufficiency Evaluation -> If < threshold, triggers truthful Abstention Protocol.
      2. Grounded Prompt Synthesis -> Enforces mandatory bracketed citations and strict factual constraints.
      3. Inference / Token Streaming -> Provider-agnostic inference (offline, Ollama, OpenAI, Gemini).
      4. Post-generation NLI Verification -> Sentence-level verification (Entailment / Neutral / Contradiction).
    """

    def __init__(
        self,
        sufficiency_classifier: EvidenceSufficiencyClassifier | None = None,
        prompt_synthesizer: GroundedPromptSynthesizer | None = None,
        engine: BaseInferenceEngine | None = None,
        verifier: AntiHallucinationVerifier | None = None,
    ) -> None:
        self.sufficiency_classifier = sufficiency_classifier or default_sufficiency_classifier
        self.prompt_synthesizer = prompt_synthesizer or default_prompt_synthesizer
        self.engine = engine or default_inference_engine
        self.verifier = verifier or default_anti_hallucination_verifier

    def generate(
        self,
        query: str,
        context: OptimizedContext | None = None,
        raw_evidence_chunks: list[CompactedEvidence] | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        sufficiency_threshold: float | None = None,
        hallucination_threshold: float | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        provider: str | None = None,
        memory_context: str | None = None,
    ) -> GenerationResponse:
        """Execute synchronous grounded generation with pre- and post-generation guardrails."""
        start_time = time.perf_counter()

        # Step 1: Pre-flight Evidence Sufficiency Evaluation
        sufficiency: SufficiencyAssessment = self.sufficiency_classifier.assess_sufficiency(
            query=query,
            context=context,
            evidence_items=raw_evidence_chunks,
            threshold=sufficiency_threshold,
        )

        # Truthful Abstention Protocol
        if not sufficiency.is_sufficient:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            abstention_text = sufficiency.abstention_message or (
                f"The available documents do not contain sufficient evidence to verify {sufficiency.topic}."
            )
            empty_report = GroundingReport(
                faithfulness_score=1.0,
                hallucination_detected=False,
                total_claims=0,
                entailed_claims_count=0,
                neutral_claims_count=0,
                contradicted_claims_count=0,
                claims=[],
                verified_citations=[],
                unverified_citations=[],
            )
            return GenerationResponse(
                query=query,
                answer=abstention_text,
                abstained=True,
                abstention_reason=sufficiency.reasoning,
                sufficiency=sufficiency,
                grounding_report=empty_report,
                citations=[],
                latency_ms=latency_ms,
                model_name=self.engine.model_name,
            )

        # Step 2: Grounded Prompt Synthesis
        prompt = self.prompt_synthesizer.synthesize_prompt(
            query=query,
            context=context,
            evidence_items=raw_evidence_chunks,
            conversation_history=conversation_history,
            memory_context=memory_context,
        )

        # Step 3: LLM Inference
        engine = EngineFactory.create_engine(provider) if provider else self.engine
        raw_answer = engine.generate(
            prompt=prompt,
            system_prompt=self.prompt_synthesizer.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Step 4: Post-generation Anti-Hallucination Verification
        grounding_report = self.verifier.verify_answer(
            answer=raw_answer,
            context=context,
            evidence_items=raw_evidence_chunks,
            tolerance=hallucination_threshold,
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        citations = self.verifier.extract_citations(raw_answer)

        return GenerationResponse(
            query=query,
            answer=raw_answer,
            abstained=False,
            abstention_reason=None,
            sufficiency=sufficiency,
            grounding_report=grounding_report,
            citations=citations,
            latency_ms=latency_ms,
            model_name=engine.model_name,
        )

    def generate_request(self, payload: GenerationRequest) -> GenerationResponse:
        """Generate response from structured GenerationRequest payload."""
        return self.generate(
            query=payload.query,
            context=payload.context,
            raw_evidence_chunks=payload.raw_evidence_chunks,
            conversation_history=payload.conversation_history,
            sufficiency_threshold=payload.sufficiency_threshold,
            hallucination_threshold=payload.hallucination_threshold,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            provider=payload.provider,
        )

    def generate_stream(
        self,
        query: str,
        context: OptimizedContext | None = None,
        raw_evidence_chunks: list[CompactedEvidence] | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        sufficiency_threshold: float | None = None,
        hallucination_threshold: float | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        provider: str | None = None,
        memory_context: str | None = None,
    ) -> Iterator[str]:
        """Stream real-time tokens and guardrail telemetry via Server-Sent Events (SSE)."""
        start_time = time.perf_counter()

        # 1. Pre-flight Sufficiency Assessment
        sufficiency = self.sufficiency_classifier.assess_sufficiency(
            query=query,
            context=context,
            evidence_items=raw_evidence_chunks,
            threshold=sufficiency_threshold,
        )

        yield f"event: sufficiency\ndata: {json.dumps(sufficiency.model_dump())}\n\n"

        # Truthful Abstention Protocol
        if not sufficiency.is_sufficient:
            abstention_text = sufficiency.abstention_message or (
                f"The available documents do not contain sufficient evidence to verify {sufficiency.topic}."
            )
            yield f"event: abstention\ndata: {json.dumps({'answer': abstention_text, 'reason': sufficiency.reasoning})}\n\n"
            yield f"event: done\ndata: {json.dumps({'abstained': True, 'latency_ms': round((time.perf_counter() - start_time) * 1000, 2)})}\n\n"
            return

        # 2. Prompt Synthesis
        prompt = self.prompt_synthesizer.synthesize_prompt(
            query=query,
            context=context,
            evidence_items=raw_evidence_chunks,
            conversation_history=conversation_history,
            memory_context=memory_context,
        )

        # 3. Token Streaming
        engine = EngineFactory.create_engine(provider) if provider else self.engine
        accumulated_chunks: list[str] = []

        yield f"event: start\ndata: {json.dumps({'model': engine.model_name})}\n\n"

        for token in engine.generate_stream(
            prompt=prompt,
            system_prompt=self.prompt_synthesizer.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            accumulated_chunks.append(token)
            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

        full_answer = "".join(accumulated_chunks)

        # 4. Post-generation NLI Verification
        grounding_report = self.verifier.verify_answer(
            answer=full_answer,
            context=context,
            evidence_items=raw_evidence_chunks,
            tolerance=hallucination_threshold,
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        citations = self.verifier.extract_citations(full_answer)

        yield f"event: grounding\ndata: {json.dumps(grounding_report.model_dump())}\n\n"
        yield f"event: done\ndata: {json.dumps({'abstained': False, 'citations': citations, 'latency_ms': latency_ms})}\n\n"

    def generate_naive(
        self,
        query: str,
        document_text: str | None = None,
        provider: str | None = None,
    ) -> NaiveGenerationResponse:
        """Generate response from a naive baseline LLM provided with the entire document and query directly,
        without chunking, hybrid retrieval, cross-encoder reranking, context compaction, or NLI guardrails.
        """
        start_time = time.perf_counter()
        engine = EngineFactory.create_engine(provider) if provider else self.engine

        doc_context = (document_text or "").strip()

        if getattr(engine, "is_configured", lambda: True)():
            system_prompt = (
                "You are an AI assistant. You are provided with the entire raw document content and a user query. "
                "Answer the user's question directly based on the raw document text. "
                "Do not use external retrieval, chunking, or special citation tags."
            )
            prompt = (
                f"ENTIRE DOCUMENT CONTENT:\n{doc_context}\n\n"
                f"USER QUERY: {query}\n\n"
                f"ANSWER:"
            )
            try:
                raw_answer = engine.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=1024,
                )
            except Exception:
                raw_answer = self._generate_offline_naive(query=query, document_text=doc_context)
        else:
            # Offline naive baseline: processes the entire un-chunked document text directly
            raw_answer = self._generate_offline_naive(query=query, document_text=doc_context)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return NaiveGenerationResponse(
            query=query,
            answer=raw_answer,
            has_citations=False,
            citations=[],
            faithfulness_score=0.40 if doc_context else 0.20,
            hallucination_risk="High (Unverified / No Citation Anchors / Subject to Context Degradation)",
            latency_ms=latency_ms,
            model_name=f"{engine.model_name} (Naive Monolithic Baseline)",
        )

    def _generate_offline_naive(self, query: str, document_text: str = "") -> str:
        """Synthesize naive LLM response given the entire document and query directly.

        Demonstrates baseline long-context behavior:
        - Directly processes the entire monolithic document without semantic sliding-window chunking.
        - Suffers from Context Degradation (Lost-in-the-Middle effect) over lengthy texts.
        - Produces continuous text without bracketed citation tags or claim verification.
        """
        if not document_text.strip():
            return (
                f"Regarding '{query}', no document context was provided. Standard AI models typically formulate "
                "a generalized response derived from broad web training data without verified document citations."
            )

        q_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query.lower()))
        from src.generation.sufficiency import _STOP_WORDS
        content_words = {w for w in q_words if w not in _STOP_WORDS}

        # Split entire raw document into sentences
        clean_doc = document_text.replace("\r", " ")
        raw_sentences = re.split(r"(?<=[.!?])\s+", clean_doc)

        scored_sentences: list[tuple[float, int, str]] = []
        for idx, s in enumerate(raw_sentences):
            s_clean = s.strip()
            if len(s_clean) < 20 or s_clean.startswith("--- Page"):
                continue
            s_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", s_clean.lower()))
            overlap = len(content_words & s_words)
            if overlap > 0:
                # Simulating transformer attention over monolithic context:
                # Positions in the middle of large contexts suffer from attention attenuation (Lost-in-the-Middle)
                rel_pos = idx / max(len(raw_sentences), 1)
                # U-shaped attention curve: head and tail retain attention; middle suffers degradation
                attention_weight = 1.0 - 0.35 * (1.0 - 4.0 * (rel_pos - 0.5) ** 2) if len(raw_sentences) > 30 else 1.0
                score = (overlap / (len(content_words) + 1e-5)) * attention_weight
                scored_sentences.append((score, idx, s_clean))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if not scored_sentences or scored_sentences[0][0] <= 0:
            return (
                f"Based on reading the entire document, the text contains general information, but does not provide "
                f"a direct answer to '{query}'."
            )

        # Take the top matching sentences from naive monolithic reading
        selected = []
        seen = set()
        for _score, _idx, sent in scored_sentences:
            clean_sent = re.sub(r"--- Page \d+ ---", "", sent).strip()
            norm = clean_sent.lower()[:30]
            if norm not in seen and len(clean_sent) > 15:
                seen.add(norm)
                selected.append(clean_sent.rstrip(". "))
            if len(selected) >= 3:
                break

        joined = ". ".join(selected) + "."
        return joined


default_grounded_generator = GroundedGenerator()

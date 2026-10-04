"""FastAPI REST routes for Module 6: Grounded LLM Inference & Anti-Hallucination Guardrails."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.context.compactor import default_context_compactor
from src.generation.generator import default_grounded_generator
from src.generation.models import (
    ComparisonResponse,
    GenerationRequest,
    GenerationResponse,
)
from src.query_processing.models import ConversationTurn
from src.reranking.reranker import default_reranker
from src.retrieval.hybrid import default_hybrid_retriever

router = APIRouter(prefix="/api/generate", tags=["Grounded Generation & Guardrails"])


class FullQAPipelineRequest(BaseModel):
    """Payload executing the complete end-to-end question answering pipeline:

    Query -> Hybrid Retrieval -> Cross-Encoder Rerank -> Context Compaction -> Grounded Generation / Abstention.
    """

    query: str = Field(..., description="User search query string.")
    conversation_history: list[ConversationTurn] | None = Field(
        default=None,
        description="Optional prior interactive conversational turns.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured metadata filters.",
    )
    top_k_fused: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Candidate pool size to retrieve from Module 3 hybrid retrieval.",
    )
    rerank_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Cutoff threshold for Module 4 cross-encoder reranking.",
    )
    rerank_top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum candidates retained after Module 4 reranking.",
    )
    max_token_budget: int = Field(
        default=1500,
        ge=100,
        le=8000,
        description="Target maximum token allowance for compacted prompt context.",
    )
    sufficiency_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum sufficiency required before triggering truthful abstention.",
    )
    hallucination_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum faithfulness score required before flagging hallucination warnings.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM generation.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Maximum generation token allowance.",
    )
    provider: str | None = Field(
        default=None,
        description="Optional LLM provider override ('offline', 'openai', 'gemini', 'ollama').",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID to enable cross-session external memory sync (Module 7).",
    )


@router.post(
    "/answer",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate grounded answer or trigger truthful abstention with anti-hallucination verification",
)
def generate_grounded_answer(payload: GenerationRequest) -> GenerationResponse:
    """Evaluate context sufficiency, synthesize grounded prompt, infer answer with citations,

    and verify claim faithfulness using sentence-level NLI.
    """
    return default_grounded_generator.generate_request(payload)


@router.post(
    "/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream real-time tokens and guardrail telemetry via Server-Sent Events (SSE)",
)
def stream_grounded_answer(payload: GenerationRequest) -> StreamingResponse:
    """Stream token-by-token output with sufficiency assessment and final NLI grounding report."""
    generator = default_grounded_generator.generate_stream(
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
    return StreamingResponse(generator, media_type="text/event-stream")


@router.post(
    "/pipeline/qa",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="End-to-End Pipeline: Retrieval (M3) -> Reranking (M4) -> Compaction (M5) -> Generation (M6) + Optional Memory (M7)",
)
def run_full_qa_pipeline(payload: FullQAPipelineRequest) -> GenerationResponse:
    """Execute end-to-end question answering pipeline:

    1. Pre-Inference Memory Sync (if session_id provided)
    2. Hybrid Retrieval (Dense FAISS + Sparse BM25 + RRF)
    3. Cross-Encoder Reranking & Pruning
    4. Extractive Context Compaction & Lost-in-the-Middle Reordering
    5. Evidence Sufficiency Assessment & Truthful Abstention Protocol
    6. Grounded LLM Generation & Sentence-Level NLI Verification
    7. Post-Inference Memory Sync (verified facts & persistent summaries)
    """
    # 0. Optional Module 7: Pre-Inference Memory Sync
    memory_context_str: str | None = None
    if payload.session_id:
        from src.memory.synchronizer import default_memory_synchronizer

        sync_ctx = default_memory_synchronizer.pre_inference_sync(
            query=payload.query,
            session_id=payload.session_id,
        )
        if sync_ctx.read_enabled and sync_ctx.formatted_memory_context:
            memory_context_str = sync_ctx.formatted_memory_context

    # 1. Module 3: Hybrid Retrieval
    retrieval_res = default_hybrid_retriever.retrieve(
        query=payload.query,
        conversation_history=payload.conversation_history,
        filters=payload.filters,
        top_k_fused=payload.top_k_fused,
    )

    # 2. Module 4: Cross-Encoder Reranking & Pruning
    ranked_context = default_reranker.rerank_retrieval_response(
        response=retrieval_res,
        threshold=payload.rerank_threshold,
        top_n=payload.rerank_top_n,
    )

    # 3. Module 5: Extractive Context Compaction & Reordering
    compacted_context = default_context_compactor.compact(
        query=payload.query,
        chunks=ranked_context.chunks,
        max_token_budget=payload.max_token_budget,
    )

    # 4 & 5. Module 6: Grounded Generation & Guardrails
    response = default_grounded_generator.generate(
        query=payload.query,
        context=compacted_context,
        conversation_history=payload.conversation_history,
        sufficiency_threshold=payload.sufficiency_threshold,
        hallucination_threshold=payload.hallucination_threshold,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        provider=payload.provider,
        memory_context=memory_context_str,
    )

    # 6. Optional Module 7: Post-Inference Memory Sync
    if payload.session_id:
        from src.memory.synchronizer import default_memory_synchronizer

        default_memory_synchronizer.post_inference_sync(
            session_id=payload.session_id,
            query=payload.query,
            answer=response.answer,
            grounding_report=response.grounding_report,
            referenced_doc_ids=[c.document_id for c in ranked_context.chunks],
        )

    return response


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Side-by-Side Comparison: Naive LLM vs Our Grounded Mitigation Pipeline",
)
def compare_naive_vs_grounded(payload: FullQAPipelineRequest) -> ComparisonResponse:
    """Execute both pipelines in parallel on the same query:

    1. Naive baseline LLM (direct ungrounded generation)
    2. Full Grounded Mitigation Pipeline (Retrieval + Rerank + Compaction + Gate + Citations + NLI Verification)
    """
    # 1. Run Grounded Pipeline (the complete multi-stage mitigation pipeline)
    grounded_res = run_full_qa_pipeline(payload)

    # 2. Extract entire document text for Naive LLM (monolithic prompt baseline)
    from src.ingestion.store import default_store

    entire_document_text = ""
    if payload.filters and "document_id" in payload.filters:
        doc = default_store.get(payload.filters["document_id"])
        if doc and doc.content:
            entire_document_text = doc.content

    if not entire_document_text:
        all_docs = default_store.list_all_documents()
        if all_docs:
            entire_document_text = "\n\n".join(d.content for d in all_docs if d.content)

    # 3. Run Naive Baseline with the ENTIRE document and the query directly
    naive_res = default_grounded_generator.generate_naive(
        query=payload.query,
        document_text=entire_document_text,
        provider=payload.provider,
    )

    # 3. Assemble Comparative Metrics
    metrics_comp = {
        "naive_citations_count": len(naive_res.citations),
        "grounded_citations_count": len(grounded_res.citations),
        "naive_faithfulness": "Unverified (0% anchored)",
        "grounded_faithfulness": f"{round(grounded_res.grounding_report.faithfulness_score * 100)}% verified",
        "naive_sufficiency_gate": "Disabled (Blind Generation)",
        "grounded_sufficiency_gate": (
            f"{'Sufficient' if grounded_res.sufficiency.is_sufficient else 'Abstained'} "
            f"({round(grounded_res.sufficiency.sufficiency_score * 100)}%)"
        ),
        "naive_nli_claims": "0 claims verified",
        "grounded_nli_claims": (
            f"{grounded_res.grounding_report.total_claims} verified "
            f"({grounded_res.grounding_report.entailed_claims_count} entailed, "
            f"{grounded_res.grounding_report.neutral_claims_count} neutral, "
            f"{grounded_res.grounding_report.contradicted_claims_count} contradicted)"
        ),
    }

    return ComparisonResponse(
        query=payload.query,
        naive=naive_res,
        grounded=grounded_res,
        metrics_comparison=metrics_comp,
    )


@router.post(
    "/compare/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream Real-Time Side-by-Side Comparison: Naive LLM vs Grounded Mitigation Pipeline",
)
def stream_compare_pipeline(payload: FullQAPipelineRequest) -> StreamingResponse:
    """Stream end-to-end question answering comparison with live token streaming and guardrail telemetry."""

    def event_stream():
        start_time = time.perf_counter()
        try:
            # Stage 1: Retrieval
            yield f"event: stage\ndata: {json.dumps({'stage': 'retrieval', 'message': 'Executing Hybrid Retrieval (FAISS Dense + BM25 Sparse + RRF)...'})}\n\n"
            retrieval_res = default_hybrid_retriever.retrieve(
                query=payload.query,
                conversation_history=payload.conversation_history,
                filters=payload.filters,
                top_k_fused=payload.top_k_fused,
            )

            # Stage 2: Cross-Encoder Rerank
            yield f"event: stage\ndata: {json.dumps({'stage': 'rerank', 'message': f'Cross-Encoder reranking {len(retrieval_res.candidates)} candidates and pruning unanchored noise...'})}\n\n"
            ranked_context = default_reranker.rerank_retrieval_response(
                response=retrieval_res,
                threshold=payload.rerank_threshold,
                top_n=payload.rerank_top_n,
            )

            # Stage 3: Extractive Compaction & Reordering
            yield f"event: stage\ndata: {json.dumps({'stage': 'compaction', 'message': 'Compacting context to budget and mitigating Lost-in-the-Middle...'})}\n\n"
            compacted_context = default_context_compactor.compact(
                query=payload.query,
                chunks=ranked_context.chunks,
                max_token_budget=payload.max_token_budget,
            )

            # Stage 4: Sufficiency Gate
            yield f"event: stage\ndata: {json.dumps({'stage': 'sufficiency', 'message': 'Evaluating evidence sufficiency gate...'})}\n\n"
            sufficiency = default_grounded_generator.sufficiency_classifier.assess_sufficiency(
                query=payload.query,
                context=compacted_context,
                threshold=payload.sufficiency_threshold,
            )
            yield f"event: sufficiency\ndata: {json.dumps(sufficiency.model_dump())}\n\n"

            # Memory context if applicable
            memory_context_str = None
            if payload.session_id:
                from src.memory.synchronizer import default_memory_synchronizer

                sync_ctx = default_memory_synchronizer.pre_inference_sync(
                    query=payload.query,
                    session_id=payload.session_id,
                )
                if sync_ctx.read_enabled and sync_ctx.formatted_memory_context:
                    memory_context_str = sync_ctx.formatted_memory_context

            # Stage 5: Grounded Inference Streaming
            from src.generation.engine import EngineFactory

            engine = EngineFactory.create_engine(payload.provider) if payload.provider else default_grounded_generator.engine

            yield f"event: stage\ndata: {json.dumps({'stage': 'generation', 'message': f'Streaming grounded response via {engine.model_name}...'})}\n\n"
            yield f"event: provider\ndata: {json.dumps({'model': engine.model_name, 'provider': getattr(engine, 'last_used_name', 'Auto-Fallback')})}\n\n"

            grounded_answer_chunks = []
            if not sufficiency.is_sufficient:
                abstention_text = sufficiency.abstention_message or (
                    f"The available documents do not contain sufficient evidence to verify {sufficiency.topic}."
                )
                grounded_answer_chunks.append(abstention_text)
                yield f"event: abstention\ndata: {json.dumps({'answer': abstention_text, 'reason': sufficiency.reasoning})}\n\n"
            else:
                prompt = default_grounded_generator.prompt_synthesizer.synthesize_prompt(
                    query=payload.query,
                    context=compacted_context,
                    conversation_history=payload.conversation_history,
                    memory_context=memory_context_str,
                )
                for token in engine.generate_stream(
                    prompt=prompt,
                    system_prompt=default_grounded_generator.prompt_synthesizer.system_prompt,
                    temperature=payload.temperature,
                    max_tokens=payload.max_tokens,
                ):
                    grounded_answer_chunks.append(token)
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            full_grounded_answer = "".join(grounded_answer_chunks)

            # Stage 6: Post-generation NLI Verification
            yield f"event: stage\ndata: {json.dumps({'stage': 'verification', 'message': 'Verifying grounded claims with sentence-level NLI...'})}\n\n"
            grounding_report = default_grounded_generator.verifier.verify_answer(
                answer=full_grounded_answer,
                context=compacted_context,
                tolerance=payload.hallucination_threshold,
            )
            citations = default_grounded_generator.verifier.extract_citations(full_grounded_answer)
            yield f"event: grounding\ndata: {json.dumps(grounding_report.model_dump())}\n\n"

            # Stage 7: Naive Baseline LLM Generation
            yield f"event: stage\ndata: {json.dumps({'stage': 'naive', 'message': 'Evaluating Naive LLM monolithic baseline...'})}\n\n"
            from src.ingestion.store import default_store

            entire_document_text = ""
            if payload.filters and "document_id" in payload.filters:
                doc = default_store.get(payload.filters["document_id"])
                if doc and doc.content:
                    entire_document_text = doc.content
            if not entire_document_text:
                all_docs = default_store.list_all_documents()
                if all_docs:
                    entire_document_text = "\n\n".join(d.content for d in all_docs if d.content)

            naive_res = default_grounded_generator.generate_naive(
                query=payload.query,
                document_text=entire_document_text,
                provider=payload.provider,
            )
            yield f"event: naive\ndata: {json.dumps(naive_res.model_dump())}\n\n"

            # Assemble Metrics Comparison
            metrics_comp = {
                "naive_citations_count": len(naive_res.citations),
                "grounded_citations_count": len(citations),
                "naive_faithfulness": "Unverified (0% anchored)",
                "grounded_faithfulness": f"{round(grounding_report.faithfulness_score * 100)}% verified",
                "naive_sufficiency_gate": "Disabled (Blind Generation)",
                "grounded_sufficiency_gate": (
                    f"{'Sufficient' if sufficiency.is_sufficient else 'Abstained'} "
                    f"({round(sufficiency.sufficiency_score * 100)}%)"
                ),
                "naive_nli_claims": "0 claims verified",
                "grounded_nli_claims": (
                    f"{grounding_report.total_claims} verified "
                    f"({grounding_report.entailed_claims_count} entailed, "
                    f"{grounding_report.neutral_claims_count} neutral, "
                    f"{grounding_report.contradicted_claims_count} contradicted)"
                ),
            }

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            yield f"event: done\ndata: {json.dumps({'model': getattr(engine, 'model_name', 'Auto-Fallback'), 'provider': getattr(engine, 'last_used_name', 'Auto-Fallback'), 'citations': citations, 'latency_ms': latency_ms, 'metrics_comparison': metrics_comp})}\n\n"
        except Exception as exc:
            logger.error(f"[stream_compare_pipeline] Stream failed with error: {exc}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Get status and configurations of the grounded generation and guardrails engine",
)
def get_generation_status() -> dict[str, Any]:
    """Retrieve engine status, active provider, and default thresholds."""
    return {
        "status": "ready",
        "module": "Grounded LLM Inference & Anti-Hallucination Guardrails",
        "active_model": default_grounded_generator.engine.model_name,
        "default_sufficiency_threshold": default_grounded_generator.sufficiency_classifier.default_threshold,
        "default_hallucination_threshold": default_grounded_generator.verifier.default_tolerance,
        "abstention_protocol": "enabled",
        "nli_verification": "sentence-level",
    }



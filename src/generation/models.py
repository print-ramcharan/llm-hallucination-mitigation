"""Data models and schemas for Module 6: Grounded LLM Inference & Anti-Hallucination Guardrails."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.context.models import CompactedEvidence, OptimizedContext
from src.query_processing.models import ConversationTurn


class ClaimStatus(str, Enum):
    """Natural Language Inference (NLI) verification status for atomic claims."""

    ENTAILED = "entailed"
    NEUTRAL = "neutral"
    CONTRADICTED = "contradicted"


class ClaimVerification(BaseModel):
    """Fine-grained verification telemetry for an individual claim extracted from generated output."""

    claim_text: str = Field(..., description="The individual claim statement extracted from the answer.")
    status: ClaimStatus = Field(..., description="NLI verification classification against provided evidence.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the NLI verification classification.",
    )
    cited_sources: list[str] = Field(
        default_factory=list,
        description="Citation tags explicitly attached to this claim (e.g. ['[Doc 1, Chunk 0]']).",
    )
    entailing_chunk_id: str | None = Field(
        default=None,
        description="Chunk identifier that empirically entails or supports the claim.",
    )
    evidence_snippet: str | None = Field(
        default=None,
        description="Exact supporting or contradictory excerpt from the context passage.",
    )
    reasoning: str | None = Field(
        default=None,
        description="Diagnostic explanation of the entailment/neutral/contradiction determination.",
    )


class SufficiencyAssessment(BaseModel):
    """Pre-generation evaluation of whether retrieved evidence can answer the query."""

    is_sufficient: bool = Field(
        ...,
        description="Whether the retrieved evidence satisfies the information needs of the query.",
    )
    sufficiency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quantified evidence sufficiency score (0.0 = completely inadequate, 1.0 = fully answered).",
    )
    threshold: float = Field(
        ...,
        description="Sufficiency threshold theta below which the Abstention Protocol triggers.",
    )
    topic: str = Field(
        ...,
        description="Extracted focal topic or target entity of the query.",
    )
    abstention_message: str | None = Field(
        default=None,
        description="Formulated truthful abstention response if evidence is insufficient.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of sufficiency evaluation findings.",
    )
    matched_aspects: list[str] = Field(
        default_factory=list,
        description="Key query aspects and entities directly covered in the evidence passages.",
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="Query information requirements absent from the retrieved evidence passages.",
    )


class GroundingReport(BaseModel):
    """Post-generation anti-hallucination verification telemetry and faithfulness report."""

    faithfulness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of verified claims supported by evidence (entailed / total claims).",
    )
    hallucination_detected: bool = Field(
        ...,
        description="True if any claim is contradicted or if faithfulness is below acceptable tolerance.",
    )
    total_claims: int = Field(default=0, description="Total number of atomic claims evaluated.")
    entailed_claims_count: int = Field(default=0, description="Number of claims directly entailed by evidence.")
    neutral_claims_count: int = Field(default=0, description="Number of claims lacking direct evidence support.")
    contradicted_claims_count: int = Field(
        default=0,
        description="Number of claims directly refuted by evidence facts.",
    )
    claims: list[ClaimVerification] = Field(
        default_factory=list,
        description="Detailed verification for each atomic claim in the generated answer.",
    )
    verified_citations: list[str] = Field(
        default_factory=list,
        description="Citation tags in the generated answer that match valid context passages.",
    )
    unverified_citations: list[str] = Field(
        default_factory=list,
        description="Citation tags found in text that do not correspond to any provided evidence chunk.",
    )


class GenerationRequest(BaseModel):
    """Payload for submitting queries and evidence context to the grounded generator."""

    query: str = Field(..., description="User query or prompt to be answered.")
    context: OptimizedContext | None = Field(
        default=None,
        description="Compacted and reordered evidence context from Module 5.",
    )
    raw_evidence_chunks: list[CompactedEvidence] | None = Field(
        default=None,
        description="Alternative direct list of evidence chunks if OptimizedContext is not pre-packaged.",
    )
    conversation_history: list[ConversationTurn] | None = Field(
        default=None,
        description="Interactive conversational turns for contextual awareness.",
    )
    sufficiency_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Cutoff threshold below which generator truthfully abstains from answering.",
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
        description="Sampling temperature (default 0.0 for deterministic factual generation).",
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


class GenerationResponse(BaseModel):
    """Complete response payload containing generated answer, grounding telemetry, and provenance."""

    query: str = Field(..., description="Query evaluated.")
    answer: str = Field(..., description="Grounded answer text or truthful abstention statement.")
    abstained: bool = Field(
        ...,
        description="True if generator triggered the Abstention Protocol due to insufficient evidence.",
    )
    abstention_reason: str | None = Field(
        default=None,
        description="Rationale for abstaining if abstention was triggered.",
    )
    sufficiency: SufficiencyAssessment = Field(
        ...,
        description="Telemetry from the pre-generation Evidence Sufficiency Classifier.",
    )
    grounding_report: GroundingReport = Field(
        ...,
        description="Telemetry from the post-generation Anti-Hallucination NLI Verifier.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="All citation tags incorporated into the answer.",
    )
    latency_ms: float = Field(
        ...,
        description="End-to-end generation and verification latency in milliseconds.",
    )
    model_name: str = Field(
        ...,
        description="Identifier of the inference engine/model used for generation.",
    )


class StreamEvent(BaseModel):
    """Server-Sent Event (SSE) message payload for real-time token streaming."""

    event: str = Field(..., description="Event type: 'sufficiency', 'token', 'grounding', 'done', 'error'")
    data: dict[str, Any] = Field(..., description="Structured payload associated with the event.")


class NaiveGenerationResponse(BaseModel):
    """Response produced by a naive/baseline LLM without hybrid retrieval or evidence gating."""

    query: str = Field(..., description="User query evaluated.")
    answer: str = Field(..., description="Direct naive LLM output without evidence citations.")
    has_citations: bool = Field(default=False, description="Whether citations are present.")
    citations: list[str] = Field(default_factory=list, description="List of citations (empty for naive).")
    faithfulness_score: float | None = Field(
        default=None,
        description="Faithfulness score (unverified for naive).",
    )
    hallucination_risk: str = Field(
        default="High (Unverified / No Evidence Anchors)",
        description="Assessed hallucination risk level.",
    )
    latency_ms: float = Field(..., description="Generation latency in milliseconds.")
    model_name: str = Field(..., description="Model identifier used for naive inference.")


class ComparisonResponse(BaseModel):
    """Side-by-side comparison payload evaluating Naive LLM vs Our Grounded Mitigation Pipeline."""

    query: str = Field(..., description="The query processed by both pipelines.")
    naive: NaiveGenerationResponse = Field(..., description="Baseline Naive LLM output.")
    grounded: GenerationResponse = Field(..., description="Our Grounded Mitigation Pipeline output.")
    metrics_comparison: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary comparison metrics (citations, hallucination risk, sufficiency, NLI).",
    )

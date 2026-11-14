"""Module 6: Grounded LLM Inference & Anti-Hallucination Guardrails."""

from __future__ import annotations

from src.generation.engine import (
    BaseInferenceEngine,
    EngineFactory,
    OfflineGroundedEngine,
    OpenAICompatibleEngine,
    default_inference_engine,
)
from src.generation.generator import GroundedGenerator, default_grounded_generator
from src.generation.models import (
    ClaimStatus,
    ClaimVerification,
    GenerationRequest,
    GenerationResponse,
    GroundingReport,
    StreamEvent,
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

__all__ = [
    "BaseInferenceEngine",
    "EngineFactory",
    "OfflineGroundedEngine",
    "OpenAICompatibleEngine",
    "default_inference_engine",
    "GroundedGenerator",
    "default_grounded_generator",
    "ClaimStatus",
    "ClaimVerification",
    "GenerationRequest",
    "GenerationResponse",
    "GroundingReport",
    "StreamEvent",
    "SufficiencyAssessment",
    "GroundedPromptSynthesizer",
    "default_prompt_synthesizer",
    "EvidenceSufficiencyClassifier",
    "default_sufficiency_classifier",
    "AntiHallucinationVerifier",
    "default_anti_hallucination_verifier",
]

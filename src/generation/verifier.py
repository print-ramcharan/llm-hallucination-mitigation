"""Post-Generation Anti-Hallucination Verifier & Sentence-Level NLI Guardrail."""

from __future__ import annotations

import re

from src.context.models import CompactedEvidence, OptimizedContext
from src.generation.models import ClaimStatus, ClaimVerification, GroundingReport

# Negative/affirmative polarity indicators for contradiction detection
_NEGATIONS = {"not", "no", "never", "cannot", "can't", "won't", "prohibited", "disallowed", "banned", "restricted"}
_AFFIRMATIVES = {"allowed", "permitted", "mandatory", "required", "always", "authorized", "eligible"}


class AntiHallucinationVerifier:
    """Performs sentence-level Natural Language Inference (NLI) and claim grounding checks.

    Classifies each claim into:
      - ENTAILED: Directly supported by context facts and cited evidence.
      - CONTRADICTED: Directly conflicts with facts, numbers, or polarity in the evidence.
      - NEUTRAL: Unsubstantiated speculation or external facts absent from the context.
    """

    def __init__(self, default_tolerance: float = 0.70) -> None:
        self.default_tolerance = default_tolerance

    @staticmethod
    def extract_claims(text: str) -> list[str]:
        """Segment answer into atomic claims, stripping trailing citations."""
        # Split on sentence terminals
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims = []
        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean or len(s_clean) < 10:
                continue
            claims.append(s_clean)
        return claims

    @staticmethod
    def extract_citations(text: str) -> list[str]:
        """Extract all bracketed citation tags (e.g. ['[Doc 1, Chunk 0]'])."""
        return re.findall(r"\[Doc\s+[^\]]+\]", text)

    def _evaluate_nli(
        self,
        claim_text: str,
        evidence_text: str,
    ) -> tuple[ClaimStatus, float, str | None]:
        """Evaluate NLI status between claim (hypothesis) and evidence (premise).

        Returns:
            Tuple of (ClaimStatus, confidence_score, rationale).
        """
        # Clean claim text of citation tags for textual comparison
        clean_claim = re.sub(r"\[Doc\s+[^\]]+\]", "", claim_text).strip().lower()
        ev_lower = evidence_text.lower()

        claim_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", clean_claim))
        from src.generation.sufficiency import _STOP_WORDS
        content_tokens = {t for t in claim_tokens if t not in _STOP_WORDS}

        if not content_tokens:
            return ClaimStatus.ENTAILED, 0.95, "Claim contains no specific factual assertions."

        # 1. Contradiction Check: Numeric Discrepancies
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", clean_claim))
        ev_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", ev_lower))

        if claim_numbers:
            # If claim asserts numbers not present anywhere in the premise
            unsupported_numbers = claim_numbers - ev_numbers
            if unsupported_numbers and ev_numbers:
                return (
                    ClaimStatus.CONTRADICTED,
                    0.90,
                    f"Numerical contradiction: claim asserts numbers {unsupported_numbers} conflicting with evidence numbers {ev_numbers}.",
                )

        # 2. Contradiction Check: Polarity / Negation Flip on Most Relevant Evidence Sentence
        ev_sentences = re.split(r"(?<=[.!?])\s+", ev_lower)
        best_sentence = ""
        best_overlap = 0
        for s in ev_sentences:
            s_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", s))
            ov = len(content_tokens & s_words)
            if ov > best_overlap:
                best_overlap = ov
                best_sentence = s

        if best_sentence and best_overlap >= 2:
            best_sent_tokens = set(re.findall(r"\b\w+\b", best_sentence))
            claim_has_neg = bool(_NEGATIONS & claim_tokens)
            best_has_neg = bool(_NEGATIONS & best_sent_tokens)
            claim_has_aff = bool(_AFFIRMATIVES & claim_tokens)
            best_has_aff = bool(_AFFIRMATIVES & best_sent_tokens)

            # Contradiction occurs if claim explicitly inverts the polarity of the premise sentence:
            # E.g. Claim asserts negative ("not allowed") while premise is affirmative ("allowed"),
            # or Claim asserts affirmative ("permitted") while premise explicitly forbids ("prohibited/cannot").
            if (claim_has_neg and best_has_aff and not best_has_neg) or (
                claim_has_aff and best_has_neg and not claim_has_neg
            ):
                return (
                    ClaimStatus.CONTRADICTED,
                    0.85,
                    "Polarity contradiction: claim inverts the affirmative/negative constraints of the evidence.",
                )

        # 3. Entailment vs Neutral: Lexical & Semantic Overlap
        ev_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", ev_lower))
        overlap = content_tokens & ev_words
        overlap_ratio = len(overlap) / len(content_tokens)

        if overlap_ratio >= 0.50:
            confidence = min(0.99, 0.60 + 0.40 * overlap_ratio)
            return (
                ClaimStatus.ENTAILED,
                round(confidence, 3),
                f"Directly supported by evidence ({len(overlap)}/{len(content_tokens)} key factual terms matched).",
            )
        if overlap_ratio >= 0.25:
            # Weak or partial support
            return (
                ClaimStatus.NEUTRAL,
                0.60,
                f"Partially matched evidence ({len(overlap)}/{len(content_tokens)} terms), but critical assertions lack direct support.",
            )
        return (
            ClaimStatus.NEUTRAL,
            0.85,
            "Unsupported assertion: factual terms not found in the referenced evidence.",
        )

    def verify_answer(
        self,
        answer: str,
        context: OptimizedContext | None = None,
        evidence_items: list[CompactedEvidence] | None = None,
        tolerance: float | None = None,
    ) -> GroundingReport:
        """Execute comprehensive NLI verification across all claims in the generated answer."""
        tol = tolerance if tolerance is not None else self.default_tolerance

        # Map citation tags to evidence items
        items: list[CompactedEvidence] = []
        if context is not None and context.evidence_items:
            items = context.evidence_items
        elif evidence_items:
            items = evidence_items

        tag_to_evidence: dict[str, CompactedEvidence] = {}
        for item in items:
            tag_to_evidence[item.citation_tag] = item
            # Also support normalised tag (e.g. without extra spaces)
            norm_tag = re.sub(r"\s+", " ", item.citation_tag)
            tag_to_evidence[norm_tag] = item

        all_citations = self.extract_citations(answer)
        verified_citations: list[str] = []
        unverified_citations: list[str] = []

        for cite in all_citations:
            norm_cite = re.sub(r"\s+", " ", cite)
            if norm_cite in tag_to_evidence:
                if cite not in verified_citations:
                    verified_citations.append(cite)
            else:
                if cite not in unverified_citations:
                    unverified_citations.append(cite)

        claims = self.extract_claims(answer)
        if not claims:
            return GroundingReport(
                faithfulness_score=1.0,
                hallucination_detected=False,
                total_claims=0,
                entailed_claims_count=0,
                neutral_claims_count=0,
                contradicted_claims_count=0,
                claims=[],
                verified_citations=verified_citations,
                unverified_citations=unverified_citations,
            )

        verifications: list[ClaimVerification] = []
        entailed_count = 0
        neutral_count = 0
        contradicted_count = 0

        # Build full combined context fallback if claim citation is missing
        full_context_text = " ".join(
            f"{it.extracted_text} {it.original_text}" for it in items
        )

        for claim in claims:
            claim_cites = self.extract_citations(claim)
            supporting_item: CompactedEvidence | None = None

            # Look up cited evidence
            for c in claim_cites:
                norm_c = re.sub(r"\s+", " ", c)
                if norm_c in tag_to_evidence:
                    supporting_item = tag_to_evidence[norm_c]
                    break

            if supporting_item is not None:
                ev_text = f"{supporting_item.extracted_text} {supporting_item.original_text}"
                chunk_id = supporting_item.chunk_id
            else:
                ev_text = full_context_text
                chunk_id = None

            status, confidence, reasoning = self._evaluate_nli(claim, ev_text)

            if status == ClaimStatus.ENTAILED:
                entailed_count += 1
            elif status == ClaimStatus.CONTRADICTED:
                contradicted_count += 1
            else:
                neutral_count += 1

            snippet = ev_text[:160].strip() + "..." if ev_text else None

            verifications.append(
                ClaimVerification(
                    claim_text=claim,
                    status=status,
                    confidence=confidence,
                    cited_sources=claim_cites,
                    entailing_chunk_id=chunk_id,
                    evidence_snippet=snippet,
                    reasoning=reasoning,
                )
            )

        total = len(claims)
        faithfulness = round(entailed_count / total, 3) if total > 0 else 1.0

        # Hallucination detected if ANY claim is contradicted or if faithfulness is below tolerance
        hallucination_detected = (contradicted_count > 0) or (faithfulness < tol)

        return GroundingReport(
            faithfulness_score=faithfulness,
            hallucination_detected=hallucination_detected,
            total_claims=total,
            entailed_claims_count=entailed_count,
            neutral_claims_count=neutral_count,
            contradicted_claims_count=contradicted_count,
            claims=verifications,
            verified_citations=verified_citations,
            unverified_citations=unverified_citations,
        )


default_anti_hallucination_verifier = AntiHallucinationVerifier()

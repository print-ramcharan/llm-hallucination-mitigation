"""Token Budget Manager and Citation Formatter."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.chunking.tokenizer import count_tokens


class TokenBudgetManager:
    """Enforces prompt context token allowances and formats explicit evidence citations.

    Generates structured citation blocks:
      [Doc X, Chunk Y] Source: filename (Page Z, Section: S)
      <compacted evidence text>
    """

    def __init__(self, default_budget: int = 1500) -> None:
        self.default_budget = default_budget

    @staticmethod
    def generate_citation_tag(
        doc_num: int,
        chunk_index: int,
    ) -> str:
        """Generate canonical citation tag e.g. '[Doc 1, Chunk 2]'."""
        return f"[Doc {doc_num}, Chunk {chunk_index + 1}]"

    def format_evidence_block(
        self,
        citation_tag: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Format an individual evidence section with citation header and source provenance."""
        source_name = metadata.get("source", metadata.get("source_name", "Unknown Source"))
        page = metadata.get("page")
        section = metadata.get("section")

        provenance_parts = [f"Source: {source_name}"]
        if page is not None:
            provenance_parts.append(f"Page {page}")
        if section:
            provenance_parts.append(f"Section: {section}")

        header = f"{citation_tag} ({', '.join(provenance_parts)})"
        return f"{header}\n{text.strip()}"

    def fit_within_budget(
        self,
        evidence_candidates: List[Dict[str, Any]],
        max_budget: int | None = None,
    ) -> Tuple[str, List[Dict[str, Any]], int]:
        """Pack evidence items into the prompt context strictly respecting the token budget.

        Args:
            evidence_candidates: Ordered evidence dicts containing 'text', 'metadata', etc.
            max_budget: Token limit (defaults to self.default_budget).

        Returns:
            Tuple of (formatted_prompt_context, accepted_evidence_dicts, total_tokens_consumed).
        """
        budget = self.default_budget if max_budget is None else max_budget

        accepted_blocks: List[str] = []
        accepted_items: List[Dict[str, Any]] = []
        current_token_count = 0

        # Unique document tracking for Doc 1, Doc 2 numbering
        doc_id_to_num: Dict[str, int] = {}

        for item in evidence_candidates:
            doc_id = item["document_id"]
            if doc_id not in doc_id_to_num:
                doc_id_to_num[doc_id] = len(doc_id_to_num) + 1
            doc_num = doc_id_to_num[doc_id]

            chunk_idx = item.get("chunk_index", 0)
            citation_tag = self.generate_citation_tag(doc_num, chunk_idx)

            formatted_block = self.format_evidence_block(
                citation_tag=citation_tag,
                text=item["extracted_text"],
                metadata=item.get("metadata", {}),
            )

            block_tokens = count_tokens(formatted_block)

            # Check if admitting this block fits within the remaining allowance
            if current_token_count + block_tokens <= budget:
                accepted_blocks.append(formatted_block)
                item["citation_tag"] = citation_tag
                item["block_token_count"] = block_tokens
                accepted_items.append(item)
                current_token_count += block_tokens
            else:
                # If budget is full, stop admitting further lower-priority blocks
                break

        full_context = "\n\n".join(accepted_blocks)
        total_tokens = count_tokens(full_context)
        return full_context, accepted_items, total_tokens

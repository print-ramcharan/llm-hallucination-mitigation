"""Module 5: Extractive Context Compaction & Optimization Package."""

from src.context.budget_manager import TokenBudgetManager
from src.context.compactor import ContextCompactor, default_context_compactor
from src.context.deduplicator import ContextDeduplicator
from src.context.models import (
    CompactedEvidence,
    CompactionRequest,
    OptimizedContext,
)
from src.context.reorderer import LostInTheMiddleReorderer
from src.context.sentence_pruner import SentencePruner

__all__ = [
    "CompactedEvidence",
    "CompactionRequest",
    "ContextCompactor",
    "ContextDeduplicator",
    "LostInTheMiddleReorderer",
    "OptimizedContext",
    "SentencePruner",
    "TokenBudgetManager",
    "default_context_compactor",
]

"""Module 7: Cross-Session External Memory & Client Sync package."""

from src.memory.models import (
    EpisodicMemoryItem,
    HierarchicalSummary,
    MemoryExport,
    MemoryPermissions,
    MemorySearchQuery,
    MemorySearchResult,
    MemorySyncContext,
    Session,
    WebContextCaptureRequest,
)
from src.memory.session_registry import SessionRegistry, default_session_registry
from src.memory.store import ExternalMemoryStore, default_memory_store
from src.memory.summarizer import (
    HierarchicalSummarizer,
    default_hierarchical_summarizer,
)
from src.memory.synchronizer import (
    CrossSessionMemorySynchronizer,
    default_memory_synchronizer,
)

__all__ = [
    "EpisodicMemoryItem",
    "HierarchicalSummary",
    "MemoryExport",
    "MemoryPermissions",
    "MemorySearchQuery",
    "MemorySearchResult",
    "MemorySyncContext",
    "Session",
    "WebContextCaptureRequest",
    "ExternalMemoryStore",
    "default_memory_store",
    "SessionRegistry",
    "default_session_registry",
    "HierarchicalSummarizer",
    "default_hierarchical_summarizer",
    "CrossSessionMemorySynchronizer",
    "default_memory_synchronizer",
]

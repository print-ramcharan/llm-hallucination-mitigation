"""Shared test configuration and offline environment for test suites."""

try:
    import faiss  # noqa: F401 - Must be imported before torch on Windows to avoid OpenMP DLL collision
except Exception:
    pass

import hashlib
import os
import re

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LLM_PROVIDER", "offline")

EMBEDDING_DIMENSION = 384
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "each",
    "for",
    "from",
    "full",
    "how",
    "in",
    "is",
    "it",
    "many",
    "of",
    "on",
    "or",
    "per",
    "receive",
    "staff",
    "taking",
    "the",
    "to",
    "verified",
    "what",
    "where",
    "which",
    "with",
    "year",
}
SYNONYMS = {
    "allowance": "stipend",
    "allowances": "stipend",
    "annual": "year",
    "annually": "year",
    "benefits": "benefit",
    "days": "day",
    "employees": "employee",
    "healthcare": "health",
    "holiday": "leave",
    "holidays": "leave",
    "medical": "health",
    "timeoff": "leave",
    "vacation": "leave",
    "vacations": "leave",
    "yearly": "year",
}


def _tokens(text: str) -> set[str]:
    text = re.sub(r"\btime\s+off\b", "timeoff", text.lower())
    return {
        SYNONYMS.get(token, token)
        for token in re.findall(r"\b[a-z0-9]+\b", text)
        if token not in STOP_WORDS
    }


class _OfflineSentenceTransformer:
    """Deterministic lexical stand-in for tests that exercise embedding plumbing."""

    def __init__(self, _model_name: str) -> None:
        pass

    def get_sentence_embedding_dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def encode(self, texts: list[str], **_kwargs) -> np.ndarray:
        vectors = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _tokens(text):
                feature = (
                    int.from_bytes(
                        hashlib.sha256(token.encode("utf-8")).digest()[:4], "big"
                    )
                    % EMBEDDING_DIMENSION
                )
                vectors[row, feature] += 1.0
        return vectors


class _OfflineCrossEncoder:
    """Deterministic lexical stand-in for reranking behavior tests."""

    def __init__(self, _model_name: str) -> None:
        pass

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = []
        for query, passage in pairs:
            query_tokens = _tokens(query)
            overlap = len(query_tokens & _tokens(passage))
            scores.append(5.0 * overlap / max(len(query_tokens), 1) - 1.0)
        return np.asarray(scores, dtype=np.float32)


try:
    import sentence_transformers

    sentence_transformers.SentenceTransformer = _OfflineSentenceTransformer
    sentence_transformers.CrossEncoder = _OfflineCrossEncoder
except Exception:
    pass


@pytest.fixture(autouse=True)
def use_offline_transformer_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit and integration tests independent of model downloads and Hub access."""
    import sentence_transformers

    from src.indexing import embeddings
    from src.reranking import engine

    monkeypatch.setattr(embeddings, "SentenceTransformer", _OfflineSentenceTransformer)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _OfflineCrossEncoder)
    monkeypatch.setattr(
        embeddings.default_embedding_engine,
        "_model",
        _OfflineSentenceTransformer("all-MiniLM-L6-v2"),
    )
    monkeypatch.setattr(
        engine.default_cross_encoder_engine,
        "_model",
        _OfflineCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )

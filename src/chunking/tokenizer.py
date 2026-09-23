"""Tokenizer utility for token counting and overlap slicing."""

from __future__ import annotations

import re

# Try importing tiktoken if available, else fallback to standard subword/word regex
_TIKTOKEN_ENCODER = None
try:
    import tiktoken
    try:
        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TIKTOKEN_ENCODER = None
except ImportError:
    _TIKTOKEN_ENCODER = None

# Fallback regex for token extraction (matching words, contractions, numbers, punctuation)
_TOKEN_REGEX = re.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\s]|\w+""")


def tokenize(text: str) -> list[str]:
    """Tokenize text into string tokens."""
    if not text:
        return []
    if _TIKTOKEN_ENCODER is not None:
        token_ids = _TIKTOKEN_ENCODER.encode(text)
        return [_TIKTOKEN_ENCODER.decode([tid]) for tid in token_ids]
    return _TOKEN_REGEX.findall(text)


def count_tokens(text: str) -> int:
    """Return the exact or estimated token count of a given text."""
    if not text or not text.strip():
        return 0
    if _TIKTOKEN_ENCODER is not None:
        return len(_TIKTOKEN_ENCODER.encode(text))
    tokens = _TOKEN_REGEX.findall(text)
    return len(tokens)


def get_trailing_token_overlap(text: str, overlap_tokens: int = 50) -> str:
    """Extract approximately `overlap_tokens` from the tail of `text`.

    Preserves exact text spacing, casing, and snaps to sentence boundaries
    where feasible for semantic coherence.
    """
    if not text or overlap_tokens <= 0:
        return ""

    if _TIKTOKEN_ENCODER is not None:
        token_ids = _TIKTOKEN_ENCODER.encode(text)
        if len(token_ids) <= overlap_tokens:
            return text.strip()
        tail_ids = token_ids[-overlap_tokens:]
        raw_overlap = _TIKTOKEN_ENCODER.decode(tail_ids).strip()
    else:
        matches = list(_TOKEN_REGEX.finditer(text))
        if len(matches) <= overlap_tokens:
            return text.strip()
        start_char_idx = matches[-overlap_tokens].start()
        raw_overlap = text[start_char_idx:].strip()

    # If there is a sentence boundary near the start of the overlap,
    # snap to it if it retains at least 70% of requested overlap tokens
    sentence_breaks = list(re.finditer(r"[.!?]\s+", raw_overlap))
    if sentence_breaks:
        first_break = sentence_breaks[0]
        candidate = raw_overlap[first_break.end():].strip()
        if count_tokens(candidate) >= int(overlap_tokens * 0.7):
            return candidate

    return raw_overlap

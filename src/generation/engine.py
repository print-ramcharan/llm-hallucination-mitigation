"""Inference & Token Streaming Engines supporting fallback cascade across Gemini, Groq, and OpenRouter."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import NoReturn

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from openai import APIConnectionError, APIStatusError

from src.chunking.tokenizer import count_tokens

load_dotenv()

logger = logging.getLogger("uvicorn.error")


_GROQ_MODEL_ALIASES = {
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
    "llama-3.1-70b-versatile": "openai/gpt-oss-120b",
    "llama-3.1-8b-instant": "openai/gpt-oss-20b",
    "llama3-70b-8192": "openai/gpt-oss-120b",
    "llama3-8b-8192": "openai/gpt-oss-20b",
}

_GEMINI_MODEL_ALIASES = {
    "gemini-1.5-flash": "gemini-3.5-flash-lite",
    "gemini-1.5-flash-latest": "gemini-3.5-flash-lite",
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-2.0-flash": "gemini-3.5-flash-lite",
    "gemini-2.0-flash-lite": "gemini-3.5-flash-lite",
    "gemini-2.5-flash": "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
}


class BaseInferenceEngine(ABC):
    """Abstract base class for LLM inference providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the underlying model."""

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return self.__class__.__name__.replace("Engine", "")

    def is_configured(self) -> bool:
        """Check if this engine has required credentials configured."""
        return True

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Synchronously generate a complete completion string."""

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream token-by-token completion chunks."""


class OfflineGroundedEngine(BaseInferenceEngine):
    """High-fidelity, deterministic offline engine for grounded synthesis and testing.

    Operates hermetically without external network access, API keys, or GPU dependencies.
    Extracts the most salient factual propositions directly from the provided evidence context
    and injects verified bracketed citations.
    """

    def __init__(self, model_name: str = "offline-grounded-synthesizer") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def _extract_evidence_passages(self, prompt: str) -> list[tuple[str, str]]:
        """Parse (citation_tag, passage_text) from prompt evidence block."""
        passages: list[tuple[str, str]] = []
        # Match patterns like: --- EVIDENCE [Doc 1, Chunk 0] --- or [Doc 1, Chunk 0]
        pattern = r"(?:---\s*EVIDENCE\s*)?(\[Doc\s+[^\]]+\])(?:\s*\(Source:[^)]*\)\s*---)?\s*\n(.*?)(?=(?:---\s*EVIDENCE\s*\[Doc|USER QUERY:|$))"
        matches = re.findall(pattern, prompt, re.DOTALL)
        for tag, text in matches:
            cleaned = text.strip()
            if cleaned:
                passages.append((tag.strip(), cleaned))

        if not passages:
            # Fallback for plain blocks with tags inside
            lines = prompt.splitlines()
            cur_tag = "[Doc 1, Chunk 0]"
            cur_buf = []
            for line in lines:
                tag_match = re.search(r"(\[Doc\s+[^\]]+\])", line)
                if tag_match:
                    if cur_buf:
                        passages.append((cur_tag, "\n".join(cur_buf).strip()))
                        cur_buf = []
                    cur_tag = tag_match.group(1)
                elif "USER QUERY:" in line or "Question:" in line:
                    break
                else:
                    cur_buf.append(line)
            if cur_buf:
                passages.append((cur_tag, "\n".join(cur_buf).strip()))

        if not passages:
            # Fallback for unbracketed conventional context blocks (e.g. Context:\n...\n\nQuestion:)
            context_match = re.search(
                r"(?:Context:\s*\n|---\s*CONTEXT\s*---\s*\n)(.*?)(?=\n\nQuestion:|\n\nUSER QUERY:|$)",
                prompt,
                re.DOTALL | re.IGNORECASE,
            )
            if context_match and context_match.group(1).strip():
                passages.append(("[Doc 1, Chunk 0]", context_match.group(1).strip()))

        return passages

    def _extract_query(self, prompt: str) -> str:
        """Extract the query string from the prompt."""
        m = re.search(
            r"(?:USER QUERY:|Question:)\s*\n?(.*?)(?=\n\nGROUNDED ANSWER|\nAnswer:|$)",
            prompt,
            re.DOTALL | re.IGNORECASE,
        )
        return m.group(1).strip() if m else ""

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Synthesize a grounded answer with citations from the prompt context."""
        query = self._extract_query(prompt)
        passages = self._extract_evidence_passages(prompt)

        if not passages:
            return "No evidence passages were available to answer this inquiry."

        query_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query.lower()))
        # Filter stopwords
        from src.generation.sufficiency import _SEMANTIC_SYNONYMS, _STOP_WORDS

        content_words = {w for w in query_words if w not in _STOP_WORDS}
        expanded_words = set(content_words)
        for w in content_words:
            expanded_words.update(_SEMANTIC_SYNONYMS.get(w, set()))

        from src.context.sentence_pruner import SentencePruner

        is_list_query = bool(
            re.search(
                r"\b(?:questions?|steps?|factors?|criteria|reasons?|items?|list|all|what are)\b",
                query,
                re.IGNORECASE,
            )
        )
        max_sentences = 12 if is_list_query else 4

        scored_sentences: list[tuple[float, int, str, str]] = []
        sentence_seq = 0
        for tag, passage in passages:
            # Split into individual sentences preserving list items
            sentences = SentencePruner.split_sentences(passage)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean or len(s_clean) < 10:
                    continue
                s_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", s_clean.lower()))
                overlap = len(expanded_words & s_words)
                score = overlap / (len(content_words) + 1e-5)
                if is_list_query and SentencePruner._is_list_or_schedule_item(s_clean):
                    score += 0.15
                scored_sentences.append((score, sentence_seq, s_clean, tag))
                sentence_seq += 1

        # Sort by relevance score descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if not scored_sentences or scored_sentences[0][0] <= 0:
            # If no direct term overlap, use the first high-salience sentence in original sequence
            top_sentences = (
                sorted(scored_sentences[:2], key=lambda x: x[1])
                if scored_sentences
                else []
            )
        else:
            selected: list[tuple[float, int, str, str]] = []
            seen = set()
            for score, seq, sent, tag in scored_sentences:
                normalized = sent.lower()[:30]
                if normalized not in seen and score > 0:
                    seen.add(normalized)
                    selected.append((score, seq, sent, tag))
                if len(selected) >= max_sentences:
                    break
            top_sentences = sorted(selected, key=lambda x: x[1])

        if not top_sentences:
            return "The retrieved context does not contain sufficient details to address the query."

        is_plain_prompt = bool(
            re.search(r"Question:.*?\nAnswer:", prompt, re.IGNORECASE)
        )
        # Format answer statements with citation tags
        answer_parts = []
        for _, _, sent, tag in top_sentences:
            sent_trimmed = sent.rstrip(". ")
            if is_plain_prompt:
                answer_parts.append(f"{sent_trimmed}.")
            else:
                answer_parts.append(f"{sent_trimmed} {tag}.")

        return " ".join(answer_parts)

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream generated text as word/token chunks."""
        full_text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        tokens = re.findall(r"\S+\s*", full_text)
        yield from tokens


class OpenAICompatibleEngine(BaseInferenceEngine):
    """LangChain client for OpenAI-compatible chat-completion APIs."""

    requires_api_key = False
    context_window_tokens: int | None = None
    context_safety_margin_tokens = 8192
    _TRUNCATION_MARKER = "\n\n[... Context truncated to fit model context window ...]\n\n"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ).rstrip("/")
        self.api_key = (
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "dummy-key")
        )
        self._model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "llama3.2")
        self.default_headers = default_headers

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_configured(self) -> bool:
        return not self.requires_api_key or bool(self.api_key and self.api_key.strip())

    def _truncate_prompt_by_chars(self, prompt: str, max_chars: int) -> str:
        marker = self._TRUNCATION_MARKER
        if len(prompt) <= max_chars:
            return prompt

        query_start = prompt.rfind("USER QUERY:")
        suffix = prompt[query_start:] if query_start >= 0 else ""
        if len(suffix) + len(marker) < max_chars:
            prefix_limit = max_chars - len(suffix) - len(marker)
            return prompt[:prefix_limit] + marker + suffix

        return prompt[: max_chars - len(marker)] + marker

    def _truncate_prompt_by_tokens(self, prompt: str, max_tokens: int) -> str:
        marker = self._TRUNCATION_MARKER
        query_start = prompt.rfind("USER QUERY:")
        prefix = prompt[:query_start] if query_start >= 0 else prompt
        suffix = prompt[query_start:] if query_start >= 0 else ""
        suffix_budget = min(4096, max_tokens // 4)

        if suffix and count_tokens(suffix) > suffix_budget:
            query_header = "USER QUERY:"
            query_body = suffix[len(query_header) :]
            body_budget = max(0, suffix_budget - count_tokens(query_header))
            left, right = 0, len(query_body)
            while left < right:
                middle = (left + right) // 2
                if count_tokens(query_body[middle:]) <= body_budget:
                    right = middle
                else:
                    left = middle + 1
            suffix = query_header + query_body[left:]

        left, right = 0, len(prefix)
        while left < right:
            middle = (left + right + 1) // 2
            candidate = prefix[:middle] + marker + suffix
            if count_tokens(candidate) <= max_tokens:
                left = middle
            else:
                right = middle - 1

        return prefix[:left] + marker + suffix

    def _truncate_prompt_if_needed(
        self,
        prompt: str,
        max_chars: int = 350_000,
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> str:
        """Prevent character and provider token limits from rejecting large prompts."""
        safe_prompt = self._truncate_prompt_by_chars(prompt, max_chars)
        if len(safe_prompt) != len(prompt):
            logger.warning(
                f"[{self.name}] Prompt length ({len(prompt)} chars) exceeds safe threshold ({max_chars} chars). Truncating."
            )

        if self.context_window_tokens is not None:
            available_prompt_tokens = (
                self.context_window_tokens
                - max_tokens
                - count_tokens(system_prompt)
                - self.context_safety_margin_tokens
            )
            if count_tokens(safe_prompt) > available_prompt_tokens:
                logger.warning(
                    f"[{self.name}] Prompt exceeds the {self.context_window_tokens}-token context window. Truncating."
                )
                safe_prompt = self._truncate_prompt_by_tokens(
                    safe_prompt, max(0, available_prompt_tokens)
                )

        return safe_prompt

    def _raise_api_error(self, error: APIConnectionError | APIStatusError) -> NoReturn:
        if isinstance(error, APIConnectionError):
            raise RuntimeError(
                f"Cannot connect to LLM service at {self.base_url}. If using Ollama, ensure it is running (`ollama serve`), or switch LLM Engine to 'Offline (Deterministic)'."
            ) from error

        if error.status_code in (401, 403):
            raise RuntimeError(
                f"LLM API authorization failed (HTTP {error.status_code}) for model '{self._model_name}'. Configure a valid API key or switch to 'Offline (Deterministic)'."
            ) from error

        if error.status_code == 400:
            err_text = error.response.text[:300] if hasattr(error, "response") else str(error)
            if any(term in err_text.lower() for term in ("context length", "too large", "maximum context", "tokens")):
                raise RuntimeError(
                    f"LLM context limit exceeded for model '{self._model_name}': {err_text}"
                ) from error

        raise RuntimeError(
            f"LLM API error (HTTP {error.status_code}): {error.response.text[:200]}"
        ) from error

    def _chat_model(self, temperature: float, max_tokens: int) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self._model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60.0,
            default_headers=self.default_headers,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a completion through LangChain's shared model interface."""
        from langchain_core.messages import HumanMessage, SystemMessage

        safe_prompt = self._truncate_prompt_if_needed(
            prompt, system_prompt=system_prompt, max_tokens=max_tokens
        )
        try:
            response = self._chat_model(temperature, max_tokens).invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=safe_prompt)]
            )
        except (APIConnectionError, APIStatusError) as error:
            self._raise_api_error(error)
        if not isinstance(response.content, str):
            raise TypeError("The configured chat model returned non-text content.")
        return response.content

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream completion chunks through the shared LangChain model interface."""
        from langchain_core.messages import HumanMessage, SystemMessage

        safe_prompt = self._truncate_prompt_if_needed(
            prompt, system_prompt=system_prompt, max_tokens=max_tokens
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=safe_prompt),
        ]
        try:
            for chunk in self._chat_model(temperature, max_tokens).stream(messages):
                if not isinstance(chunk.content, str):
                    raise TypeError(
                        "The configured chat model returned non-text content."
                    )
                if chunk.content:
                    yield chunk.content
        except (APIConnectionError, APIStatusError) as error:
            self._raise_api_error(error)


class AnthropicEngine(BaseInferenceEngine):
    """Client for Anthropic Claude inference API (LLM Provider Layer: Claude)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._model_name = model_name or os.getenv(
            "ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20241022"
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if not self.api_key:
            return OfflineGroundedEngine(
                model_name=f"{self._model_name}-offline"
            ).generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        import httpx

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        with httpx.Client(timeout=60.0) as client:
            try:
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["content"][0]["text"]
            except httpx.ConnectError:
                raise RuntimeError(
                    "Cannot connect to Anthropic API. Check your network or switch to 'Offline (Deterministic)'."
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403):
                    raise RuntimeError(
                        f"Anthropic API authorization failed (HTTP {e.response.status_code}). Configure ANTHROPIC_API_KEY or switch to 'Offline (Deterministic)'."
                    )
                raise RuntimeError(
                    f"Anthropic API error (HTTP {e.response.status_code}): {e.response.text[:200]}"
                )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        full_text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        tokens = re.findall(r"\S+\s*", full_text)
        yield from tokens


class GeminiEngine(BaseInferenceEngine):
    """Client for Google Gemini API with native SSE token streaming."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        )
        raw_model = model_name or os.getenv(
            "GEMINI_MODEL_NAME", "gemini-3.5-flash-lite"
        )
        self._model_name = _GEMINI_MODEL_ALIASES.get(raw_model, raw_model)

    @property
    def model_name(self) -> str:
        return f"gemini/{self._model_name}"

    @property
    def name(self) -> str:
        return "Gemini"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _truncate_prompt_if_needed(self, prompt: str, max_chars: int = 500_000) -> str:
        """Prevent context length overflow on massive inputs."""
        if len(prompt) > max_chars:
            logger.warning(
                f"[Gemini] Prompt length ({len(prompt)} chars) exceeds threshold ({max_chars} chars). Truncating."
            )
            return (
                prompt[:max_chars]
                + "\n\n[... Warning: Context truncated to fit model context window ...]"
            )
        return prompt

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if not self.is_configured():
            raise ValueError("GEMINI_API_KEY is not configured in .env.")
        import httpx

        safe_prompt = self._truncate_prompt_if_needed(prompt)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent"
        headers = {"x-goog-api-key": self.api_key}
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": safe_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 404 and self._model_name != "gemini-3.5-flash-lite":
                logger.warning(
                    f"[GeminiEngine] Model {self._model_name} returned 404. Falling back to gemini-3.5-flash-lite..."
                )
                url_fallback = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
                resp = client.post(url_fallback, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates or "content" not in candidates[0]:
                raise RuntimeError("No candidate response returned by Gemini API.")
            return candidates[0]["content"]["parts"][0]["text"]

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        if not self.is_configured():
            raise ValueError("GEMINI_API_KEY is not configured in .env.")
        import json

        import httpx

        safe_prompt = self._truncate_prompt_if_needed(prompt)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:streamGenerateContent?alt=sse"
        headers = {"x-goog-api-key": self.api_key}
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": safe_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        with httpx.Client(timeout=45.0) as client:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code == 404 and self._model_name != "gemini-3.5-flash-lite":
                    logger.warning(
                        f"[GeminiEngine] Model {self._model_name} stream returned 404. Falling back to gemini-3.5-flash-lite..."
                    )
                    url_fallback = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:streamGenerateContent?alt=sse"
                    with client.stream("POST", url_fallback, headers=headers, json=payload) as fb_resp:
                        fb_resp.raise_for_status()
                        for line in fb_resp.iter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            try:
                                data = json.loads(line[len("data: ") :].strip())
                                candidates = data.get("candidates", [])
                                if candidates and "content" in candidates[0]:
                                    for part in candidates[0]["content"].get("parts", []):
                                        text = part.get("text", "")
                                        if text:
                                            yield text
                            except Exception:
                                continue
                    return

                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[len("data: ") :].strip()
                        try:
                            data = json.loads(data_str)
                            candidates = data.get("candidates", [])
                            if candidates and "content" in candidates[0]:
                                parts = candidates[0]["content"].get("parts", [])
                                for part in parts:
                                    text = part.get("text", "")
                                    if text:
                                        yield text
                        except Exception:
                            continue


class GroqEngine(OpenAICompatibleEngine):
    """Groq adapter using its OpenAI-compatible chat-completions API."""

    requires_api_key = True
    context_window_tokens = 131_072

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        raw_model = model_name or os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
        resolved_model = _GROQ_MODEL_ALIASES.get(raw_model, raw_model)
        super().__init__(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key or os.getenv("GROQ_API_KEY", ""),
            model_name=resolved_model,
        )

    @property
    def model_name(self) -> str:
        return f"groq/{self._model_name}"

    @property
    def name(self) -> str:
        return "Groq"


class OpenRouterEngine(OpenAICompatibleEngine):
    """OpenRouter adapter using its OpenAI-compatible chat-completions API."""

    requires_api_key = True
    context_window_tokens = 131_072

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
            model_name=model_name
            or os.getenv("OPENROUTER_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct"),
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Mitigating Context Degradation",
            },
        )

    @property
    def model_name(self) -> str:
        return f"openrouter/{self._model_name}"

    @property
    def name(self) -> str:
        return "OpenRouter"


class FallbackInferenceEngine(BaseInferenceEngine):
    """Dynamic Auto-Fallback Cascade across Gemini, Groq, and OpenRouter.

    Attempts primary provider first. If rate limited (HTTP 429), quota exceeded,
    or connection fails, seamlessly switches to the next configured provider.
    """

    def __init__(
        self,
        engines: list[BaseInferenceEngine] | None = None,
    ) -> None:
        if engines is None:
            order_env = os.getenv("LLM_FALLBACK_ORDER", "gemini,groq,openrouter")
            order = [p.strip().lower() for p in order_env.split(",") if p.strip()]
            engine_map: dict[str, BaseInferenceEngine] = {
                "gemini": GeminiEngine(),
                "groq": GroqEngine(),
                "openrouter": OpenRouterEngine(),
            }
            self.engines = [engine_map[k] for k in order if k in engine_map]
            for k, eng in engine_map.items():
                if eng not in self.engines:
                    self.engines.append(eng)
        else:
            self.engines = engines

        self._emergency_offline = OfflineGroundedEngine()
        self._last_used_engine: BaseInferenceEngine | None = None

    @property
    def model_name(self) -> str:
        if self._last_used_engine:
            return self._last_used_engine.model_name
        configured = [e for e in self.engines if e.is_configured()]
        if configured:
            return f"Auto-Fallback ({' -> '.join(e.name for e in configured)})"
        return "Auto-Fallback (Awaiting API Keys in .env)"

    @property
    def name(self) -> str:
        return "Auto-Fallback Cascade"

    @property
    def last_used_name(self) -> str:
        if self._last_used_engine:
            return getattr(
                self._last_used_engine, "name", self._last_used_engine.model_name
            )
        return "Auto-Fallback"

    def is_configured(self) -> bool:
        return any(e.is_configured() for e in self.engines)

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        configured_engines = [e for e in self.engines if e.is_configured()]

        if not configured_engines:
            logger.warning(
                "[FallbackEngine] No API keys in .env. Using offline emergency synthesis."
            )
            self._last_used_engine = self._emergency_offline
            return self._emergency_offline.generate(
                prompt, system_prompt, temperature, max_tokens
            )

        errors: list[str] = []
        for engine in configured_engines:
            try:
                logger.info(
                    f"[FallbackEngine] Attempting generation with {engine.name} ({engine.model_name})..."
                )
                answer = engine.generate(prompt, system_prompt, temperature, max_tokens)
                self._last_used_engine = engine
                logger.info(f"[FallbackEngine] Succeeded with {engine.name}.")
                return answer
            except Exception as err:
                logger.warning(
                    f"[FallbackEngine] Provider {engine.name} failed: {err}. Cascading to next..."
                )
                errors.append(f"{engine.name}: {err}")

        err_summary = "; ".join(errors)
        logger.error(f"[FallbackEngine] All providers failed: {err_summary}")
        self._last_used_engine = self._emergency_offline
        fallback_note = f"[Notice: Cloud LLM providers were unavailable ({err_summary}). Running offline fallback.]\n\n"
        return fallback_note + self._emergency_offline.generate(
            prompt, system_prompt, temperature, max_tokens
        )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        configured_engines = [e for e in self.engines if e.is_configured()]

        if not configured_engines:
            logger.warning(
                "[FallbackEngine] No API keys in .env. Using offline emergency synthesis."
            )
            self._last_used_engine = self._emergency_offline
            yield from self._emergency_offline.generate_stream(
                prompt, system_prompt, temperature, max_tokens
            )
            return

        errors: list[str] = []
        stream_started = False

        for engine in configured_engines:
            try:
                logger.info(
                    f"[FallbackEngine] Connecting stream with {engine.name} ({engine.model_name})..."
                )
                token_iterator = engine.generate_stream(
                    prompt, system_prompt, temperature, max_tokens
                )
                first_token = next(token_iterator, None)
                if first_token is None:
                    logger.warning(
                        f"[FallbackEngine] Stream with {engine.name} produced no tokens. Cascading..."
                    )
                    errors.append(f"{engine.name}: empty stream")
                    continue

                self._last_used_engine = engine
                stream_started = True

                yield first_token
                for token in token_iterator:
                    yield token
                return
            except Exception as err:
                logger.warning(
                    f"[FallbackEngine] Stream with {engine.name} failed: {err}. Cascading..."
                )
                errors.append(f"{engine.name}: {err}")
                if stream_started:
                    raise

        err_summary = "; ".join(errors)
        logger.error(f"[FallbackEngine] All streaming providers failed: {err_summary}")
        self._last_used_engine = self._emergency_offline
        yield f"[Notice: Cloud LLM providers were unavailable ({err_summary}). Running offline fallback.]\n\n"
        yield from self._emergency_offline.generate_stream(
            prompt, system_prompt, temperature, max_tokens
        )


class EngineFactory:
    """Factory creating appropriate inference engine matching LLM Provider Layer."""

    @staticmethod
    def create_engine(provider: str | None = None) -> BaseInferenceEngine:
        selected = (provider or os.getenv("LLM_PROVIDER", "fallback")).lower()
        if selected in ("gemini", "google"):
            return GeminiEngine()
        if selected in ("groq",):
            return GroqEngine()
        if selected in ("openrouter",):
            return OpenRouterEngine()
        if selected in ("ollama",):
            return OpenAICompatibleEngine(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                model_name=os.getenv("OLLAMA_MODEL_NAME", "llama3.2"),
            )
        if selected in ("chatgpt", "openai"):
            return OpenAICompatibleEngine()
        if selected in ("claude", "anthropic"):
            return AnthropicEngine()
        if selected in ("offline",):
            return OfflineGroundedEngine()
        return default_fallback_engine


default_fallback_engine = FallbackInferenceEngine()
default_inference_engine = default_fallback_engine

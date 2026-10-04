"""Inference & Token Streaming Engines supporting fallback cascade across Gemini, Groq, and OpenRouter."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("uvicorn.error")


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
                elif "USER QUERY:" in line:
                    break
                else:
                    cur_buf.append(line)
            if cur_buf:
                passages.append((cur_tag, "\n".join(cur_buf).strip()))

        return passages

    def _extract_query(self, prompt: str) -> str:
        """Extract the query string from the prompt."""
        m = re.search(r"USER QUERY:\s*\n(.*?)(?=\n\nGROUNDED ANSWER|$)", prompt, re.DOTALL)
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

        scored_sentences: list[tuple[float, str, str]] = []
        for tag, passage in passages:
            # Split into individual sentences
            sentences = re.split(r"(?<=[.!?])\s+", passage)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean or len(s_clean) < 15:
                    continue
                s_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", s_clean.lower()))
                overlap = len(expanded_words & s_words)
                score = overlap / (len(content_words) + 1e-5)
                scored_sentences.append((score, s_clean, tag))

        # Sort by relevance score descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if not scored_sentences or scored_sentences[0][0] <= 0:
            # If no direct term overlap, use the first high-salience sentence
            top_sentences = scored_sentences[:2] if scored_sentences else []
        else:
            # Take top 2-3 most relevant unique sentences
            top_sentences = []
            seen = set()
            for score, sent, tag in scored_sentences:
                normalized = sent.lower()[:30]
                if normalized not in seen and score > 0:
                    seen.add(normalized)
                    top_sentences.append((score, sent, tag))
                if len(top_sentences) >= 3:
                    break

        if not top_sentences:
            return "The retrieved context does not contain sufficient details to address the query."

        # Format answer statements with citation tags
        answer_parts = []
        for _, sent, tag in top_sentences:
            sent_trimmed = sent.rstrip(". ")
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
    """Client for OpenAI-compatible APIs (OpenAI, Ollama, vLLM, LMStudio, LocalAI)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy-key")
        self._model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "llama3.2")

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
        """Call standard OpenAI chat completions endpoint."""
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        with httpx.Client(timeout=60.0) as client:
            try:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.ConnectError:
                raise RuntimeError(
                    f"Cannot connect to LLM service at {self.base_url}. If using Ollama, ensure it is running (`ollama serve`), or switch LLM Engine to 'Offline (Deterministic)'."
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403):
                    raise RuntimeError(
                        f"LLM API authorization failed (HTTP {e.response.status_code}) for model '{self._model_name}'. Configure a valid API key or switch to 'Offline (Deterministic)'."
                    )
                raise RuntimeError(f"LLM API error (HTTP {e.response.status_code}): {e.response.text[:200]}")

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream tokens via Server-Sent Events from OpenAI-compatible endpoint."""
        import json

        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        with httpx.Client(timeout=60.0) as client:
            with client.stream(
                "POST", f"{self.base_url}/chat/completions", headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[len("data: ") :].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


class AnthropicEngine(BaseInferenceEngine):
    """Client for Anthropic Claude inference API (LLM Provider Layer: Claude)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._model_name = model_name or os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20241022")

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
            return OfflineGroundedEngine(model_name=f"{self._model_name}-offline").generate(
                prompt=prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
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
                resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["content"][0]["text"]
            except httpx.ConnectError:
                raise RuntimeError("Cannot connect to Anthropic API. Check your network or switch to 'Offline (Deterministic)'.")
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403):
                    raise RuntimeError(f"Anthropic API authorization failed (HTTP {e.response.status_code}). Configure ANTHROPIC_API_KEY or switch to 'Offline (Deterministic)'.")
                raise RuntimeError(f"Anthropic API error (HTTP {e.response.status_code}): {e.response.text[:200]}")

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        full_text = self.generate(
            prompt=prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
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
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        self._model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

    @property
    def model_name(self) -> str:
        return f"gemini/{self._model_name}"

    @property
    def name(self) -> str:
        return "Gemini"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

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

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent?key={self.api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, json=payload)
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

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:streamGenerateContent?key={self.api_key}&alt=sse"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        with httpx.Client(timeout=45.0) as client:
            with client.stream("POST", url, json=payload) as response:
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


class GroqEngine(BaseInferenceEngine):
    """Client for Groq Cloud ultra-fast inference API with OpenAI-compatible SSE streaming."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._model_name = model_name or os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        self.base_url = "https://api.groq.com/openai/v1"

    @property
    def model_name(self) -> str:
        return f"groq/{self._model_name}"

    @property
    def name(self) -> str:
        return "Groq"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if not self.is_configured():
            raise ValueError("GROQ_API_KEY is not configured in .env.")
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        if not self.is_configured():
            raise ValueError("GROQ_API_KEY is not configured in .env.")
        import json

        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with httpx.Client(timeout=45.0) as client:
            with client.stream(
                "POST", f"{self.base_url}/chat/completions", headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[len("data: ") :].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    yield token
                        except Exception:
                            continue


class OpenRouterEngine(BaseInferenceEngine):
    """Client for OpenRouter unified LLM API with OpenAI-compatible SSE streaming."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._model_name = model_name or os.getenv("OPENROUTER_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct")
        self.base_url = "https://openrouter.ai/api/v1"

    @property
    def model_name(self) -> str:
        return f"openrouter/{self._model_name}"

    @property
    def name(self) -> str:
        return "OpenRouter"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if not self.is_configured():
            raise ValueError("OPENROUTER_API_KEY is not configured in .env.")
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Mitigating Context Degradation",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        if not self.is_configured():
            raise ValueError("OPENROUTER_API_KEY is not configured in .env.")
        import json

        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Mitigating Context Degradation",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with httpx.Client(timeout=45.0) as client:
            with client.stream(
                "POST", f"{self.base_url}/chat/completions", headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[len("data: ") :].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    yield token
                        except Exception:
                            continue


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
            return getattr(self._last_used_engine, "name", self._last_used_engine.model_name)
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
            logger.warning("[FallbackEngine] No API keys in .env. Using offline emergency synthesis.")
            self._last_used_engine = self._emergency_offline
            return self._emergency_offline.generate(prompt, system_prompt, temperature, max_tokens)

        errors: list[str] = []
        for engine in configured_engines:
            try:
                logger.info(f"[FallbackEngine] Attempting generation with {engine.name} ({engine.model_name})...")
                answer = engine.generate(prompt, system_prompt, temperature, max_tokens)
                self._last_used_engine = engine
                logger.info(f"[FallbackEngine] Succeeded with {engine.name}.")
                return answer
            except Exception as err:
                logger.warning(f"[FallbackEngine] Provider {engine.name} failed: {err}. Cascading to next...")
                errors.append(f"{engine.name}: {err}")

        err_summary = "; ".join(errors)
        logger.error(f"[FallbackEngine] All providers failed: {err_summary}")
        self._last_used_engine = self._emergency_offline
        fallback_note = f"[Notice: Cloud LLM providers were unavailable ({err_summary}). Running offline fallback.]\n\n"
        return fallback_note + self._emergency_offline.generate(prompt, system_prompt, temperature, max_tokens)

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        configured_engines = [e for e in self.engines if e.is_configured()]

        if not configured_engines:
            logger.warning("[FallbackEngine] No API keys in .env. Using offline emergency synthesis.")
            self._last_used_engine = self._emergency_offline
            yield from self._emergency_offline.generate_stream(prompt, system_prompt, temperature, max_tokens)
            return

        errors: list[str] = []
        stream_started = False

        for engine in configured_engines:
            try:
                logger.info(f"[FallbackEngine] Connecting stream with {engine.name} ({engine.model_name})...")
                token_iterator = engine.generate_stream(prompt, system_prompt, temperature, max_tokens)
                first_token = next(token_iterator, None)
                self._last_used_engine = engine
                stream_started = True

                if first_token is not None:
                    yield first_token
                for token in token_iterator:
                    yield token
                return
            except Exception as err:
                logger.warning(f"[FallbackEngine] Stream with {engine.name} failed: {err}. Cascading...")
                errors.append(f"{engine.name}: {err}")
                if stream_started:
                    raise

        err_summary = "; ".join(errors)
        logger.error(f"[FallbackEngine] All streaming providers failed: {err_summary}")
        self._last_used_engine = self._emergency_offline
        yield f"[Notice: Cloud LLM providers were unavailable ({err_summary}). Running offline fallback.]\n\n"
        yield from self._emergency_offline.generate_stream(prompt, system_prompt, temperature, max_tokens)


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


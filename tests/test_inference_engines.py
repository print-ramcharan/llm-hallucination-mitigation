from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from src.chunking.tokenizer import count_tokens
from src.generation.engine import (
    GeminiEngine,
    GroqEngine,
    OpenAICompatibleEngine,
    OpenRouterEngine,
)


def install_fake_langchain_openai(
    monkeypatch, response_content: str = "answer", invoke_error: Exception | None = None
):
    calls = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls["configuration"] = kwargs

        def invoke(self, messages):
            calls["messages"] = messages
            if invoke_error is not None:
                raise invoke_error
            return SimpleNamespace(content=response_content)

        def stream(self, messages):
            calls["messages"] = messages
            return iter(
                [
                    SimpleNamespace(content="first "),
                    SimpleNamespace(content="second"),
                ]
            )

    module = ModuleType("langchain_openai")
    module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    return calls


def test_openai_compatible_engine_uses_langchain_for_generation(monkeypatch):
    calls = install_fake_langchain_openai(monkeypatch)
    engine = OpenAICompatibleEngine(
        base_url="https://llm.example/v1",
        api_key="test-key",
        model_name="test-model",
    )

    answer = engine.generate("question", "system", temperature=0.25, max_tokens=128)

    assert answer == "answer"
    assert calls["configuration"] == {
        "model": "test-model",
        "base_url": "https://llm.example/v1",
        "api_key": "test-key",
        "temperature": 0.25,
        "max_tokens": 128,
        "timeout": 60.0,
        "default_headers": None,
    }
    assert [message.content for message in calls["messages"]] == ["system", "question"]


def test_openai_compatible_engine_streams_langchain_chunks(monkeypatch):
    calls = install_fake_langchain_openai(monkeypatch)
    engine = OpenAICompatibleEngine(api_key="test-key")

    chunks = list(engine.generate_stream("question", "system"))

    assert chunks == ["first ", "second"]
    assert [message.content for message in calls["messages"]] == ["system", "question"]


def test_provider_adapters_keep_endpoints_and_openrouter_headers(monkeypatch):
    calls = install_fake_langchain_openai(monkeypatch)
    groq = GroqEngine(api_key="groq-key", model_name="groq-model")
    openrouter = OpenRouterEngine(api_key="router-key", model_name="router-model")

    assert groq.base_url == "https://api.groq.com/openai/v1"
    assert groq.model_name == "groq/groq-model"
    assert groq.is_configured()
    assert openrouter.base_url == "https://openrouter.ai/api/v1"
    assert openrouter.model_name == "openrouter/router-model"
    assert openrouter.default_headers == {
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Mitigating Context Degradation",
    }
    assert openrouter.is_configured()
    openrouter.generate("question", "system")
    assert calls["configuration"]["default_headers"] == openrouter.default_headers


def test_retired_provider_model_ids_are_mapped_to_supported_models():
    gemini = GeminiEngine(api_key="gemini-key", model_name="gemini-1.5-flash")
    groq = GroqEngine(api_key="groq-key", model_name="llama-3.3-70b-versatile")

    assert gemini.model_name == "gemini/gemini-3.5-flash-lite"
    assert groq.model_name == "groq/openai/gpt-oss-120b"


def test_gemini_api_key_is_sent_in_header_not_url(monkeypatch):
    calls = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "answer"}]}}]}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, **kwargs):
            calls["url"] = url
            calls.update(kwargs)
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)
    engine = GeminiEngine(api_key="test-key", model_name="gemini-1.5-flash")

    assert engine.generate("question", "system") == "answer"
    assert "test-key" not in calls["url"]
    assert calls["headers"] == {"x-goog-api-key": "test-key"}


def test_openrouter_prompt_respects_model_context_window(monkeypatch):
    calls = install_fake_langchain_openai(monkeypatch)
    engine = OpenRouterEngine(api_key="router-key", model_name="router-model")
    system_prompt = "Use only the supplied evidence."
    prompt = "a!" * 170_000 + "\nUSER QUERY:\nWhat is the answer?"

    engine.generate(prompt, system_prompt, max_tokens=64)

    sent_prompt = calls["messages"][1].content
    assert "USER QUERY:\nWhat is the answer?" in sent_prompt
    assert (
        count_tokens(sent_prompt)
        + count_tokens(system_prompt)
        + 64
        + engine.context_safety_margin_tokens
        <= engine.context_window_tokens
    )


def test_connection_errors_keep_actionable_provider_message(monkeypatch):
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    error = APIConnectionError(message="connection failed", request=request)
    install_fake_langchain_openai(monkeypatch, invoke_error=error)
    engine = OpenAICompatibleEngine(
        base_url="https://llm.example/v1",
        api_key="test-key",
        model_name="test-model",
    )

    with pytest.raises(RuntimeError, match="Cannot connect to LLM service"):
        engine.generate("question", "system")


def test_api_key_requirements_match_existing_provider_behavior(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert OpenAICompatibleEngine(api_key="").is_configured()
    assert not GroqEngine(api_key="").is_configured()
    assert not OpenRouterEngine(api_key="").is_configured()

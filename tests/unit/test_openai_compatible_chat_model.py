"""Tests for the isolated OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from personal_knowledge_agent.domain.errors import (
    LanguageModelError,
    ModelOutputTruncatedError,
)
from personal_knowledge_agent.domain.models import ChatMessage, MessageRole
from personal_knowledge_agent.infrastructure.openai_compatible_chat_model import (
    OpenAICompatibleChatModel,
)


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str | None


@dataclass
class FakeResponse:
    model: str
    choices: list[FakeChoice]


class FakeCompletions:
    def __init__(self, result: FakeResponse | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result: FakeResponse | Exception) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(result))


def messages() -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(MessageRole.SYSTEM, "Return JSON."),
        ChatMessage(MessageRole.USER, "Question and evidence."),
    )


def test_adapter_sends_json_chat_completion_and_maps_output() -> None:
    client = FakeClient(
        FakeResponse("deepseek-v4-flash", [FakeChoice(FakeMessage("{}"), "stop")])
    )
    model = OpenAICompatibleChatModel(
        api_key="secret",
        base_url="https://api.deepseek.com",
        model_id="deepseek-v4-flash",
        client=client,
    )

    output = model.generate(messages())

    assert output.model_id == "deepseek-v4-flash"
    assert output.content == "{}"
    assert output.finish_reason == "stop"
    assert client.chat.completions.calls == [
        {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "Question and evidence."},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
            "max_tokens": 1024,
        }
    ]


def test_real_client_construction_is_lazy_and_configures_sdk_retries() -> None:
    fake_client = FakeClient(
        FakeResponse("provider-model", [FakeChoice(FakeMessage("{}"), "stop")])
    )
    with patch(
        "personal_knowledge_agent.infrastructure.openai_compatible_chat_model.OpenAI",
        return_value=fake_client,
    ) as constructor:
        model = OpenAICompatibleChatModel(
            api_key="secret",
            base_url="https://provider.example/v1",
            model_id="provider-model",
            max_tokens=256,
        )

        constructor.assert_not_called()
        model.generate(messages())

    constructor.assert_called_once_with(
        api_key="secret",
        base_url="https://provider.example/v1",
        max_retries=2,
    )
    assert fake_client.chat.completions.calls[0]["max_tokens"] == 256


@pytest.mark.parametrize(
    ("response", "error_type", "message"),
    [
        (FakeResponse("model", []), LanguageModelError, "no choices"),
        (
            FakeResponse("model", [FakeChoice(FakeMessage(None), "stop")]),
            LanguageModelError,
            "empty content",
        ),
        (
            FakeResponse("model", [FakeChoice(FakeMessage("   "), "stop")]),
            LanguageModelError,
            "empty content",
        ),
        (
            FakeResponse("model", [FakeChoice(FakeMessage("{"), "length")]),
            ModelOutputTruncatedError,
            "truncated",
        ),
        (
            FakeResponse("model", [FakeChoice(FakeMessage("{}"), "content_filter")]),
            LanguageModelError,
            "content_filter",
        ),
    ],
)
def test_adapter_rejects_unusable_provider_responses(
    response: FakeResponse,
    error_type: type[Exception],
    message: str,
) -> None:
    model = OpenAICompatibleChatModel(
        api_key="secret",
        base_url="https://provider.example",
        model_id="model",
        client=FakeClient(response),
    )

    with pytest.raises(error_type, match=message):
        model.generate(messages())


def status_error(error_type: type[Exception], status_code: int) -> Exception:
    request = httpx.Request("POST", "https://provider.example/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": "secret"})
    return error_type("provider leaked api-key-value", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (status_error(AuthenticationError, 401), "authentication"),
        (status_error(PermissionDeniedError, 403), "permission"),
        (status_error(BadRequestError, 400), "rejected the request"),
        (status_error(RateLimitError, 429), "rate limit"),
        (status_error(InternalServerError, 500), "server error"),
        (
            APITimeoutError(
                request=httpx.Request("POST", "https://provider.example")
            ),
            "timed out",
        ),
        (
            APIConnectionError(
                request=httpx.Request("POST", "https://provider.example")
            ),
            "connect",
        ),
        (RuntimeError("api-key-value"), "unexpected"),
    ],
)
def test_adapter_maps_provider_errors_without_leaking_secret(
    error: Exception,
    message: str,
) -> None:
    model = OpenAICompatibleChatModel(
        api_key="api-key-value",
        base_url="https://provider.example",
        model_id="model",
        client=FakeClient(error),
    )

    with pytest.raises(LanguageModelError, match=message) as captured:
        model.generate(messages())

    assert "api-key-value" not in str(captured.value)


@pytest.mark.parametrize(
    ("api_key", "base_url", "model_id", "max_tokens", "message"),
    [
        ("", "https://provider.example", "model", 10, "api_key"),
        ("key", " ", "model", 10, "base_url"),
        ("key", "https://provider.example", "\n", 10, "model_id"),
        ("key", "https://provider.example", "model", 0, "max_tokens"),
    ],
)
def test_adapter_rejects_invalid_configuration(
    api_key: str,
    base_url: str,
    model_id: str,
    max_tokens: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OpenAICompatibleChatModel(
            api_key=api_key,
            base_url=base_url,
            model_id=model_id,
            max_tokens=max_tokens,
            client=FakeClient(FakeResponse("model", [])),
        )

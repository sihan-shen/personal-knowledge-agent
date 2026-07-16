"""OpenAI-compatible Chat Completions adapter for structured text output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from personal_knowledge_agent.domain.errors import (
    LanguageModelError,
    ModelOutputTruncatedError,
)
from personal_knowledge_agent.domain.models import ChatMessage, ModelOutput


class _ProviderMessage(Protocol):
    content: object


class _ProviderChoice(Protocol):
    message: _ProviderMessage
    finish_reason: object


class _ProviderResponse(Protocol):
    model: object
    choices: Sequence[_ProviderChoice]


class _Completions(Protocol):
    def create(self, **kwargs: Any) -> _ProviderResponse: ...


class _Chat(Protocol):
    completions: _Completions


class _Client(Protocol):
    chat: _Chat


class OpenAICompatibleChatModel:
    """Generate one JSON object through an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_id: str,
        max_tokens: int = 1024,
        client: object | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be blank")
        if not base_url.strip():
            raise ValueError("base_url must not be blank")
        if not model_id.strip():
            raise ValueError("model_id must not be blank")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self._api_key = api_key
        self._base_url = base_url
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._client = cast(_Client, client) if client is not None else None

    def generate(self, messages: Sequence[ChatMessage]) -> ModelOutput:
        """Request non-streaming JSON output and map it to project-owned values."""
        provider_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]
        try:
            response = self._get_client().chat.completions.create(
                model=self._model_id,
                messages=provider_messages,
                response_format={"type": "json_object"},
                stream=False,
                max_tokens=self._max_tokens,
            )
        except AuthenticationError as error:
            raise LanguageModelError(
                "Language-model provider authentication failed"
            ) from error
        except PermissionDeniedError as error:
            raise LanguageModelError(
                "Language-model provider denied permission"
            ) from error
        except BadRequestError as error:
            raise LanguageModelError(
                "Language-model provider rejected the request"
            ) from error
        except RateLimitError as error:
            raise LanguageModelError(
                "Language-model provider rate limit was reached"
            ) from error
        except APITimeoutError as error:
            raise LanguageModelError("Language-model request timed out") from error
        except APIConnectionError as error:
            raise LanguageModelError(
                "Could not connect to language-model provider"
            ) from error
        except InternalServerError as error:
            raise LanguageModelError(
                "Language-model provider returned a server error"
            ) from error
        except Exception as error:
            raise LanguageModelError(
                "Language-model provider returned an unexpected error"
            ) from error

        if not response.choices:
            raise LanguageModelError("Language-model response contained no choices")
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        if finish_reason == "length":
            raise ModelOutputTruncatedError(
                "Language-model structured output was truncated"
            )
        if finish_reason != "stop":
            safe_reason = finish_reason if isinstance(finish_reason, str) else "unknown"
            raise LanguageModelError(
                f"Language-model response stopped with reason: {safe_reason}"
            )

        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise LanguageModelError("Language-model response contained empty content")
        if not isinstance(response.model, str) or not response.model.strip():
            raise LanguageModelError("Language-model response omitted the model ID")
        return ModelOutput(response.model, content, finish_reason)

    def _get_client(self) -> _Client:
        if self._client is None:
            self._client = cast(
                _Client,
                OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    max_retries=2,
                ),
            )
        return self._client

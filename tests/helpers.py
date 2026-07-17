"""Deterministic test doubles shared by retrieval tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import fsum, isfinite, sqrt

from personal_knowledge_agent.domain.errors import EmbeddingError
from personal_knowledge_agent.domain.memory import (
    MemoryActivationAction,
    MemoryProposal,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
)
from personal_knowledge_agent.domain.models import ChatMessage, Embedding, ModelOutput


class FakeEmbedder:
    """Return configured normalized vectors without importing a model runtime."""

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        *,
        model_id: str = "test-embedding-model",
        default_vector: Sequence[float] | None = None,
    ) -> None:
        self._vectors = vectors
        self._model_id = model_id
        self._default_vector = default_vector
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        self.document_calls.append(list(texts))
        return [self._embedding_for(text) for text in texts]

    def embed_query(self, query: str) -> Embedding:
        self.query_calls.append(query)
        return self._embedding_for(query)

    def _embedding_for(self, text: str) -> Embedding:
        try:
            values = tuple(float(value) for value in self._vectors[text])
        except KeyError as error:
            if self._default_vector is None:
                raise EmbeddingError(
                    f"No fake embedding configured for: {text}"
                ) from error
            values = tuple(float(value) for value in self._default_vector)

        if not values or not all(isfinite(value) for value in values):
            raise EmbeddingError(f"Invalid fake embedding configured for: {text}")
        magnitude = sqrt(fsum(value * value for value in values))
        if magnitude == 0.0:
            raise EmbeddingError(f"Zero fake embedding configured for: {text}")
        return Embedding(
            self.model_id,
            tuple(value / magnitude for value in values),
        )


class FakeLanguageModel:
    """Return one configured output or failure and record immutable calls."""

    def __init__(self, result: ModelOutput | Exception) -> None:
        self._result = result
        self.calls: list[tuple[ChatMessage, ...]] = []

    def generate(self, messages: Sequence[ChatMessage]) -> ModelOutput:
        self.calls.append(tuple(messages))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class ScriptedLanguageModel:
    """Return a finite sequence of configured outputs or failures."""

    def __init__(self, results: Sequence[ModelOutput | Exception]) -> None:
        self._results = tuple(results)
        self._position = 0
        self.calls: list[tuple[ChatMessage, ...]] = []

    def generate(self, messages: Sequence[ChatMessage]) -> ModelOutput:
        self.calls.append(tuple(messages))
        if self._position >= len(self._results):
            raise AssertionError("No scripted language-model result remains")
        result = self._results[self._position]
        self._position += 1
        if isinstance(result, Exception):
            raise result
        return result


class FakeMemoryStore:
    """Return configured memory values or failures and record protocol calls."""

    def __init__(
        self,
        *,
        create_pending_result: Sequence[MemoryRecord] | Exception = (),
        remember_result: MemoryRecord | Exception | None = None,
        approve_result: MemoryRecord | Exception | None = None,
        reject_result: Exception | None = None,
        forget_result: int | Exception = 0,
        list_memories_result: Sequence[MemoryRecord] | Exception = (),
        search_result: Sequence[MemorySearchResult] | Exception = (),
    ) -> None:
        self._create_pending_result = (
            create_pending_result
            if isinstance(create_pending_result, Exception)
            else tuple(create_pending_result)
        )
        self._remember_result = remember_result
        self._approve_result = approve_result
        self._reject_result = reject_result
        self._forget_result = forget_result
        self._list_memories_result = (
            list_memories_result
            if isinstance(list_memories_result, Exception)
            else tuple(list_memories_result)
        )
        self._search_result = (
            search_result
            if isinstance(search_result, Exception)
            else tuple(search_result)
        )
        self.create_pending_calls: list[tuple[tuple[MemoryProposal, ...], str]] = []
        self.remember_calls: list[
            tuple[
                MemoryType,
                str,
                str,
                MemoryActivationAction,
                str | None,
            ]
        ] = []
        self.approve_calls: list[tuple[str, MemoryActivationAction, str | None]] = []
        self.reject_calls: list[str] = []
        self.forget_calls: list[str] = []
        self.list_memories_calls: list[
            tuple[MemoryStatus | None, MemoryType | None]
        ] = []
        self.search_calls: list[tuple[str, int, MemoryType | None]] = []

    def create_pending(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        source_hash: str,
    ) -> tuple[MemoryRecord, ...]:
        self.create_pending_calls.append((tuple(proposals), source_hash))
        if isinstance(self._create_pending_result, Exception):
            raise self._create_pending_result
        return self._create_pending_result

    def remember(
        self,
        memory_type: MemoryType,
        memory_key: str,
        content: str,
        *,
        action: MemoryActivationAction = MemoryActivationAction.ACTIVATE,
        replace_id: str | None = None,
    ) -> MemoryRecord:
        self.remember_calls.append(
            (memory_type, memory_key, content, action, replace_id)
        )
        if isinstance(self._remember_result, Exception):
            raise self._remember_result
        if self._remember_result is None:
            raise AssertionError("No fake remember result configured")
        return self._remember_result

    def approve(
        self,
        candidate_id: str,
        *,
        action: MemoryActivationAction,
        replace_id: str | None = None,
    ) -> MemoryRecord:
        self.approve_calls.append((candidate_id, action, replace_id))
        if isinstance(self._approve_result, Exception):
            raise self._approve_result
        if self._approve_result is None:
            raise AssertionError("No fake approve result configured")
        return self._approve_result

    def reject(self, candidate_id: str) -> None:
        self.reject_calls.append(candidate_id)
        if self._reject_result is not None:
            raise self._reject_result

    def forget(self, memory_id: str) -> int:
        self.forget_calls.append(memory_id)
        if isinstance(self._forget_result, Exception):
            raise self._forget_result
        return self._forget_result

    def list_memories(
        self,
        *,
        status: MemoryStatus | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        self.list_memories_calls.append((status, memory_type))
        if isinstance(self._list_memories_result, Exception):
            raise self._list_memories_result
        return list(self._list_memories_result)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
    ) -> list[MemorySearchResult]:
        self.search_calls.append((query, limit, memory_type))
        if isinstance(self._search_result, Exception):
            raise self._search_result
        return list(self._search_result)

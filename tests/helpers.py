"""Deterministic test doubles shared by retrieval tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import fsum, isfinite, sqrt

from personal_knowledge_agent.domain.errors import EmbeddingError
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

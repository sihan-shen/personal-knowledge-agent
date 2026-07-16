"""Lazy CPU adapter for multilingual E5 sentence embeddings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from math import fsum, isfinite, sqrt
from typing import Protocol, cast

from personal_knowledge_agent.domain.errors import EmbeddingError
from personal_knowledge_agent.domain.models import Embedding

E5_MODEL_ID = "intfloat/multilingual-e5-small"


class _SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> object: ...


type _ModelFactory = Callable[[str, str], _SentenceEncoder]


def _create_sentence_transformer(model_id: str, device: str) -> _SentenceEncoder:
    module = import_module("sentence_transformers")
    constructor = module.SentenceTransformer
    return cast(_SentenceEncoder, constructor(model_id, device=device))


class E5Embedder:
    """Encode E5 passages and queries without loading the model at import time."""

    def __init__(self, *, model_factory: _ModelFactory | None = None) -> None:
        self._model_factory = model_factory or _create_sentence_transformer
        self._model: _SentenceEncoder | None = None

    @property
    def model_id(self) -> str:
        return E5_MODEL_ID

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        prefixed = [f"passage: {text}" for text in texts]
        return self._embed(prefixed)

    def embed_query(self, query: str) -> Embedding:
        embeddings = self._embed([f"query: {query}"])
        return embeddings[0]

    def _load_model(self) -> _SentenceEncoder:
        if self._model is not None:
            return self._model

        try:
            self._model = self._model_factory(self.model_id, "cpu")
        except Exception as error:
            raise EmbeddingError(
                f"Could not load embedding model {self.model_id}: {error}"
            ) from error
        return self._model

    def _embed(self, inputs: Sequence[str]) -> list[Embedding]:
        model = self._load_model()
        try:
            output = model.encode(
                inputs,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise EmbeddingError(
                f"Could not encode text with embedding model {self.model_id}: {error}"
            ) from error

        rows = self._convert_rows(output)
        if len(rows) != len(inputs):
            raise EmbeddingError(
                f"Embedding model returned {len(rows)} vectors for "
                f"{len(inputs)} inputs; expected one vector per input"
            )

        dimensions = {len(row) for row in rows}
        if len(dimensions) != 1:
            raise EmbeddingError("Embedding vectors must have one common dimension")

        return [self._normalize(row) for row in rows]

    def _convert_rows(self, output: object) -> list[list[float]]:
        to_list = getattr(output, "tolist", None)
        if callable(to_list):
            output = to_list()

        if not isinstance(output, (list, tuple)):
            raise EmbeddingError(
                "Embedding model output must be a two-dimensional array"
            )

        rows: list[list[float]] = []
        for row in output:
            if not isinstance(row, (list, tuple)):
                raise EmbeddingError("Embedding model must return one vector per input")
            try:
                rows.append([float(value) for value in row])
            except (TypeError, ValueError, OverflowError) as error:
                raise EmbeddingError(
                    "Embedding vector values must be real numbers"
                ) from error
        return rows

    def _normalize(self, values: Sequence[float]) -> Embedding:
        if not values:
            raise EmbeddingError("Embedding vector must not be empty")
        if not all(isfinite(value) for value in values):
            raise EmbeddingError("Embedding vector values must be finite")

        magnitude = sqrt(fsum(value * value for value in values))
        if magnitude == 0.0:
            raise EmbeddingError("Embedding model returned a zero vector")

        try:
            return Embedding(
                model_id=self.model_id,
                values=tuple(value / magnitude for value in values),
            )
        except ValueError as error:
            raise EmbeddingError(f"Invalid embedding vector: {error}") from error

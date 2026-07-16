"""Tests for the lazy local E5 embedding adapter."""

from collections.abc import Sequence
from math import nan

import pytest

from personal_knowledge_agent.domain.errors import EmbeddingError
from personal_knowledge_agent.infrastructure.e5_embedder import E5Embedder


class FakeSentenceTransformer:
    def __init__(self, outputs: object) -> None:
        self.outputs = outputs
        self.calls: list[tuple[list[str], bool]] = []

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> object:
        del convert_to_numpy, show_progress_bar
        self.calls.append((list(sentences), normalize_embeddings))
        if isinstance(self.outputs, Exception):
            raise self.outputs
        return self.outputs


class RecordingFactory:
    def __init__(self, model: FakeSentenceTransformer) -> None:
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model_id: str, device: str) -> FakeSentenceTransformer:
        self.calls.append((model_id, device))
        return self.model


def test_model_loading_is_lazy_and_cached() -> None:
    model = FakeSentenceTransformer([[3.0, 4.0]])
    factory = RecordingFactory(model)
    embedder = E5Embedder(model_factory=factory)

    assert factory.calls == []

    first = embedder.embed_query("减少幻觉")
    second = embedder.embed_query("检索知识")

    assert factory.calls == [("intfloat/multilingual-e5-small", "cpu")]
    assert first.values == pytest.approx((0.6, 0.8))
    assert second.values == pytest.approx((0.6, 0.8))


def test_documents_and_queries_use_e5_prefixes() -> None:
    model = FakeSentenceTransformer([[1.0, 0.0], [0.0, 2.0]])
    embedder = E5Embedder(model_factory=RecordingFactory(model))

    documents = embedder.embed_documents(["RAG notes", "代理笔记"])

    assert [embedding.values for embedding in documents] == [
        pytest.approx((1.0, 0.0)),
        pytest.approx((0.0, 1.0)),
    ]
    assert model.calls == [
        (["passage: RAG notes", "passage: 代理笔记"], True),
    ]

    model.outputs = [[1.0, 1.0]]
    query = embedder.embed_query("如何检索？")

    assert query.values == pytest.approx((2**-0.5, 2**-0.5))
    assert model.calls[-1] == (["query: 如何检索？"], True)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ([], "returned 0 vectors"),
        ([1.0, 2.0], "one vector per input"),
        ([[1.0], [2.0]], "returned 2 vectors"),
        ([[0.0, 0.0]], "zero vector"),
        ([[nan, 1.0]], "finite"),
    ],
)
def test_invalid_encoder_outputs_raise_embedding_error(
    output: object,
    message: str,
) -> None:
    embedder = E5Embedder(
        model_factory=RecordingFactory(FakeSentenceTransformer(output))
    )

    with pytest.raises(EmbeddingError, match=message):
        embedder.embed_query("query")


def test_document_vectors_must_have_a_common_dimension() -> None:
    model = FakeSentenceTransformer([[1.0], [1.0, 2.0]])
    embedder = E5Embedder(model_factory=RecordingFactory(model))

    with pytest.raises(EmbeddingError, match="common dimension"):
        embedder.embed_documents(["first", "second"])


def test_model_loading_failures_raise_embedding_error() -> None:
    def fail_to_load(model_id: str, device: str) -> FakeSentenceTransformer:
        del model_id, device
        raise RuntimeError("model unavailable")

    embedder = E5Embedder(model_factory=fail_to_load)

    with pytest.raises(EmbeddingError, match="load.*model unavailable"):
        embedder.embed_query("query")


def test_encoding_failures_raise_embedding_error() -> None:
    model = FakeSentenceTransformer(RuntimeError("encoder failed"))
    embedder = E5Embedder(model_factory=RecordingFactory(model))

    with pytest.raises(EmbeddingError, match="encode.*encoder failed"):
        embedder.embed_documents(["notes"])

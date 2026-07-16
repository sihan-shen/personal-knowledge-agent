"""Tests for shared deterministic retrieval doubles."""

import pytest

from personal_knowledge_agent.domain.errors import EmbeddingError
from tests.helpers import FakeEmbedder


def test_fake_embedder_normalizes_vectors_and_records_boundary_calls() -> None:
    embedder = FakeEmbedder({"passage": (3.0, 4.0), "query": (0.0, 2.0)})

    documents = embedder.embed_documents(["passage"])
    query = embedder.embed_query("query")

    assert documents[0].values == pytest.approx((0.6, 0.8))
    assert query.values == pytest.approx((0.0, 1.0))
    assert embedder.document_calls == [["passage"]]
    assert embedder.query_calls == ["query"]


def test_fake_embedder_rejects_unknown_text_without_model_runtime() -> None:
    embedder = FakeEmbedder({})

    with pytest.raises(EmbeddingError, match="No fake embedding"):
        embedder.embed_query("unknown")


def test_fake_embedder_can_supply_a_default_for_unimportant_test_text() -> None:
    embedder = FakeEmbedder({}, default_vector=(2.0, 0.0))

    assert embedder.embed_documents(["anything"])[0].values == (1.0, 0.0)

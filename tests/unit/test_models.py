"""Tests for retrieval domain models."""

from dataclasses import FrozenInstanceError

import pytest

from personal_knowledge_agent.domain.models import (
    Chunk,
    Document,
    IngestionResult,
    SearchResult,
    make_chunk_id,
    make_document_id,
)


def test_document_id_is_stable_and_source_specific() -> None:
    assert make_document_id("/notes/rag.md") == make_document_id("/notes/rag.md")
    assert make_document_id("/notes/rag.md") != make_document_id("/notes/agent.md")


def test_chunk_id_is_stable_and_position_specific() -> None:
    document_id = make_document_id("/notes/rag.md")

    assert make_chunk_id(document_id, 0) == make_chunk_id(document_id, 0)
    assert make_chunk_id(document_id, 0) != make_chunk_id(document_id, 1)


def test_models_are_frozen() -> None:
    document = Document("doc-id", "/notes/rag.md", "RAG notes")

    with pytest.raises(FrozenInstanceError):
        document.content = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("value", ["", "   "])
def test_document_rejects_blank_identifiers(value: str) -> None:
    with pytest.raises(ValueError, match="document_id"):
        Document(value, "/notes/rag.md", "RAG notes")


def test_chunk_rejects_negative_position() -> None:
    with pytest.raises(ValueError, match="position"):
        Chunk("chunk-id", "doc-id", -1, "text")


@pytest.mark.parametrize("value", ["", "\n\t"])
def test_chunk_rejects_blank_text(value: str) -> None:
    with pytest.raises(ValueError, match="text"):
        Chunk("chunk-id", "doc-id", 0, value)


def test_result_records_keep_retrieval_metadata() -> None:
    chunk = Chunk("chunk-id", "doc-id", 0, "RAG text", ("RAG",))

    result = SearchResult(chunk, "/notes/rag.md", 1.5)
    ingestion = IngestionResult("/notes/rag.md", 1, False)

    assert result.chunk == chunk
    assert result.score == 1.5
    assert ingestion.chunk_count == 1

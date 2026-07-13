"""Tests for ingestion and search application orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from personal_knowledge_agent.application.ingest import IngestService
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.errors import EmptyDocumentError, InvalidQueryError
from personal_knowledge_agent.domain.models import Chunk, Document, SearchResult


class FakeLoader:
    def __init__(self, document: Document, events: list[str]) -> None:
        self._document = document
        self._events = events

    def load(self, path: Path) -> Document:
        self._events.append(f"load:{path}")
        return self._document


class FakeChunker:
    def __init__(self, chunks: list[Chunk], events: list[str]) -> None:
        self._chunks = chunks
        self._events = events

    def split(self, document: Document) -> list[Chunk]:
        self._events.append(f"split:{document.document_id}")
        return self._chunks


class FakeIndex:
    def __init__(
        self,
        *,
        replaced: bool = False,
        results: list[SearchResult] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self._replaced = replaced
        self._results = results or []
        self._events = events if events is not None else []

    def replace_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> bool:
        self._events.append(f"replace:{document.document_id}:{len(chunks)}")
        return self._replaced

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        self._events.append(f"search:{query}:{limit}")
        return self._results


def make_records() -> tuple[Document, Chunk]:
    document = Document("doc-id", "/notes/rag.md", "RAG")
    chunk = Chunk("chunk-id", "doc-id", 0, "RAG", ("Notes",))
    return document, chunk


def test_ingest_orchestrates_loader_chunker_and_index_in_order() -> None:
    events: list[str] = []
    document, chunk = make_records()
    service = IngestService(
        FakeLoader(document, events),
        FakeChunker([chunk], events),
        FakeIndex(replaced=True, events=events),
    )

    result = service.ingest(Path("notes.md"))

    assert events == ["load:notes.md", "split:doc-id", "replace:doc-id:1"]
    assert result.source == document.source
    assert result.chunk_count == 1
    assert result.replaced is True


def test_ingest_rejects_document_that_produces_no_chunks() -> None:
    events: list[str] = []
    document, _ = make_records()
    service = IngestService(
        FakeLoader(document, events),
        FakeChunker([], events),
        FakeIndex(events=events),
    )

    with pytest.raises(EmptyDocumentError, match="chunks"):
        service.ingest(Path("notes.md"))

    assert not any(event.startswith("replace:") for event in events)


def test_search_delegates_query_and_limit() -> None:
    _, chunk = make_records()
    expected = [SearchResult(chunk, "/notes/rag.md", 1.0)]
    events: list[str] = []
    service = SearchService(FakeIndex(results=expected, events=events))

    assert service.search("RAG 检索", limit=3) == expected
    assert events == ["search:RAG 检索:3"]


@pytest.mark.parametrize("limit", [0, 21])
def test_search_rejects_limit_outside_supported_range(limit: int) -> None:
    service = SearchService(FakeIndex())

    with pytest.raises(InvalidQueryError, match="between 1 and 20"):
        service.search("rag", limit=limit)

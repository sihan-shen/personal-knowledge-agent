"""Tests for the graph build orchestration service."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

from personal_knowledge_agent.application.graph_build import GraphBuildService
from personal_knowledge_agent.domain.graph import (
    GraphBuildResult,
    GraphChunkExtraction,
    GraphSourceDocument,
)
from personal_knowledge_agent.domain.models import Chunk


class _FakeSourceReader:
    def __init__(self, result: GraphSourceDocument | None = None) -> None:
        self._result = result
        self.calls: list[Path] = []

    def load(self, source: Path) -> GraphSourceDocument:
        self.calls.append(source)
        if self._result is None:
            raise ValueError("No source configured")
        return self._result


class _FakeExtractor:
    def __init__(self, results: Sequence[GraphChunkExtraction] | None = None) -> None:
        self._results = list(results) if results else []
        self.calls: list[Chunk] = []

    def extract(self, chunk: Chunk) -> GraphChunkExtraction:
        self.calls.append(chunk)
        idx = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[idx] if self._results else GraphChunkExtraction(chunk, (), ())


class _FakeStore:
    def __init__(self) -> None:
        self.replace_calls: list[tuple[GraphSourceDocument, tuple[GraphChunkExtraction, ...]]] = []

    def replace_document(
        self,
        source_document: GraphSourceDocument,
        extractions: Sequence[GraphChunkExtraction],
    ) -> GraphBuildResult:
        self.replace_calls.append((source_document, tuple(extractions)))
        return GraphBuildResult(
            source=str(source_document.source),
            chunks=len(extractions),
            entities=sum(len(e.entities) for e in extractions),
            relations=sum(len(e.relation_evidence) for e in extractions),
            evidence=sum(len(e.relation_evidence) for e in extractions),
        )

    def search_entities(self, query: str, *, limit: int = 10):
        return []

    def get_neighborhood(self, entity_id: str):
        raise NotImplementedError


class TestGraphBuildService:
    def test_loads_source_once(self) -> None:
        chunk = Chunk("c1", "d1", 0, "test")
        doc = GraphSourceDocument("d1", Path("/test.md"), "a" * 64, (chunk,))
        reader = _FakeSourceReader(doc)
        extractor = _FakeExtractor()
        store = _FakeStore()
        service = GraphBuildService(reader, extractor, store)

        result = service.build(Path("/test.md"))
        assert len(reader.calls) == 1
        assert result.chunks == 1

    def test_calls_extractor_for_each_chunk(self) -> None:
        chunks = [
            Chunk("c1", "d1", 0, "first"),
            Chunk("c2", "d1", 1, "second"),
        ]
        doc = GraphSourceDocument("d1", Path("/test.md"), "a" * 64, chunks)
        reader = _FakeSourceReader(doc)
        extractor = _FakeExtractor()
        store = _FakeStore()
        service = GraphBuildService(reader, extractor, store)

        result = service.build(Path("/test.md"))
        assert len(extractor.calls) == 2
        assert result.chunks == 2

    def test_calls_store_once(self) -> None:
        chunk = Chunk("c1", "d1", 0, "test")
        doc = GraphSourceDocument("d1", Path("/test.md"), "a" * 64, (chunk,))
        reader = _FakeSourceReader(doc)
        extractor = _FakeExtractor()
        store = _FakeStore()
        service = GraphBuildService(reader, extractor, store)

        service.build(Path("/test.md"))
        assert len(store.replace_calls) == 1

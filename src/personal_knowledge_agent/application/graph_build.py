"""All-or-nothing per-document graph build orchestration."""

from __future__ import annotations

from pathlib import Path

from personal_knowledge_agent.domain.errors import InvalidGraphError
from personal_knowledge_agent.domain.graph import (
    GraphBuildResult,
    GraphChunkExtraction,
    GraphSourceDocument,
)
from personal_knowledge_agent.domain.protocols import (
    GraphSourceReader,
    KnowledgeGraphExtractor,
    KnowledgeGraphStore,
)


class GraphBuildService:
    """Build a knowledge graph from one indexed source document."""

    def __init__(
        self,
        source_reader: GraphSourceReader,
        extractor: KnowledgeGraphExtractor,
        store: KnowledgeGraphStore,
    ) -> None:
        self._source_reader = source_reader
        self._extractor = extractor
        self._store = store

    def build(self, source: Path) -> GraphBuildResult:
        """Build or rebuild the graph for one source document.

        Loads the indexed source, extracts entities and relations per chunk,
        and atomically replaces the document's graph contribution.
        """
        source_document = self._source_reader.load(source.expanduser().resolve())
        return self.build_document(source_document)

    def build_document(self, source_document: GraphSourceDocument) -> GraphBuildResult:
        """Build from a source document that has already been loaded safely."""

        extractions: list[GraphChunkExtraction] = []
        for chunk in source_document.chunks:
            extraction = self._extractor.extract(chunk)
            if extraction.chunk != chunk:
                raise InvalidGraphError(
                    "Graph extractor returned facts for a different chunk"
                )
            extractions.append(extraction)

        return self._store.replace_document(source_document, tuple(extractions))

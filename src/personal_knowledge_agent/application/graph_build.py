"""All-or-nothing per-document graph build orchestration."""

from __future__ import annotations

from pathlib import Path

from personal_knowledge_agent.domain.graph import GraphBuildResult
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
        source_document = self._source_reader.load(source)

        extractions = []
        for chunk in source_document.chunks:
            extraction = self._extractor.extract(chunk)
            extractions.append(extraction)

        return self._store.replace_document(source_document, extractions)

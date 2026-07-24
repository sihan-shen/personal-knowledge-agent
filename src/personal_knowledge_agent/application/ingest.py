"""Application service for document ingestion."""

from __future__ import annotations

from personal_knowledge_agent.domain.errors import EmptyDocumentError
from personal_knowledge_agent.domain.models import IngestionResult
from personal_knowledge_agent.domain.protocols import (
    Chunker,
    DocumentLoader,
    KnowledgeIndex,
)


class IngestService[Source]:
    """Coordinate loading, chunking, and atomic indexing."""

    def __init__(
        self,
        loader: DocumentLoader[Source],
        chunker: Chunker,
        index: KnowledgeIndex,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._index = index

    def ingest(self, source: Source) -> IngestionResult:
        """Ingest one source and report what changed."""
        document = self._loader.load(source)
        chunks = self._chunker.split(document)
        if not chunks:
            raise EmptyDocumentError(
                f"Document produced no searchable chunks: {document.source}"
            )
        replaced = self._index.replace_document(document, chunks)
        return IngestionResult(document.source, len(chunks), replaced)

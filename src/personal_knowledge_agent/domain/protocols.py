"""Protocols that define replaceable retrieval components."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from personal_knowledge_agent.domain.models import Chunk, Document, SearchResult


class DocumentLoader(Protocol):
    """Load one source path into a document."""

    def load(self, path: Path) -> Document:
        """Load and validate a document."""
        ...


class Chunker(Protocol):
    """Split a document into ordered searchable chunks."""

    def split(self, document: Document) -> list[Chunk]:
        """Create deterministic chunks from a document."""
        ...


class KnowledgeIndex(Protocol):
    """Persist documents and return ranked chunks."""

    def replace_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> bool:
        """Replace a source atomically and report whether it existed."""
        ...

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Return the highest-ranked chunks for a query."""
        ...

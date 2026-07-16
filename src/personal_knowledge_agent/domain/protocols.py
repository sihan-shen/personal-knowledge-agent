"""Protocols that define replaceable retrieval components."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from personal_knowledge_agent.domain.models import (
    Chunk,
    Document,
    Embedding,
    SearchMode,
    SearchResult,
)


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


class EmbeddingModel(Protocol):
    """Encode documents and questions in a shared vector space."""

    @property
    def model_id(self) -> str:
        """Return the stable identifier for vectors this model creates."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        """Encode document passages for persistent retrieval."""
        ...

    def embed_query(self, query: str) -> Embedding:
        """Encode one user question for retrieval."""
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

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        mode: SearchMode = SearchMode.LEXICAL,
    ) -> list[SearchResult]:
        """Return the highest-ranked chunks for a query."""
        ...

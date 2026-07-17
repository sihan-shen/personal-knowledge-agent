"""Protocols that define replaceable retrieval components."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from personal_knowledge_agent.domain.memory import (
    MemoryActivationAction,
    MemoryProposal,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
)
from personal_knowledge_agent.domain.models import (
    ChatMessage,
    Chunk,
    Document,
    Embedding,
    ModelOutput,
    RouteDecision,
    SearchMode,
    SearchResult,
)


class QuestionRouter(Protocol):
    """Choose the safe response path for one user question."""

    def route(self, question: str) -> RouteDecision:
        """Return one validated route decision."""
        ...


class LanguageModel(Protocol):
    """Generate provider-neutral output from an ordered message sequence."""

    def generate(self, messages: Sequence[ChatMessage]) -> ModelOutput:
        """Return one complete model response."""
        ...


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


class MemoryExtractor(Protocol):
    """Extract normalized memory proposals from one user message."""

    def extract(self, message: str) -> tuple[MemoryProposal, ...]:
        """Return validated candidate memories."""
        ...


class MemorySafetyPolicy(Protocol):
    """Check local text against prohibited memory categories."""

    def allows(self, *values: str) -> bool:
        """Return false when any supplied local text matches a prohibited category."""
        ...


class MemoryStore(Protocol):
    """Persist and retrieve long-term memories through their lifecycle."""

    def create_pending(
        self, proposals: Sequence[MemoryProposal], *, source_hash: str
    ) -> tuple[MemoryRecord, ...]:
        """Persist automatic proposals as pending candidates."""
        ...

    def remember(
        self,
        memory_type: MemoryType,
        memory_key: str,
        content: str,
        *,
        action: MemoryActivationAction = MemoryActivationAction.ACTIVATE,
        replace_id: str | None = None,
    ) -> MemoryRecord:
        """Create one explicit active memory."""
        ...

    def approve(
        self,
        candidate_id: str,
        *,
        action: MemoryActivationAction,
        replace_id: str | None = None,
    ) -> MemoryRecord:
        """Activate one pending memory, applying conflict behavior atomically."""
        ...

    def reject(self, candidate_id: str) -> None:
        """Delete one pending candidate and retain only its fingerprint."""
        ...

    def forget(self, memory_id: str) -> int:
        """Delete one memory and report the number of records removed."""
        ...

    def list_memories(
        self,
        *,
        status: MemoryStatus | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        """List memories matching optional lifecycle filters."""
        ...

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
    ) -> list[MemorySearchResult]:
        """Return ranked active memories matching a query."""
        ...

"""Immutable domain records used by retrieval workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite


def make_document_id(source: str) -> str:
    """Create a stable identifier from a normalized document source."""
    return sha256(source.encode("utf-8")).hexdigest()


def make_chunk_id(document_id: str, position: int) -> str:
    """Create a stable identifier for a chunk position within a document."""
    value = f"{document_id}:{position}"
    return sha256(value.encode("utf-8")).hexdigest()


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


class SearchMode(StrEnum):
    """Supported strategies for ranking knowledge chunks."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class MessageRole(StrEnum):
    """Provider-neutral roles for language-model messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class AnswerStatus(StrEnum):
    """Whether an answer is supported by retrieved evidence."""

    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One provider-neutral language-model message."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        _require_text(self.content, "content")


@dataclass(frozen=True, slots=True)
class Document:
    """A loaded source document before chunking."""

    document_id: str
    source: str
    content: str

    def __post_init__(self) -> None:
        _require_text(self.document_id, "document_id")
        _require_text(self.source, "source")


@dataclass(frozen=True, slots=True)
class Chunk:
    """An ordered, searchable portion of a document."""

    chunk_id: str
    document_id: str
    position: int
    text: str
    heading_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "chunk_id")
        _require_text(self.document_id, "document_id")
        if self.position < 0:
            raise ValueError("position must not be negative")
        _require_text(self.text, "text")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A ranked chunk returned by a knowledge index."""

    chunk: Chunk
    source: str
    score: float


@dataclass(frozen=True, slots=True)
class Evidence:
    """A retrieved chunk with a request-local citation label."""

    citation_id: str
    chunk_id: str
    source: str
    heading_path: tuple[str, ...]
    text: str
    retrieval_score: float

    def __post_init__(self) -> None:
        _require_text(self.citation_id, "citation_id")
        _require_text(self.chunk_id, "chunk_id")
        _require_text(self.source, "source")
        _require_text(self.text, "text")
        if not isfinite(self.retrieval_score):
            raise ValueError("retrieval_score must be finite")


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Provider output translated into project-owned values."""

    model_id: str
    content: str
    finish_reason: str

    def __post_init__(self) -> None:
        _require_text(self.model_id, "model_id")
        _require_text(self.content, "content")
        _require_text(self.finish_reason, "finish_reason")


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """A structured answer and every evidence chunk supplied to produce it."""

    status: AnswerStatus
    answer: str
    citation_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    model_id: str | None

    def __post_init__(self) -> None:
        _require_text(self.answer, "answer")


@dataclass(frozen=True, slots=True)
class Embedding:
    """A finite vector produced by one named embedding model."""

    model_id: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_text(self.model_id, "model_id")
        if not self.values:
            raise ValueError("values must not be empty")
        if not all(isfinite(value) for value in self.values):
            raise ValueError("values must be finite")


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary of a completed ingestion operation."""

    source: str
    chunk_count: int
    replaced: bool

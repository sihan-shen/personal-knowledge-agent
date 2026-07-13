"""Immutable domain records used by retrieval workflows."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


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
class IngestionResult:
    """Summary of a completed ingestion operation."""

    source: str
    chunk_count: int
    replaced: bool

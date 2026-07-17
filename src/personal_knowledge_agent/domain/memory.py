"""Provider-neutral contracts for confirmation-gated long-term memory."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from uuid import uuid4

MAX_MEMORY_CONTENT_LENGTH = 500
MAX_MEMORY_EXTRACTION_INPUT_LENGTH = 4_000
MAX_MEMORY_CANDIDATES = 3

_MEMORY_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_MEMORY_ID_PATTERN = re.compile(r"mem_[0-9a-f]{32}")
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


class MemoryType(StrEnum):
    """Supported categories of durable user memory."""

    PREFERENCE = "preference"
    GOAL = "goal"
    PROJECT = "project"
    LEARNING_PROGRESS = "learning_progress"


class MemoryStatus(StrEnum):
    """Lifecycle states persisted for a memory record."""

    PENDING = "pending"
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class MemoryActivationAction(StrEnum):
    """User-selected behavior when activating a memory."""

    ACTIVATE = "activate"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"


class MemoryExtractionSkipReason(StrEnum):
    """Safe reasons automatic extraction did not create candidates."""

    PREFILTER_MISS = "prefilter_miss"
    INPUT_TOO_LONG = "input_too_long"
    SENSITIVE_INPUT = "sensitive_input"
    NO_NEW_CANDIDATES = "no_new_candidates"
    PROVIDER_ERROR = "provider_error"
    INVALID_OUTPUT = "invalid_output"
    UNSAFE_OUTPUT = "unsafe_output"
    EMBEDDING_ERROR = "embedding_error"
    STORAGE_ERROR = "storage_error"


def _normalize_text(value: str, field_name: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def normalize_memory_content(value: str) -> str:
    """Normalize user-visible memory content and enforce its size limit."""
    normalized = _normalize_text(value, "content")
    if len(normalized) > MAX_MEMORY_CONTENT_LENGTH:
        raise ValueError(
            f"content must not exceed {MAX_MEMORY_CONTENT_LENGTH} code points"
        )
    return normalized


def _validate_memory_key(value: str) -> None:
    if _MEMORY_KEY_PATTERN.fullmatch(value) is None:
        raise ValueError("memory_key must be a valid lower-case memory slot")


def _require_memory_id(value: str, field_name: str) -> None:
    if _MEMORY_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must use the mem_ UUID-hex format")


def _require_hash(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def _comparison_value(value: str, field_name: str) -> str:
    return _normalize_text(value, field_name).casefold()


def memory_content_hash(content: str) -> str:
    """Hash normalized memory content for duplicate detection."""
    comparison = normalize_memory_content(content).casefold()
    return sha256(comparison.encode("utf-8")).hexdigest()


def memory_source_hash(message: str) -> str:
    """Hash a normalized source message without retaining the message itself."""
    comparison = _comparison_value(message, "message")
    return sha256(comparison.encode("utf-8")).hexdigest()


def memory_proposal_hash(memory_type: MemoryType, memory_key: str, content: str) -> str:
    """Hash one normalized type, slot, and content proposal."""
    if not isinstance(memory_type, MemoryType):
        raise ValueError("memory_type must be a MemoryType")
    _validate_memory_key(memory_key)
    comparison = normalize_memory_content(content).casefold()
    value = "\0".join((memory_type.value, memory_key, comparison))
    return sha256(value.encode("utf-8")).hexdigest()


def make_memory_id() -> str:
    """Create a random project-owned memory identifier."""
    return f"mem_{uuid4().hex}"


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    """One normalized memory candidate before persistence."""

    memory_type: MemoryType
    memory_key: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.memory_type, MemoryType):
            raise ValueError("memory_type must be a MemoryType")
        _validate_memory_key(self.memory_key)
        object.__setattr__(self, "content", normalize_memory_content(self.content))


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One immutable memory and its lifecycle metadata."""

    memory_id: str
    memory_type: MemoryType
    memory_key: str
    content: str
    content_hash: str
    status: MemoryStatus
    supersedes_id: str | None
    source_hash: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_memory_id(self.memory_id, "memory_id")
        if not isinstance(self.memory_type, MemoryType):
            raise ValueError("memory_type must be a MemoryType")
        _validate_memory_key(self.memory_key)
        normalized_content = normalize_memory_content(self.content)
        if self.content != normalized_content:
            raise ValueError("content must be normalized")
        _require_hash(self.content_hash, "content_hash")
        if self.content_hash != memory_content_hash(self.content):
            raise ValueError("content_hash must match content")
        if not isinstance(self.status, MemoryStatus):
            raise ValueError("status must be a MemoryStatus")
        if self.supersedes_id is not None:
            _require_memory_id(self.supersedes_id, "supersedes_id")
        if self.source_hash is not None:
            _require_hash(self.source_hash, "source_hash")
        _require_utc(self.created_at, "created_at")
        _require_utc(self.updated_at, "updated_at")
        if self.created_at > self.updated_at:
            raise ValueError("updated_at must not precede created_at")


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """One memory and its finite retrieval score."""

    memory: MemoryRecord
    score: float

    def __post_init__(self) -> None:
        if not isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True, slots=True)
class MemoryExtractionReport:
    """Created candidates or the reason extraction safely skipped them."""

    candidates: tuple[MemoryRecord, ...]
    skipped_reason: MemoryExtractionSkipReason | None

    def __post_init__(self) -> None:
        if self.candidates and self.skipped_reason is not None:
            raise ValueError("candidates cannot be combined with a skip reason")
        if not self.candidates and self.skipped_reason is None:
            raise ValueError("report requires exactly one of candidates or skip reason")

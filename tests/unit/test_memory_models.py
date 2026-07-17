"""Tests for provider-neutral long-term memory domain contracts."""

import re
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone

import pytest

from personal_knowledge_agent.domain.memory import (
    MemoryActivationAction,
    MemoryExtractionReport,
    MemoryExtractionSkipReason,
    MemoryProposal,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
    make_memory_id,
    memory_content_hash,
    memory_proposal_hash,
    memory_source_hash,
)

_NOW = datetime(2026, 7, 17, tzinfo=UTC)


def make_record(
    *,
    created_at: datetime = _NOW,
    updated_at: datetime = _NOW,
) -> MemoryRecord:
    content = "Build an agent"
    return MemoryRecord(
        memory_id="mem_00000000000000000000000000000001",
        memory_type=MemoryType.GOAL,
        memory_key="goal.agent",
        content=content,
        content_hash=memory_content_hash(content),
        status=MemoryStatus.PENDING,
        supersedes_id=None,
        source_hash=memory_source_hash("My goal is to build an agent"),
        created_at=created_at,
        updated_at=updated_at,
    )


def test_memory_enums_have_exact_public_values() -> None:
    assert [value.value for value in MemoryType] == [
        "preference",
        "goal",
        "project",
        "learning_progress",
    ]
    assert [value.value for value in MemoryStatus] == [
        "pending",
        "active",
        "superseded",
    ]
    assert [value.value for value in MemoryActivationAction] == [
        "activate",
        "replace",
        "keep_both",
    ]
    assert [value.value for value in MemoryExtractionSkipReason] == [
        "prefilter_miss",
        "input_too_long",
        "sensitive_input",
        "no_new_candidates",
        "provider_error",
        "invalid_output",
        "unsafe_output",
        "embedding_error",
        "storage_error",
    ]


def test_proposal_normalizes_nfkc_and_whitespace() -> None:
    proposal = MemoryProposal(
        MemoryType.GOAL,
        "goal.ai_engineer",
        "  Build\u3000ＡＩ   projects  ",
    )
    assert proposal.content == "Build AI projects"


@pytest.mark.parametrize("content", ["  \t\n", "a" * 501])
def test_proposal_rejects_invalid_content(content: str) -> None:
    with pytest.raises(ValueError, match="content"):
        MemoryProposal(MemoryType.GOAL, "goal.agent", content)


@pytest.mark.parametrize("key", ["Goal.ai", "1goal", "goal/a", "a" * 65])
def test_proposal_rejects_invalid_key(key: str) -> None:
    with pytest.raises(ValueError, match="memory_key"):
        MemoryProposal(MemoryType.GOAL, key, "Build an agent")


def test_hashes_are_normalized_but_slot_specific() -> None:
    assert memory_content_hash(" RAG  Notes ") == memory_content_hash("rag notes")
    assert memory_source_hash(" My\u3000GOAL ") == memory_source_hash("my goal")
    assert memory_proposal_hash(
        MemoryType.GOAL, "goal.rag", "RAG Notes"
    ) != memory_proposal_hash(
        MemoryType.PROJECT, "project.rag", "RAG Notes"
    )


def test_hashes_are_lowercase_sha256_hex() -> None:
    hashes = (
        memory_content_hash("RAG Notes"),
        memory_source_hash("My goal is RAG"),
        memory_proposal_hash(MemoryType.GOAL, "goal.rag", "RAG Notes"),
    )
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)


def test_memory_id_uses_prefixed_uuid_hex_shape() -> None:
    assert re.fullmatch(r"mem_[0-9a-f]{32}", make_memory_id())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("memory_id", "00000000000000000000000000000001"),
        ("memory_id", "mem_NOT-A-UUID"),
        ("content_hash", "A" * 64),
        ("content_hash", "0" * 63),
        ("source_hash", "not-a-hash"),
        ("supersedes_id", "mem_not-a-uuid"),
    ],
)
def test_memory_record_rejects_invalid_identifiers_and_hashes(
    field: str, value: str
) -> None:
    values: dict[str, object] = {
        "memory_id": "mem_00000000000000000000000000000001",
        "memory_type": MemoryType.GOAL,
        "memory_key": "goal.agent",
        "content": "Build an agent",
        "content_hash": memory_content_hash("Build an agent"),
        "status": MemoryStatus.PENDING,
        "supersedes_id": None,
        "source_hash": memory_source_hash("My goal is to build an agent"),
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        MemoryRecord(**values)  # type: ignore[arg-type]


def test_memory_record_requires_utc_timestamps_and_finite_score() -> None:
    record = make_record(created_at=_NOW, updated_at=_NOW)
    with pytest.raises(ValueError, match="finite"):
        MemorySearchResult(record, float("nan"))


@pytest.mark.parametrize(
    ("created_at", "updated_at"),
    [
        (_NOW.replace(tzinfo=None), _NOW),
        (_NOW, _NOW.astimezone(timezone(timedelta(hours=8)))),
        (_NOW + timedelta(seconds=1), _NOW),
    ],
)
def test_memory_record_requires_ordered_utc_timestamps(
    created_at: datetime, updated_at: datetime
) -> None:
    with pytest.raises(ValueError, match="UTC|updated_at"):
        make_record(created_at=created_at, updated_at=updated_at)


def test_report_requires_candidates_or_skip_reason() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MemoryExtractionReport((), None)


def test_report_cannot_mix_candidates_and_skip_reason() -> None:
    with pytest.raises(ValueError, match="skip reason"):
        MemoryExtractionReport(
            (make_record(),), MemoryExtractionSkipReason.PREFILTER_MISS
        )


@pytest.mark.parametrize(
    ("value", "field", "replacement"),
    [
        (
            MemoryProposal(MemoryType.GOAL, "goal.agent", "Build an agent"),
            "content",
            "changed",
        ),
        (make_record(), "content", "changed"),
        (MemorySearchResult(make_record(), 1.0), "score", 2.0),
        (MemoryExtractionReport((make_record(),), None), "candidates", ()),
    ],
)
def test_memory_records_are_immutable(
    value: object, field: str, replacement: object
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(value, field, replacement)

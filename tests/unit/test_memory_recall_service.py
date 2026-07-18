"""Tests for safe slot-atomic memory recall orchestration."""

from datetime import UTC, datetime, timedelta

import pytest

from personal_knowledge_agent.application.memory_recall import (
    MEMORY_RECALL_CANDIDATE_LIMIT,
    MemoryRecallService,
)
from personal_knowledge_agent.domain.errors import (
    EmbeddingError,
    InvalidMemoryError,
    InvalidQueryError,
    MemoryStorageError,
)
from personal_knowledge_agent.domain.memory import (
    MemoryRecallWarning,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
    memory_content_hash,
)
from tests.helpers import FakeMemoryStore

_NOW = datetime(2026, 7, 18, tzinfo=UTC)


def record(
    number: int,
    key: str,
    content: str,
    *,
    memory_type: MemoryType = MemoryType.GOAL,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> MemoryRecord:
    created = _NOW + timedelta(seconds=number)
    return MemoryRecord(
        f"mem_{number:032x}",
        memory_type,
        key,
        content,
        memory_content_hash(content),
        status,
        None,
        None,
        created,
        created,
    )


def result(memory: MemoryRecord, score: float = 0.02) -> MemorySearchResult:
    return MemorySearchResult(memory, score, 0.9)


class RecordingRelevancePolicy:
    def __init__(self, allowed_ids: set[str]) -> None:
        self.allowed_ids = allowed_ids
        self.calls: list[tuple[str, str]] = []

    def allows(self, query: str, candidate: MemorySearchResult) -> bool:
        self.calls.append((query, candidate.memory.memory_id))
        return candidate.memory.memory_id in self.allowed_ids


class RecordingSafetyPolicy:
    def __init__(self, *, unsafe_content: str | None = None) -> None:
        self.unsafe_content = unsafe_content
        self.calls: list[tuple[str, ...]] = []

    def allows(self, *values: str) -> bool:
        self.calls.append(values)
        return self.unsafe_content not in values


def service_for(
    candidates: list[MemorySearchResult] | Exception,
    active: list[MemoryRecord] | Exception,
    *,
    allowed_ids: set[str],
    unsafe_content: str | None = None,
) -> tuple[
    MemoryRecallService,
    FakeMemoryStore,
    RecordingRelevancePolicy,
    RecordingSafetyPolicy,
]:
    store = FakeMemoryStore(search_result=candidates, list_memories_result=active)
    relevance = RecordingRelevancePolicy(allowed_ids)
    safety = RecordingSafetyPolicy(unsafe_content=unsafe_content)
    return (
        MemoryRecallService(store, relevance, safety),
        store,
        relevance,
        safety,
    )


def test_recall_uses_bounded_candidates_and_labels_eligible_slots() -> None:
    first = record(1, "goal.agent", "Build an agent")
    irrelevant = record(2, "preference.theme", "Prefer dark mode")
    service, store, relevance, safety = service_for(
        [result(first), result(irrelevant)],
        [irrelevant, first],
        allowed_ids={first.memory_id},
    )

    report = service.recall("agent goal")

    assert [(item.citation_id, item.memory) for item in report.memories] == [
        ("M1", first)
    ]
    assert report.warnings == ()
    assert store.search_calls == [("agent goal", MEMORY_RECALL_CANDIDATE_LIMIT, None)]
    assert store.list_memories_calls == [(MemoryStatus.ACTIVE, None)]
    assert relevance.calls == [
        ("agent goal", first.memory_id),
        ("agent goal", irrelevant.memory_id),
    ]
    assert safety.calls == [("goal", "goal.agent", "Build an agent")]


def test_recall_includes_complete_keep_both_slot_in_stable_order() -> None:
    older = record(1, "goal.agent", "Build an agent")
    newer = record(2, "goal.agent", "Ship an agent")
    service, _, _, _ = service_for(
        [result(newer)],
        [newer, older],
        allowed_ids={newer.memory_id},
    )

    report = service.recall("agent goal")

    assert [item.memory for item in report.memories] == [older, newer]
    assert [item.citation_id for item in report.memories] == ["M1", "M2"]


def test_recall_skips_oversized_slot_and_warns_once() -> None:
    peers = [record(index, "goal.agent", f"Goal {index}") for index in range(1, 5)]
    service, _, _, _ = service_for(
        [result(peers[0]), result(peers[1])],
        peers,
        allowed_ids={item.memory_id for item in peers},
    )

    report = service.recall("agent goal")

    assert report.memories == ()
    assert report.warnings == (MemoryRecallWarning.AMBIGUOUS_MEMORY_SLOT,)


def test_recall_never_splits_slot_to_fill_remaining_budget() -> None:
    first = record(1, "goal.first", "First goal")
    peer_one = record(2, "goal.agent", "Build an agent")
    peer_two = record(3, "goal.agent", "Ship an agent")
    service, _, _, _ = service_for(
        [result(first), result(peer_one)],
        [first, peer_one, peer_two],
        allowed_ids={first.memory_id, peer_one.memory_id},
    )

    report = service.recall("goals", limit=2)

    assert [item.memory for item in report.memories] == [first]
    assert report.warnings == ()


def test_unsafe_peer_skips_complete_slot_but_keeps_independent_slot() -> None:
    unsafe = record(1, "goal.agent", "unsafe value")
    safe_peer = record(2, "goal.agent", "Build an agent")
    independent = record(3, "preference.style", "Be concise")
    service, _, _, _ = service_for(
        [result(safe_peer), result(independent)],
        [unsafe, safe_peer, independent],
        allowed_ids={safe_peer.memory_id, independent.memory_id},
        unsafe_content="unsafe value",
    )

    report = service.recall("agent style")

    assert [item.memory for item in report.memories] == [independent]
    assert report.warnings == (MemoryRecallWarning.UNSAFE_MEMORY,)


@pytest.mark.parametrize(
    ("error", "warning"),
    [
        (
            EmbeddingError("private embedding details"),
            MemoryRecallWarning.EMBEDDING_ERROR,
        ),
        (MemoryStorageError("/private/memory.db"), MemoryRecallWarning.STORAGE_ERROR),
    ],
)
def test_recall_maps_search_failures_to_safe_empty_report(
    error: Exception,
    warning: MemoryRecallWarning,
) -> None:
    service, store, _, _ = service_for(error, [], allowed_ids=set())

    report = service.recall("agent goal")

    assert report.memories == ()
    assert report.warnings == (warning,)
    assert store.list_memories_calls == []


def test_recall_treats_unsearchable_query_as_normal_empty_result() -> None:
    service, _, _, _ = service_for(
        InvalidMemoryError("no searchable terms"),
        [],
        allowed_ids=set(),
    )

    report = service.recall("🤖")

    assert report.memories == ()
    assert report.warnings == ()


def test_recall_discards_candidates_when_active_snapshot_fails() -> None:
    candidate = record(1, "goal.agent", "Build an agent")
    service, _, _, _ = service_for(
        [result(candidate)],
        MemoryStorageError("private path"),
        allowed_ids={candidate.memory_id},
    )

    report = service.recall("agent")

    assert report.memories == ()
    assert report.warnings == (MemoryRecallWarning.STORAGE_ERROR,)


def test_empty_candidates_skip_active_snapshot() -> None:
    service, store, _, _ = service_for([], [], allowed_ids=set())

    assert service.recall("agent").memories == ()
    assert store.list_memories_calls == []


@pytest.mark.parametrize(("query", "limit"), [("", 3), ("agent", 0), ("agent", 4)])
def test_recall_rejects_invalid_options_before_dependencies(
    query: str,
    limit: int,
) -> None:
    service, store, _, _ = service_for([], [], allowed_ids=set())

    with pytest.raises(InvalidQueryError):
        service.recall(query, limit=limit)

    assert store.search_calls == []

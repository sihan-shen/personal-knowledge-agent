"""Tests for explicit long-term memory administration."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from personal_knowledge_agent.application.memory_admin import MemoryAdminService
from personal_knowledge_agent.domain.errors import (
    InvalidMemoryError,
    UnsafeMemoryError,
)
from personal_knowledge_agent.domain.memory import (
    MemoryActivationAction,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
    memory_content_hash,
)
from tests.helpers import FakeMemoryStore

_NOW = datetime(2026, 7, 17, tzinfo=UTC)
_MEMORY_ID = "mem_00000000000000000000000000000001"
_CANDIDATE_ID = "mem_00000000000000000000000000000002"
_REPLACE_ID = "mem_00000000000000000000000000000003"


def _record(
    *,
    memory_id: str = _MEMORY_ID,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> MemoryRecord:
    content = "Build an agent"
    return MemoryRecord(
        memory_id=memory_id,
        memory_type=MemoryType.GOAL,
        memory_key="goal.agent",
        content=content,
        content_hash=memory_content_hash(content),
        status=status,
        supersedes_id=None,
        source_hash=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


class RecordingSafetyPolicy:
    def __init__(self, result: bool) -> None:
        self._result = result
        self.calls: list[tuple[str, ...]] = []

    def allows(self, *values: str) -> bool:
        self.calls.append(values)
        return self._result


@pytest.mark.parametrize(
    ("action", "replace_id"),
    [
        (MemoryActivationAction.ACTIVATE, None),
        (MemoryActivationAction.REPLACE, _REPLACE_ID),
        (MemoryActivationAction.KEEP_BOTH, None),
    ],
)
def test_remember_validates_safety_checks_and_delegates_explicit_write(
    action: MemoryActivationAction,
    replace_id: str | None,
) -> None:
    expected = _record()
    store = FakeMemoryStore(remember_result=expected)
    safety = RecordingSafetyPolicy(True)
    service = MemoryAdminService(store, safety)

    result = service.remember(
        MemoryType.GOAL,
        "goal.agent",
        "  Build   an agent  ",
        action=action,
        replace_id=replace_id,
    )

    assert result is expected
    assert safety.calls == [("goal.agent", "Build an agent")]
    assert store.remember_calls == [
        (MemoryType.GOAL, "goal.agent", "Build an agent", action, replace_id)
    ]


@pytest.mark.parametrize(
    ("memory_type", "memory_key", "content", "message"),
    [
        ("goal", "goal.agent", "Build an agent", "memory_type"),
        (MemoryType.GOAL, "Goal Agent", "Build an agent", "memory_key"),
        (MemoryType.GOAL, "goal.agent", " \t\n", "content"),
    ],
)
def test_remember_maps_domain_validation_errors_before_dependencies(
    memory_type: MemoryType,
    memory_key: str,
    content: str,
    message: str,
) -> None:
    store = FakeMemoryStore(remember_result=_record())
    safety = RecordingSafetyPolicy(True)
    service = MemoryAdminService(store, safety)

    with pytest.raises(InvalidMemoryError, match=message):
        service.remember(memory_type, memory_key, content)

    assert safety.calls == []
    assert store.remember_calls == []


def test_remember_rejects_unsafe_validated_values_before_store() -> None:
    store = FakeMemoryStore(remember_result=_record())
    safety = RecordingSafetyPolicy(False)
    service = MemoryAdminService(store, safety)

    with pytest.raises(UnsafeMemoryError):
        service.remember(MemoryType.GOAL, "goal.agent", "Build an agent")

    assert safety.calls == [("goal.agent", "Build an agent")]
    assert store.remember_calls == []


@pytest.mark.parametrize(
    ("action", "replace_id"),
    [
        ("activate", None),
        (MemoryActivationAction.ACTIVATE, _REPLACE_ID),
        (MemoryActivationAction.REPLACE, None),
        (MemoryActivationAction.REPLACE, " \t"),
        (MemoryActivationAction.KEEP_BOTH, _REPLACE_ID),
    ],
)
def test_remember_rejects_invalid_action_replace_combinations_before_dependencies(
    action: MemoryActivationAction,
    replace_id: str | None,
) -> None:
    store = FakeMemoryStore(remember_result=_record())
    safety = RecordingSafetyPolicy(True)
    service = MemoryAdminService(store, safety)

    with pytest.raises(InvalidMemoryError):
        service.remember(
            MemoryType.GOAL,
            "goal.agent",
            "Build an agent",
            action=action,
            replace_id=replace_id,
        )

    assert safety.calls == []
    assert store.remember_calls == []


def test_list_returns_store_records_with_filters_unchanged() -> None:
    expected = [_record()]
    store = FakeMemoryStore(list_memories_result=expected)
    service = MemoryAdminService(store, RecordingSafetyPolicy(True))

    result = service.list(
        status=MemoryStatus.ACTIVE,
        memory_type=MemoryType.GOAL,
    )

    assert result == expected
    assert store.list_memories_calls == [(MemoryStatus.ACTIVE, MemoryType.GOAL)]


def test_search_returns_store_results_and_options_unchanged() -> None:
    expected = [MemorySearchResult(_record(), 0.75)]
    store = FakeMemoryStore(search_result=expected)
    service = MemoryAdminService(store, RecordingSafetyPolicy(True))

    result = service.search("agent goal", limit=7, memory_type=MemoryType.GOAL)

    assert result == expected
    assert store.search_calls == [("agent goal", 7, MemoryType.GOAL)]


@pytest.mark.parametrize(
    ("query", "limit"),
    [
        (" \t\n", 5),
        ("agent", 0),
        ("agent", 21),
    ],
)
def test_search_rejects_invalid_query_or_limit_before_store(
    query: str,
    limit: int,
) -> None:
    store = FakeMemoryStore(search_result=[])
    service = MemoryAdminService(store, RecordingSafetyPolicy(True))

    with pytest.raises(InvalidMemoryError):
        service.search(query, limit=limit)

    assert store.search_calls == []


@pytest.mark.parametrize(
    ("action", "replace_id"),
    [
        (MemoryActivationAction.ACTIVATE, None),
        (MemoryActivationAction.REPLACE, _REPLACE_ID),
        (MemoryActivationAction.KEEP_BOTH, None),
    ],
)
def test_approve_returns_store_record_and_options_unchanged(
    action: MemoryActivationAction,
    replace_id: str | None,
) -> None:
    expected = _record(memory_id=_CANDIDATE_ID)
    store = FakeMemoryStore(approve_result=expected)
    service = MemoryAdminService(store, RecordingSafetyPolicy(True))

    result = service.approve(
        _CANDIDATE_ID,
        action=action,
        replace_id=replace_id,
    )

    assert result is expected
    assert store.approve_calls == [(_CANDIDATE_ID, action, replace_id)]


@pytest.mark.parametrize(
    ("candidate_id", "action", "replace_id"),
    [
        (_CANDIDATE_ID, "activate", None),
        (" \t", MemoryActivationAction.ACTIVATE, None),
        (_CANDIDATE_ID, MemoryActivationAction.ACTIVATE, _REPLACE_ID),
        (_CANDIDATE_ID, MemoryActivationAction.REPLACE, None),
        (_CANDIDATE_ID, MemoryActivationAction.REPLACE, " \t"),
        (_CANDIDATE_ID, MemoryActivationAction.KEEP_BOTH, _REPLACE_ID),
    ],
)
def test_approve_rejects_blank_ids_and_invalid_action_combinations_before_store(
    candidate_id: str,
    action: MemoryActivationAction,
    replace_id: str | None,
) -> None:
    store = FakeMemoryStore(approve_result=_record())
    service = MemoryAdminService(store, RecordingSafetyPolicy(True))

    with pytest.raises(InvalidMemoryError):
        service.approve(candidate_id, action=action, replace_id=replace_id)

    assert store.approve_calls == []


def test_reject_returns_none_and_delegates_candidate_id_unchanged() -> None:
    store = FakeMemoryStore()
    service = MemoryAdminService(store, RecordingSafetyPolicy(True))

    result = service.reject(_CANDIDATE_ID)

    assert result is None
    assert store.reject_calls == [_CANDIDATE_ID]


def test_reject_rejects_blank_candidate_id_before_store() -> None:
    store = FakeMemoryStore()
    service = MemoryAdminService(store, RecordingSafetyPolicy(True))

    with pytest.raises(InvalidMemoryError):
        service.reject(" \t\n")

    assert store.reject_calls == []


def test_forget_returns_exact_deleted_count_and_delegates_id_unchanged() -> None:
    store = FakeMemoryStore(forget_result=3)
    service = MemoryAdminService(store, RecordingSafetyPolicy(True))

    result = service.forget(_MEMORY_ID)

    assert result == 3
    assert store.forget_calls == [_MEMORY_ID]


def test_forget_rejects_blank_memory_id_before_store() -> None:
    store = FakeMemoryStore()
    service = MemoryAdminService(store, RecordingSafetyPolicy(True))

    with pytest.raises(InvalidMemoryError):
        service.forget(" \t\n")

    assert store.forget_calls == []

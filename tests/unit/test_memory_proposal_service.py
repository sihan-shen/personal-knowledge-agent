"""Tests for automatic pending-memory proposal coordination."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from personal_knowledge_agent.application.memory_proposal import MemoryProposalService
from personal_knowledge_agent.domain.errors import (
    EmbeddingError,
    InvalidMemoryError,
    InvalidMemorySchemaError,
    LanguageModelError,
    MemoryStorageError,
)
from personal_knowledge_agent.domain.memory import (
    MemoryExtractionSkipReason,
    MemoryProposal,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    memory_content_hash,
    memory_source_hash,
)
from tests.helpers import FakeMemoryStore

_MESSAGE = "My goal is to build an agent"
_NOW = datetime(2026, 7, 17, tzinfo=UTC)


def _proposal(
    memory_type: MemoryType = MemoryType.GOAL,
    memory_key: str = "goal.agent",
    content: str = "Build an agent",
) -> MemoryProposal:
    return MemoryProposal(memory_type, memory_key, content)


def _pending_record() -> MemoryRecord:
    content = "Build an agent"
    return MemoryRecord(
        memory_id="mem_00000000000000000000000000000001",
        memory_type=MemoryType.GOAL,
        memory_key="goal.agent",
        content=content,
        content_hash=memory_content_hash(content),
        status=MemoryStatus.PENDING,
        supersedes_id=None,
        source_hash=memory_source_hash(_MESSAGE),
        created_at=_NOW,
        updated_at=_NOW,
    )


class RecordingPrefilter:
    def __init__(self, result: bool, events: list[str] | None = None) -> None:
        self._result = result
        self._events = events
        self.calls: list[str] = []

    def __call__(self, message: str) -> bool:
        self.calls.append(message)
        if self._events is not None:
            self._events.append("prefilter")
        return self._result


class RecordingSafetyPolicy:
    def __init__(
        self,
        results: Sequence[bool],
        events: list[str] | None = None,
    ) -> None:
        self._results = tuple(results)
        self._events = events
        self.calls: list[tuple[str, ...]] = []

    def allows(self, *values: str) -> bool:
        self.calls.append(values)
        if self._events is not None:
            event = "raw_safety" if len(self.calls) == 1 else "proposal_safety"
            self._events.append(event)
        try:
            return self._results[len(self.calls) - 1]
        except IndexError as error:
            raise AssertionError("No configured safety result remains") from error


class RecordingExtractor:
    def __init__(
        self,
        result: tuple[MemoryProposal, ...] | Exception,
        events: list[str] | None = None,
    ) -> None:
        self._result = result
        self._events = events
        self.calls: list[str] = []

    def extract(self, message: str) -> tuple[MemoryProposal, ...]:
        self.calls.append(message)
        if self._events is not None:
            self._events.append("extract")
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class EventMemoryStore(FakeMemoryStore):
    def __init__(
        self,
        events: list[str],
        result: tuple[MemoryRecord, ...] | Exception,
    ) -> None:
        super().__init__(create_pending_result=result)
        self._events = events

    def create_pending(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        source_hash: str,
    ) -> tuple[MemoryRecord, ...]:
        self._events.append("store")
        return super().create_pending(proposals, source_hash=source_hash)


def test_propose_coordinates_complete_pending_batch_in_exact_order() -> None:
    events: list[str] = []
    proposal = _proposal()
    pending_record = _pending_record()
    prefilter = RecordingPrefilter(True, events)
    safety = RecordingSafetyPolicy([True, True], events)
    extractor = RecordingExtractor((proposal,), events)
    store = EventMemoryStore(events, (pending_record,))
    service = MemoryProposalService(extractor, store, safety, prefilter)

    report = service.propose(_MESSAGE)

    assert events == [
        "prefilter",
        "raw_safety",
        "extract",
        "proposal_safety",
        "store",
    ]
    assert safety.calls == [(_MESSAGE,), (proposal.memory_key, proposal.content)]
    assert report.candidates == (pending_record,)
    assert report.skipped_reason is None
    assert store.create_pending_calls == [((proposal,), memory_source_hash(_MESSAGE))]


def test_propose_rejects_blank_before_dependencies() -> None:
    prefilter = RecordingPrefilter(True)
    safety = RecordingSafetyPolicy([True, True])
    extractor = RecordingExtractor((_proposal(),))
    store = FakeMemoryStore(create_pending_result=(_pending_record(),))
    service = MemoryProposalService(extractor, store, safety, prefilter)

    with pytest.raises(InvalidMemoryError, match="message"):
        service.propose(" \t\n")

    assert prefilter.calls == []
    assert safety.calls == []
    assert extractor.calls == []
    assert store.create_pending_calls == []


def test_propose_skips_oversized_input_before_dependencies() -> None:
    prefilter = RecordingPrefilter(True)
    safety = RecordingSafetyPolicy([True, True])
    extractor = RecordingExtractor((_proposal(),))
    store = FakeMemoryStore(create_pending_result=(_pending_record(),))
    service = MemoryProposalService(extractor, store, safety, prefilter)

    report = service.propose("x" * 4_001)

    assert report.candidates == ()
    assert report.skipped_reason is MemoryExtractionSkipReason.INPUT_TOO_LONG
    assert prefilter.calls == []
    assert safety.calls == []
    assert extractor.calls == []
    assert store.create_pending_calls == []


def test_propose_prefilter_miss_avoids_safety_extraction_and_storage() -> None:
    prefilter = RecordingPrefilter(False)
    safety = RecordingSafetyPolicy([True, True])
    extractor = RecordingExtractor((_proposal(),))
    store = FakeMemoryStore(create_pending_result=(_pending_record(),))
    service = MemoryProposalService(extractor, store, safety, prefilter)

    report = service.propose(_MESSAGE)

    assert report.skipped_reason is MemoryExtractionSkipReason.PREFILTER_MISS
    assert prefilter.calls == [_MESSAGE]
    assert safety.calls == []
    assert extractor.calls == []
    assert store.create_pending_calls == []


def test_propose_sensitive_input_avoids_extraction_and_storage() -> None:
    prefilter = RecordingPrefilter(True)
    safety = RecordingSafetyPolicy([False])
    extractor = RecordingExtractor((_proposal(),))
    store = FakeMemoryStore(create_pending_result=(_pending_record(),))
    service = MemoryProposalService(extractor, store, safety, prefilter)

    report = service.propose(_MESSAGE)

    assert report.skipped_reason is MemoryExtractionSkipReason.SENSITIVE_INPUT
    assert safety.calls == [(_MESSAGE,)]
    assert extractor.calls == []
    assert store.create_pending_calls == []


def test_propose_empty_extraction_skips_safety_and_storage() -> None:
    prefilter = RecordingPrefilter(True)
    safety = RecordingSafetyPolicy([True])
    extractor = RecordingExtractor(())
    store = FakeMemoryStore(create_pending_result=(_pending_record(),))
    service = MemoryProposalService(extractor, store, safety, prefilter)

    report = service.propose(_MESSAGE)

    assert report.skipped_reason is MemoryExtractionSkipReason.NO_NEW_CANDIDATES
    assert safety.calls == [(_MESSAGE,)]
    assert extractor.calls == [_MESSAGE]
    assert store.create_pending_calls == []


def test_propose_unsafe_parsed_proposal_rejects_complete_batch() -> None:
    proposals = (
        _proposal(),
        _proposal(
            MemoryType.PROJECT,
            "project.private_key",
            "Keep a private key",
        ),
    )
    prefilter = RecordingPrefilter(True)
    safety = RecordingSafetyPolicy([True, False])
    extractor = RecordingExtractor(proposals)
    store = FakeMemoryStore(create_pending_result=(_pending_record(),))
    service = MemoryProposalService(extractor, store, safety, prefilter)

    report = service.propose(_MESSAGE)

    assert report.skipped_reason is MemoryExtractionSkipReason.UNSAFE_OUTPUT
    assert safety.calls == [
        (_MESSAGE,),
        (
            "goal.agent",
            "Build an agent",
            "project.private_key",
            "Keep a private key",
        ),
    ]
    assert store.create_pending_calls == []


@pytest.mark.parametrize(
    ("error", "expected_reason", "failure_stage"),
    [
        (
            LanguageModelError("provider body must remain hidden"),
            MemoryExtractionSkipReason.PROVIDER_ERROR,
            "extract",
        ),
        (
            InvalidMemorySchemaError("invalid model content"),
            MemoryExtractionSkipReason.INVALID_OUTPUT,
            "extract",
        ),
        (
            EmbeddingError("invalid private vector"),
            MemoryExtractionSkipReason.EMBEDDING_ERROR,
            "store",
        ),
        (
            MemoryStorageError("database details"),
            MemoryExtractionSkipReason.STORAGE_ERROR,
            "store",
        ),
    ],
)
def test_propose_maps_expected_boundary_errors_to_safe_skip_reasons(
    error: Exception,
    expected_reason: MemoryExtractionSkipReason,
    failure_stage: str,
) -> None:
    extractor_result: tuple[MemoryProposal, ...] | Exception = (_proposal(),)
    store_result: tuple[MemoryRecord, ...] | Exception = (_pending_record(),)
    if failure_stage == "extract":
        extractor_result = error
    else:
        store_result = error
    service = MemoryProposalService(
        RecordingExtractor(extractor_result),
        FakeMemoryStore(create_pending_result=store_result),
        RecordingSafetyPolicy([True, True]),
        RecordingPrefilter(True),
    )

    report = service.propose(_MESSAGE)

    assert report.candidates == ()
    assert report.skipped_reason is expected_reason


def test_propose_maps_fully_duplicate_filtered_batch_to_no_new_candidates() -> None:
    store = FakeMemoryStore(create_pending_result=())
    service = MemoryProposalService(
        RecordingExtractor((_proposal(),)),
        store,
        RecordingSafetyPolicy([True, True]),
        RecordingPrefilter(True),
    )

    report = service.propose(_MESSAGE)

    assert report.skipped_reason is MemoryExtractionSkipReason.NO_NEW_CANDIDATES
    assert len(store.create_pending_calls) == 1


def test_propose_propagates_unexpected_programming_errors() -> None:
    error = RuntimeError("programming defect")
    store = FakeMemoryStore(create_pending_result=(_pending_record(),))
    service = MemoryProposalService(
        RecordingExtractor(error),
        store,
        RecordingSafetyPolicy([True]),
        RecordingPrefilter(True),
    )

    with pytest.raises(RuntimeError, match="programming defect"):
        service.propose(_MESSAGE)

    assert store.create_pending_calls == []

"""Tests for shared deterministic boundary doubles."""

from datetime import UTC, datetime

import pytest

from personal_knowledge_agent.domain.errors import (
    EmbeddingError,
    LanguageModelError,
    MemoryStorageError,
)
from personal_knowledge_agent.domain.memory import (
    MemoryActivationAction,
    MemoryProposal,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
    memory_content_hash,
)
from personal_knowledge_agent.domain.models import ChatMessage, MessageRole, ModelOutput
from tests.helpers import (
    FakeEmbedder,
    FakeLanguageModel,
    FakeMemoryStore,
    ScriptedLanguageModel,
)

_NOW = datetime(2026, 7, 17, tzinfo=UTC)


def _memory_record(status: MemoryStatus = MemoryStatus.PENDING) -> MemoryRecord:
    content = "Build an agent"
    return MemoryRecord(
        memory_id="mem_00000000000000000000000000000001",
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


def test_fake_embedder_normalizes_vectors_and_records_boundary_calls() -> None:
    embedder = FakeEmbedder({"passage": (3.0, 4.0), "query": (0.0, 2.0)})

    documents = embedder.embed_documents(["passage"])
    query = embedder.embed_query("query")

    assert documents[0].values == pytest.approx((0.6, 0.8))
    assert query.values == pytest.approx((0.0, 1.0))
    assert embedder.document_calls == [["passage"]]
    assert embedder.query_calls == ["query"]


def test_fake_embedder_rejects_unknown_text_without_model_runtime() -> None:
    embedder = FakeEmbedder({})

    with pytest.raises(EmbeddingError, match="No fake embedding"):
        embedder.embed_query("unknown")


def test_fake_embedder_can_supply_a_default_for_unimportant_test_text() -> None:
    embedder = FakeEmbedder({}, default_vector=(2.0, 0.0))

    assert embedder.embed_documents(["anything"])[0].values == (1.0, 0.0)


def test_fake_language_model_returns_output_and_records_immutable_messages() -> None:
    output = ModelOutput("fake-model", '{"status":"answered"}', "stop")
    model = FakeLanguageModel(output)
    messages = (ChatMessage(MessageRole.USER, "question"),)

    assert model.generate(messages) == output
    assert model.calls == [messages]


def test_fake_language_model_raises_configured_error() -> None:
    model = FakeLanguageModel(LanguageModelError("offline"))

    with pytest.raises(LanguageModelError, match="offline"):
        model.generate((ChatMessage(MessageRole.USER, "question"),))


def test_scripted_language_model_returns_results_in_order_and_records_calls() -> None:
    first = ModelOutput("router", '{"action":"retrieve"}', "stop")
    second = ModelOutput("answer", '{"status":"answered"}', "stop")
    model = ScriptedLanguageModel([first, second])
    messages = (ChatMessage(MessageRole.USER, "question"),)

    assert model.generate(messages) == first
    assert model.generate(messages) == second
    assert model.calls == [messages, messages]


def test_scripted_language_model_raises_configured_error_then_continues() -> None:
    expected = ModelOutput("answer", '{"status":"answered"}', "stop")
    model = ScriptedLanguageModel([LanguageModelError("offline"), expected])
    messages = (ChatMessage(MessageRole.USER, "question"),)

    with pytest.raises(LanguageModelError, match="offline"):
        model.generate(messages)

    assert model.generate(messages) == expected


def test_scripted_language_model_fails_clearly_when_exhausted() -> None:
    model = ScriptedLanguageModel([])

    with pytest.raises(AssertionError, match="No scripted"):
        model.generate((ChatMessage(MessageRole.USER, "question"),))


def test_fake_memory_store_returns_configured_values_and_records_all_calls() -> None:
    proposal = MemoryProposal(MemoryType.GOAL, "goal.agent", "Build an agent")
    pending = _memory_record()
    active = _memory_record(MemoryStatus.ACTIVE)
    search_result = MemorySearchResult(active, 0.75, 0.8)
    store = FakeMemoryStore(
        create_pending_result=(pending,),
        remember_result=active,
        approve_result=active,
        forget_result=2,
        list_memories_result=(pending, active),
        search_result=(search_result,),
    )

    assert store.create_pending((proposal,), source_hash="a" * 64) == (pending,)
    assert (
        store.remember(
            MemoryType.GOAL,
            "goal.agent",
            "Build an agent",
            action=MemoryActivationAction.REPLACE,
            replace_id=pending.memory_id,
        )
        == active
    )
    assert (
        store.approve(
            pending.memory_id,
            action=MemoryActivationAction.ACTIVATE,
        )
        == active
    )
    assert store.reject(pending.memory_id) is None
    assert store.forget(active.memory_id) == 2
    assert store.list_memories(status=MemoryStatus.PENDING) == [pending, active]
    assert store.search("agent", limit=3, memory_type=MemoryType.GOAL) == [
        search_result
    ]
    assert store.create_pending_calls == [((proposal,), "a" * 64)]
    assert store.remember_calls == [
        (
            MemoryType.GOAL,
            "goal.agent",
            "Build an agent",
            MemoryActivationAction.REPLACE,
            pending.memory_id,
        )
    ]
    assert store.approve_calls == [
        (pending.memory_id, MemoryActivationAction.ACTIVATE, None)
    ]
    assert store.reject_calls == [pending.memory_id]
    assert store.forget_calls == [active.memory_id]
    assert store.list_memories_calls == [(MemoryStatus.PENDING, None)]
    assert store.search_calls == [("agent", 3, MemoryType.GOAL)]


def test_fake_memory_store_raises_configured_error_after_recording_call() -> None:
    proposal = MemoryProposal(MemoryType.GOAL, "goal.agent", "Build an agent")
    store = FakeMemoryStore(
        create_pending_result=MemoryStorageError("storage unavailable")
    )

    with pytest.raises(MemoryStorageError, match="storage unavailable"):
        store.create_pending((proposal,), source_hash="b" * 64)

    assert store.create_pending_calls == [((proposal,), "b" * 64)]

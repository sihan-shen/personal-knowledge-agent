"""Integration tests for the persistent SQLite memory lifecycle."""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import (
    EmbeddingError,
    InvalidMemoryError,
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryStorageError,
)
from personal_knowledge_agent.domain.memory import (
    MemoryActivationAction,
    MemoryProposal,
    MemoryStatus,
    MemoryType,
    memory_content_hash,
    memory_proposal_hash,
    memory_source_hash,
)
from personal_knowledge_agent.domain.models import Embedding
from personal_knowledge_agent.domain.protocols import EmbeddingModel
from personal_knowledge_agent.infrastructure.sqlite_memory_store import (
    SQLiteMemoryStore,
)
from tests.helpers import FakeEmbedder

_FIRST_ID = "mem_00000000000000000000000000000001"
_SECOND_ID = "mem_00000000000000000000000000000002"
_THIRD_ID = "mem_00000000000000000000000000000003"
_FIXED_TIME = datetime(2026, 7, 18, 9, 30, tzinfo=UTC)


class SequenceFactory:
    """Return deterministic values in call order."""

    def __init__(self, values: Sequence[str]) -> None:
        self._values: Iterator[str] = iter(values)

    def __call__(self) -> str:
        return next(self._values)


class SequenceClock:
    """Return deterministic UTC instants in call order."""

    def __init__(self, *values: datetime) -> None:
        self._values: Iterator[datetime] = iter(values)

    def __call__(self) -> datetime:
        return next(self._values)


class WrongCountEmbedder:
    @property
    def model_id(self) -> str:
        return "wrong-count"

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        del texts
        return []

    def embed_query(self, query: str) -> Embedding:
        del query
        return Embedding(self.model_id, (1.0,))


class WrongModelIdEmbedder(WrongCountEmbedder):
    @property
    def model_id(self) -> str:
        return "configured-model"

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [Embedding("different-model", (1.0,)) for _ in texts]


def read_rows(
    database: Path,
    query: str,
    parameters: Sequence[object] = (),
) -> list[tuple[object, ...]]:
    with sqlite3.connect(database) as connection:
        return connection.execute(query, parameters).fetchall()


def snapshot_memory_tables(
    database: Path,
) -> dict[str, list[tuple[object, ...]]]:
    """Capture all durable memory table groups for rollback assertions."""
    return {
        table: read_rows(database, f"SELECT * FROM {table} ORDER BY rowid")
        for table in (
            "memories",
            "memory_postings",
            "memory_embeddings",
            "memory_rejections",
        )
    }


def make_store(
    database: Path,
    *,
    embedder: EmbeddingModel | None = None,
    clock: Callable[[], datetime] = lambda: _FIXED_TIME,
    ids: Sequence[str] = (_FIRST_ID, _SECOND_ID, _THIRD_ID),
) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(
        database,
        embedder or FakeEmbedder({}, default_vector=(3.0, 4.0)),
        clock=clock,
        id_factory=SequenceFactory(ids),
    )


def test_initializes_constrained_schema_version_one(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "memory.db"

    make_store(database)

    assert read_rows(database, "SELECT singleton, version FROM schema_metadata") == [
        (1, 1)
    ]
    table_sql = {
        str(name): str(sql)
        for name, sql in read_rows(
            database,
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'",
        )
    }
    assert {
        "schema_metadata",
        "memories",
        "memory_postings",
        "memory_embeddings",
        "memory_rejections",
    } <= table_sql.keys()
    assert "CHECK (singleton = 1)" in table_sql["schema_metadata"]
    assert "memory_type IN ('preference', 'goal', 'project', 'learning_progress')" in (
        " ".join(table_sql["memories"].split())
    )
    assert "status IN ('pending', 'active', 'superseded')" in " ".join(
        table_sql["memories"].split()
    )
    assert "DEFERRABLE INITIALLY DEFERRED" in table_sql["memories"]

    indexes = {
        str(name): str(sql)
        for name, sql in read_rows(
            database,
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'index' AND sql IS NOT NULL",
        )
    }
    assert {
        "idx_memory_live_duplicate",
        "idx_memories_status_type_key",
        "idx_memory_postings_token",
        "idx_memory_embeddings_model_id",
    } <= indexes.keys()
    assert "WHERE status IN ('pending', 'active')" in " ".join(
        indexes["idx_memory_live_duplicate"].split()
    )

    posting_fks = read_rows(database, "PRAGMA foreign_key_list(memory_postings)")
    embedding_fks = read_rows(database, "PRAGMA foreign_key_list(memory_embeddings)")
    assert posting_fks[0][2:4] == ("memories", "memory_id")
    assert posting_fks[0][6] == "CASCADE"
    assert embedding_fks[0][2:4] == ("memories", "memory_id")
    assert embedding_fks[0][6] == "CASCADE"


def test_persists_normalized_pending_batch_with_postings_and_vectors(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    source_hash = memory_source_hash("I prefer Python and 学习 RAG")
    embedder = FakeEmbedder(
        {
            "Uses Python": (3.0, 4.0),
            "学习 RAG": (0.0, 5.0),
        },
        model_id="fake-e5",
    )
    store = make_store(database, embedder=embedder)

    records = store.create_pending(
        (
            MemoryProposal(
                MemoryType.PREFERENCE, "preference.language", "Ｕｓｅｓ  Python"
            ),
            MemoryProposal(MemoryType.LEARNING_PROGRESS, "learning.rag", "学习 RAG"),
        ),
        source_hash=source_hash,
    )

    assert [record.memory_id for record in records] == [_FIRST_ID, _SECOND_ID]
    assert [record.status for record in records] == [
        MemoryStatus.PENDING,
        MemoryStatus.PENDING,
    ]
    assert records[0].content == "Uses Python"
    assert {record.source_hash for record in records} == {source_hash}
    assert read_rows(
        database,
        "SELECT memory_id, token_count FROM memories ORDER BY memory_id",
    ) == [(_FIRST_ID, 2), (_SECOND_ID, 2)]
    assert read_rows(
        database,
        "SELECT token, memory_id, frequency FROM memory_postings "
        "ORDER BY memory_id, token",
    ) == [
        ("python", _FIRST_ID, 1),
        ("uses", _FIRST_ID, 1),
        ("rag", _SECOND_ID, 1),
        ("学习", _SECOND_ID, 1),
    ]
    embedding_rows = read_rows(
        database,
        "SELECT memory_id, model_id, dimensions, vector FROM memory_embeddings "
        "ORDER BY memory_id",
    )
    assert embedding_rows[0][:3] == (_FIRST_ID, "fake-e5", 2)
    assert struct.unpack("<2f", bytes(embedding_rows[0][3])) == pytest.approx(
        (0.6, 0.8)
    )
    assert embedding_rows[1][:3] == (_SECOND_ID, "fake-e5", 2)
    assert struct.unpack("<2f", bytes(embedding_rows[1][3])) == pytest.approx(
        (0.0, 1.0)
    )
    assert embedder.document_calls == [["Uses Python", "学习 RAG"]]


def test_explicit_memory_is_active_without_source_hash(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)

    record = store.remember(
        MemoryType.PROJECT,
        "project.rag",
        "Ｂｕｉｌｄ  RAG",
    )

    assert record.content == "Build RAG"
    assert record.status is MemoryStatus.ACTIVE
    assert record.source_hash is None
    assert record.supersedes_id is None
    assert read_rows(database, "SELECT source_hash FROM memories") == [(None,)]


def test_lists_newest_first_with_optional_status_and_type_filters(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    store = make_store(
        database,
        clock=SequenceClock(_FIXED_TIME, _FIXED_TIME + timedelta(minutes=1)),
    )
    first = store.remember(MemoryType.GOAL, "goal.agent", "Build an agent")
    second = store.create_pending(
        (MemoryProposal(MemoryType.PROJECT, "project.rag", "Build RAG"),),
        source_hash=memory_source_hash("My RAG project"),
    )[0]

    assert store.list_memories() == [second, first]
    assert store.list_memories(status=MemoryStatus.ACTIVE) == [first]
    assert store.list_memories(memory_type=MemoryType.PROJECT) == [second]


def test_persists_memories_after_reopening(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    expected = make_store(database).remember(
        MemoryType.GOAL,
        "goal.agent",
        "Build an agent",
    )

    reopened = make_store(database, ids=(_SECOND_ID,))

    assert reopened.list_memories() == [expected]


def test_suppresses_exact_live_duplicate_triples(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    proposal = MemoryProposal(MemoryType.GOAL, "goal.agent", "Build an agent")
    first = store.create_pending(
        (proposal,),
        source_hash=memory_source_hash("first source"),
    )

    duplicate = store.create_pending(
        (
            proposal,
            MemoryProposal(MemoryType.PROJECT, "goal.agent", "Build an agent"),
            MemoryProposal(MemoryType.GOAL, "goal.other", "Build an agent"),
        ),
        source_hash=memory_source_hash("second source"),
    )

    assert len(first) == 1
    assert [(record.memory_type, record.memory_key) for record in duplicate] == [
        (MemoryType.PROJECT, "goal.agent"),
        (MemoryType.GOAL, "goal.other"),
    ]
    assert read_rows(database, "SELECT COUNT(*) FROM memories") == [(3,)]


def test_pending_creation_suppresses_rejected_fingerprints(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    rejected = MemoryProposal(MemoryType.GOAL, "goal.agent", "Build an agent")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO memory_rejections(proposal_hash, rejected_at) VALUES (?, ?)",
            (
                memory_proposal_hash(
                    rejected.memory_type,
                    rejected.memory_key,
                    rejected.content,
                ),
                _FIXED_TIME.isoformat(),
            ),
        )

    created = store.create_pending(
        (
            rejected,
            MemoryProposal(MemoryType.GOAL, "goal.rag", "Learn RAG"),
        ),
        source_hash=memory_source_hash("source"),
    )

    assert [record.content for record in created] == ["Learn RAG"]
    assert read_rows(database, "SELECT content FROM memories") == [("Learn RAG",)]


def test_duplicate_ids_roll_back_entire_pending_batch(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database, ids=(_FIRST_ID, _FIRST_ID))
    before = snapshot_memory_tables(database)

    with pytest.raises(MemoryStorageError):
        store.create_pending(
            (
                MemoryProposal(MemoryType.GOAL, "goal.first", "First goal"),
                MemoryProposal(MemoryType.GOAL, "goal.second", "Second goal"),
            ),
            source_hash=memory_source_hash("source"),
        )

    assert snapshot_memory_tables(database) == before


def test_sql_abort_mid_batch_restores_all_memory_table_groups(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(
        database,
        ids=(_FIRST_ID, _SECOND_ID, _THIRD_ID),
    )
    store.remember(MemoryType.PROJECT, "project.existing", "Existing project")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO memory_rejections(proposal_hash, rejected_at) VALUES (?, ?)",
            (
                memory_proposal_hash(
                    MemoryType.GOAL,
                    "goal.rejected",
                    "Rejected goal",
                ),
                _FIXED_TIME.isoformat(),
            ),
        )
        connection.execute(
            f"""
            CREATE TRIGGER fail_second_batch_insert
            BEFORE INSERT ON memories
            WHEN NEW.memory_id = '{_THIRD_ID}'
            BEGIN
                SELECT RAISE(ABORT, 'private mid-batch constraint detail');
            END
            """
        )
    before = snapshot_memory_tables(database)

    with pytest.raises(MemoryStorageError) as captured:
        store.create_pending(
            (
                MemoryProposal(MemoryType.GOAL, "goal.first", "First goal"),
                MemoryProposal(MemoryType.GOAL, "goal.second", "Second goal"),
            ),
            source_hash=memory_source_hash("source"),
        )

    assert "private mid-batch constraint detail" not in str(captured.value)
    assert snapshot_memory_tables(database) == before


@pytest.mark.parametrize(
    ("embedder", "message"),
    [
        (WrongCountEmbedder(), "0 vectors for 1 memories"),
        (WrongModelIdEmbedder(), "identifier"),
    ],
)
def test_rejects_incompatible_embedding_batches_before_writing(
    tmp_path: Path,
    embedder: EmbeddingModel,
    message: str,
) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database, embedder=embedder)
    before = snapshot_memory_tables(database)

    with pytest.raises(EmbeddingError, match=message):
        store.remember(MemoryType.GOAL, "goal.agent", "Build an agent")

    assert snapshot_memory_tables(database) == before


def test_rejects_mixed_embedding_dimensions_before_writing(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    embedder = FakeEmbedder(
        {
            "First goal": (1.0,),
            "Second goal": (1.0, 0.0),
        }
    )
    store = make_store(database, embedder=embedder)
    before = snapshot_memory_tables(database)

    with pytest.raises(EmbeddingError, match="common dimension"):
        store.create_pending(
            (
                MemoryProposal(MemoryType.GOAL, "goal.first", "First goal"),
                MemoryProposal(MemoryType.GOAL, "goal.second", "Second goal"),
            ),
            source_hash=memory_source_hash("source"),
        )

    assert snapshot_memory_tables(database) == before


def test_list_maps_corrupt_rows_to_safe_storage_error(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    secret = "private content that must not leak"
    store.remember(MemoryType.GOAL, "goal.agent", secret)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE memories SET content_hash = ?",
            (memory_content_hash("different content"),),
        )

    with pytest.raises(MemoryStorageError) as captured:
        store.list_memories()

    assert secret not in str(captured.value)


def test_search_ranks_higher_bm25_term_frequency(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    embedder = FakeEmbedder(
        {
            "Python Python Python notes": (0.0, 1.0),
            "Python notes notes notes": (0.0, 1.0),
            "python": (1.0, 0.0),
        }
    )
    store = make_store(database, embedder=embedder)
    frequent = store.remember(
        MemoryType.LEARNING_PROGRESS,
        "learning.python",
        "Python Python Python notes",
    )
    occasional = store.remember(
        MemoryType.LEARNING_PROGRESS,
        "learning.notes",
        "Python notes notes notes",
    )

    results = store.search("python")

    assert [result.memory.memory_id for result in results] == [
        frequent.memory_id,
        occasional.memory_id,
    ]
    assert [result.score for result in results] == pytest.approx([1 / 61, 1 / 62])


def test_search_supports_chinese_and_mixed_language_terms(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    embedder = FakeEmbedder(
        {
            "正在学习 RAG 检索": (0.0, 1.0),
            "RAG 检索": (1.0, 0.0),
        }
    )
    store = make_store(database, embedder=embedder)
    memory = store.remember(
        MemoryType.LEARNING_PROGRESS,
        "learning.rag",
        "正在学习 RAG 检索",
    )

    results = store.search("RAG 检索")

    assert [result.memory for result in results] == [memory]


def test_search_orders_only_positive_cosine_candidates(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    embedder = FakeEmbedder(
        {
            "closest": (1.0, 0.0),
            "partial": (1.0, 1.0),
            "opposite": (-1.0, 0.0),
            "semantic question": (1.0, 0.0),
        }
    )
    store = make_store(database, embedder=embedder)
    closest = store.remember(MemoryType.GOAL, "goal.closest", "closest")
    partial = store.remember(MemoryType.GOAL, "goal.partial", "partial")
    store.remember(MemoryType.GOAL, "goal.opposite", "opposite")

    results = store.search("semantic question")

    assert [result.memory.memory_id for result in results] == [
        closest.memory_id,
        partial.memory_id,
    ]
    assert [result.score for result in results] == pytest.approx([1 / 61, 1 / 62])
    assert embedder.query_calls == ["semantic question"]


def test_search_breaks_equal_rankings_by_memory_id(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    embedder = FakeEmbedder(
        {
            "same first": (1.0, 0.0),
            "same second": (1.0, 0.0),
            "same query": (1.0, 0.0),
        }
    )
    store = make_store(
        database,
        embedder=embedder,
        ids=(_SECOND_ID, _FIRST_ID),
    )
    created = [
        store.remember(MemoryType.GOAL, "goal.first", "same first"),
        store.remember(MemoryType.GOAL, "goal.second", "same second"),
    ]

    results = store.search("same query")

    assert [result.memory.memory_id for result in results] == sorted(
        memory.memory_id for memory in created
    )


def test_search_fuses_lexical_and_semantic_rankings_by_memory_id(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    embedder = FakeEmbedder(
        {
            "hybrid hybrid hybrid": (0.0, 1.0),
            "semantic only": (1.0, 0.0),
            "hybrid shared": (1.0, 1.0),
            "hybrid question": (1.0, 0.0),
        }
    )
    store = make_store(database, embedder=embedder)
    lexical_first = store.remember(
        MemoryType.PROJECT,
        "project.lexical",
        "hybrid hybrid hybrid",
    )
    semantic_first = store.remember(
        MemoryType.PROJECT,
        "project.semantic",
        "semantic only",
    )
    shared = store.remember(
        MemoryType.PROJECT,
        "project.shared",
        "hybrid shared",
    )

    results = store.search("hybrid question")

    assert results[0].memory == shared
    assert {result.memory for result in results[1:]} == {
        lexical_first,
        semantic_first,
    }
    assert results[0].score == pytest.approx(2 / 62)
    assert results[1].score == pytest.approx(1 / 61)


@pytest.mark.parametrize("limit", [0, 21])
def test_search_rejects_limits_outside_supported_range(
    tmp_path: Path,
    limit: int,
) -> None:
    store = make_store(tmp_path / "memory.db")

    with pytest.raises(InvalidMemoryError, match="between 1 and 20"):
        store.search("memory", limit=limit)


def test_search_applies_limit_and_optional_type_filter(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    embedder = FakeEmbedder({}, default_vector=(1.0, 0.0))
    store = make_store(database, embedder=embedder)
    goal = store.remember(MemoryType.GOAL, "goal.agent", "shared memory goal")
    store.remember(MemoryType.PROJECT, "project.rag", "shared memory project")
    store.remember(
        MemoryType.LEARNING_PROGRESS,
        "learning.rag",
        "shared memory learning",
    )

    filtered = store.search("shared memory", limit=20, memory_type=MemoryType.GOAL)
    limited = store.search("shared memory", limit=1)

    assert [result.memory for result in filtered] == [goal]
    assert len(limited) == 1


def test_search_persists_after_reopening(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    embedder = FakeEmbedder(
        {
            "Persistent memory": (1.0, 0.0),
            "persistent": (1.0, 0.0),
        }
    )
    expected = make_store(database, embedder=embedder).remember(
        MemoryType.PROJECT,
        "project.persistent",
        "Persistent memory",
    )

    results = make_store(database, embedder=embedder, ids=(_SECOND_ID,)).search(
        "persistent"
    )

    assert [result.memory for result in results] == [expected]


def test_search_returns_empty_without_embedding_when_no_active_memory(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    embedder = FakeEmbedder(
        {
            "Pending memory": (1.0, 0.0),
            "Active project": (1.0, 0.0),
            "memory": (1.0, 0.0),
        }
    )
    store = make_store(database, embedder=embedder)
    store.create_pending(
        (MemoryProposal(MemoryType.GOAL, "goal.pending", "Pending memory"),),
        source_hash=memory_source_hash("source"),
    )

    assert store.search("memory") == []
    assert embedder.query_calls == []

    store.remember(MemoryType.PROJECT, "project.active", "Active project")

    assert store.search("memory", memory_type=MemoryType.GOAL) == []
    assert embedder.query_calls == []


def test_search_rejects_query_without_searchable_terms(tmp_path: Path) -> None:
    embedder = FakeEmbedder({})
    store = make_store(tmp_path / "memory.db", embedder=embedder)

    with pytest.raises(InvalidMemoryError, match="searchable terms"):
        store.search("，。！？")

    assert embedder.query_calls == []


def test_search_never_returns_pending_or_superseded_memories(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    embedder = FakeEmbedder({}, default_vector=(1.0, 0.0))
    store = make_store(database, embedder=embedder)
    old = store.remember(MemoryType.GOAL, "goal.agent", "Old hidden memory")
    active = store.remember(
        MemoryType.GOAL,
        "goal.agent",
        "Current visible memory",
        action=MemoryActivationAction.REPLACE,
        replace_id=old.memory_id,
    )
    pending = store.create_pending(
        (MemoryProposal(MemoryType.PROJECT, "project.pending", "Pending memory"),),
        source_hash=memory_source_hash("source"),
    )[0]

    results = store.search("memory")

    assert [result.memory for result in results] == [active]
    assert {old.memory_id, pending.memory_id}.isdisjoint(
        result.memory.memory_id for result in results
    )


@pytest.mark.parametrize(
    ("mutation", "parameters"),
    [
        ("UPDATE memory_embeddings SET model_id = ?", ("private-old-model",)),
        ("UPDATE memory_embeddings SET dimensions = ?", (3,)),
        ("UPDATE memory_embeddings SET vector = ?", ("private-not-a-blob",)),
        ("UPDATE memory_embeddings SET vector = ?", (b"private",)),
    ],
)
def test_search_maps_corrupt_embedding_metadata_to_safe_storage_error(
    tmp_path: Path,
    mutation: str,
    parameters: tuple[object, ...],
) -> None:
    database = tmp_path / "memory.db"
    secret = "private memory content"
    embedder = FakeEmbedder(
        {
            secret: (1.0, 0.0),
            "memory": (1.0, 0.0),
        }
    )
    store = make_store(database, embedder=embedder)
    store.remember(MemoryType.GOAL, "goal.secret", secret)
    with sqlite3.connect(database) as connection:
        connection.execute(mutation, parameters)

    with pytest.raises(MemoryStorageError) as captured:
        store.search("memory")

    assert "private" not in str(captured.value)


def test_approves_pending_candidate_without_conflict(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    approved_at = _FIXED_TIME + timedelta(minutes=1)
    store = make_store(
        database,
        clock=SequenceClock(_FIXED_TIME, approved_at),
    )
    pending = store.create_pending(
        (MemoryProposal(MemoryType.GOAL, "goal.agent", "Build an agent"),),
        source_hash=memory_source_hash("source"),
    )[0]

    approved = store.approve(
        pending.memory_id,
        action=MemoryActivationAction.ACTIVATE,
    )

    assert approved.status is MemoryStatus.ACTIVE
    assert approved.created_at == _FIXED_TIME
    assert approved.updated_at == approved_at
    assert approved.supersedes_id is None
    assert read_rows(database, "SELECT COUNT(*) FROM memory_postings") == [(3,)]
    assert read_rows(database, "SELECT COUNT(*) FROM memory_embeddings") == [(1,)]


def test_bare_approval_reports_actionable_active_conflict_ids(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    active = store.remember(MemoryType.GOAL, "goal.agent", "First goal")
    pending = store.create_pending(
        (MemoryProposal(MemoryType.GOAL, "goal.agent", "Second goal"),),
        source_hash=memory_source_hash("source"),
    )[0]

    with pytest.raises(MemoryConflictError) as captured:
        store.approve(pending.memory_id, action=MemoryActivationAction.ACTIVATE)

    assert active.memory_id in str(captured.value)
    assert store.list_memories(status=MemoryStatus.ACTIVE) == [active]
    assert store.list_memories(status=MemoryStatus.PENDING) == [pending]


def test_approval_can_replace_same_slot_active_memory(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    replaced_at = _FIXED_TIME + timedelta(minutes=2)
    store = make_store(
        database,
        clock=SequenceClock(
            _FIXED_TIME,
            _FIXED_TIME + timedelta(minutes=1),
            replaced_at,
        ),
    )
    active = store.remember(MemoryType.GOAL, "goal.agent", "First goal")
    pending = store.create_pending(
        (MemoryProposal(MemoryType.GOAL, "goal.agent", "Second goal"),),
        source_hash=memory_source_hash("source"),
    )[0]

    approved = store.approve(
        pending.memory_id,
        action=MemoryActivationAction.REPLACE,
        replace_id=active.memory_id,
    )

    assert approved.status is MemoryStatus.ACTIVE
    assert approved.supersedes_id == active.memory_id
    superseded = store.list_memories(status=MemoryStatus.SUPERSEDED)
    assert len(superseded) == 1
    assert superseded[0].memory_id == active.memory_id
    assert superseded[0].updated_at == replaced_at


def test_approval_can_keep_both_same_slot_memories_active(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    active = store.remember(MemoryType.GOAL, "goal.agent", "First goal")
    pending = store.create_pending(
        (MemoryProposal(MemoryType.GOAL, "goal.agent", "Second goal"),),
        source_hash=memory_source_hash("source"),
    )[0]

    approved = store.approve(
        pending.memory_id,
        action=MemoryActivationAction.KEEP_BOTH,
    )

    assert approved.status is MemoryStatus.ACTIVE
    assert approved.supersedes_id is None
    assert {record.memory_id for record in store.list_memories()} == {
        active.memory_id,
        approved.memory_id,
    }
    assert all(record.status is MemoryStatus.ACTIVE for record in store.list_memories())


def test_replacement_rejects_target_from_different_slot(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    wrong_slot = store.remember(MemoryType.GOAL, "goal.other", "Other goal")
    pending = store.create_pending(
        (MemoryProposal(MemoryType.GOAL, "goal.agent", "Agent goal"),),
        source_hash=memory_source_hash("source"),
    )[0]

    with pytest.raises(MemoryConflictError, match="same memory slot"):
        store.approve(
            pending.memory_id,
            action=MemoryActivationAction.REPLACE,
            replace_id=wrong_slot.memory_id,
        )

    assert store.list_memories(status=MemoryStatus.PENDING) == [pending]
    assert store.list_memories(status=MemoryStatus.ACTIVE) == [wrong_slot]


def test_keep_both_requires_same_slot_conflict(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    pending = store.create_pending(
        (MemoryProposal(MemoryType.GOAL, "goal.agent", "Agent goal"),),
        source_hash=memory_source_hash("source"),
    )[0]

    with pytest.raises(MemoryConflictError, match="requires an active"):
        store.approve(
            pending.memory_id,
            action=MemoryActivationAction.KEEP_BOTH,
        )

    assert store.list_memories(status=MemoryStatus.PENDING) == [pending]


def test_approval_rejects_stale_nonpending_candidate(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    active = store.remember(MemoryType.GOAL, "goal.agent", "Agent goal")

    with pytest.raises(InvalidMemoryError, match="pending"):
        store.approve(active.memory_id, action=MemoryActivationAction.ACTIVATE)

    assert store.list_memories() == [active]


def test_remember_uses_same_replace_and_keep_both_rules(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    first = store.remember(MemoryType.GOAL, "goal.agent", "First goal")
    second = store.remember(
        MemoryType.GOAL,
        "goal.agent",
        "Second goal",
        action=MemoryActivationAction.KEEP_BOTH,
    )
    third = store.remember(
        MemoryType.GOAL,
        "goal.agent",
        "Third goal",
        action=MemoryActivationAction.REPLACE,
        replace_id=second.memory_id,
    )

    by_id = {record.memory_id: record for record in store.list_memories()}
    assert by_id[first.memory_id].status is MemoryStatus.ACTIVE
    assert by_id[second.memory_id].status is MemoryStatus.SUPERSEDED
    assert by_id[third.memory_id].status is MemoryStatus.ACTIVE
    assert by_id[third.memory_id].supersedes_id == second.memory_id


def test_reject_deletes_pending_content_and_children_but_keeps_fingerprint(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    rejected_at = _FIXED_TIME + timedelta(minutes=1)
    store = make_store(
        database,
        clock=SequenceClock(_FIXED_TIME, rejected_at),
    )
    proposal = MemoryProposal(MemoryType.GOAL, "goal.agent", "Secret candidate")
    pending = store.create_pending(
        (proposal,),
        source_hash=memory_source_hash("source"),
    )[0]

    store.reject(pending.memory_id)

    assert read_rows(database, "SELECT COUNT(*) FROM memories") == [(0,)]
    assert read_rows(database, "SELECT COUNT(*) FROM memory_postings") == [(0,)]
    assert read_rows(database, "SELECT COUNT(*) FROM memory_embeddings") == [(0,)]
    assert read_rows(
        database,
        "SELECT proposal_hash, rejected_at FROM memory_rejections",
    ) == [
        (
            memory_proposal_hash(
                proposal.memory_type,
                proposal.memory_key,
                proposal.content,
            ),
            rejected_at.isoformat(),
        )
    ]
    assert (
        store.create_pending(
            (proposal,),
            source_hash=memory_source_hash("another source"),
        )
        == ()
    )


def test_reject_accepts_pending_only(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    active = store.remember(MemoryType.GOAL, "goal.agent", "Agent goal")

    with pytest.raises(InvalidMemoryError, match="pending"):
        store.reject(active.memory_id)

    assert store.list_memories() == [active]
    assert read_rows(database, "SELECT COUNT(*) FROM memory_rejections") == [(0,)]


@pytest.mark.parametrize("operation", ["approve", "reject", "forget"])
def test_lifecycle_operations_reject_unknown_ids(
    tmp_path: Path,
    operation: str,
) -> None:
    store = make_store(tmp_path / "memory.db")

    with pytest.raises(MemoryNotFoundError):
        if operation == "approve":
            store.approve(_FIRST_ID, action=MemoryActivationAction.ACTIVATE)
        elif operation == "reject":
            store.reject(_FIRST_ID)
        else:
            store.forget(_FIRST_ID)


def test_duplicate_id_failure_rolls_back_replacement_status_and_children(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database, ids=(_FIRST_ID, _FIRST_ID))
    active = store.remember(MemoryType.GOAL, "goal.agent", "First goal")
    before = snapshot_memory_tables(database)

    with pytest.raises(MemoryStorageError):
        store.remember(
            MemoryType.GOAL,
            "goal.agent",
            "Second goal",
            action=MemoryActivationAction.REPLACE,
            replace_id=active.memory_id,
        )

    assert snapshot_memory_tables(database) == before
    assert store.list_memories() == [active]


def test_sql_failure_rolls_back_approval_status_changes_without_leaking_details(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    active = store.remember(MemoryType.GOAL, "goal.agent", "First goal")
    pending = store.create_pending(
        (MemoryProposal(MemoryType.GOAL, "goal.agent", "Second goal"),),
        source_hash=memory_source_hash("source"),
    )[0]
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_candidate_activation
            BEFORE UPDATE OF status ON memories
            WHEN NEW.memory_id = 'mem_00000000000000000000000000000002'
            BEGIN
                SELECT RAISE(ABORT, 'private trigger detail');
            END
            """
        )
    before = snapshot_memory_tables(database)

    with pytest.raises(MemoryStorageError) as captured:
        store.approve(
            pending.memory_id,
            action=MemoryActivationAction.REPLACE,
            replace_id=active.memory_id,
        )

    assert "private trigger detail" not in str(captured.value)
    assert snapshot_memory_tables(database) == before
    by_id = {record.memory_id: record for record in store.list_memories()}
    assert by_id[active.memory_id].status is MemoryStatus.ACTIVE
    assert by_id[pending.memory_id].status is MemoryStatus.PENDING
    assert by_id[pending.memory_id].supersedes_id is None


def test_reopened_store_rejects_stale_replacement_target_without_writes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    first_store = make_store(database, ids=(_FIRST_ID, _SECOND_ID))
    stale_target = first_store.remember(
        MemoryType.GOAL,
        "goal.agent",
        "First goal",
    )
    first_store.remember(
        MemoryType.GOAL,
        "goal.agent",
        "Current goal",
        action=MemoryActivationAction.REPLACE,
        replace_id=stale_target.memory_id,
    )
    reopened = make_store(database, ids=(_THIRD_ID,))
    before = snapshot_memory_tables(database)

    with pytest.raises(MemoryConflictError, match="must be active"):
        reopened.remember(
            MemoryType.GOAL,
            "goal.agent",
            "Stale replacement attempt",
            action=MemoryActivationAction.REPLACE,
            replace_id=stale_target.memory_id,
        )

    assert snapshot_memory_tables(database) == before


@pytest.mark.parametrize("start_position", [0, 1, 2])
def test_forget_removes_entire_bidirectional_replacement_chain_only(
    tmp_path: Path,
    start_position: int,
) -> None:
    database = tmp_path / "memory.db"
    store = make_store(
        database,
        ids=(_FIRST_ID, _SECOND_ID, _THIRD_ID, "mem_00000000000000000000000000000004"),
    )
    old = store.remember(MemoryType.GOAL, "goal.agent", "Old goal")
    replacement = store.remember(
        MemoryType.GOAL,
        "goal.agent",
        "Replacement goal",
        action=MemoryActivationAction.REPLACE,
        replace_id=old.memory_id,
    )
    newest = store.remember(
        MemoryType.GOAL,
        "goal.agent",
        "Newest goal",
        action=MemoryActivationAction.REPLACE,
        replace_id=replacement.memory_id,
    )
    unrelated = store.remember(
        MemoryType.PROJECT,
        "project.rag",
        "Unrelated project",
    )
    chain = (old, replacement, newest)
    with sqlite3.connect(database) as connection:
        connection.executemany(
            "INSERT INTO memory_rejections(proposal_hash, rejected_at) VALUES (?, ?)",
            (
                (
                    memory_proposal_hash(
                        record.memory_type,
                        record.memory_key,
                        record.content,
                    ),
                    _FIXED_TIME.isoformat(),
                )
                for record in (*chain, unrelated)
            ),
        )

    removed = store.forget(chain[start_position].memory_id)

    assert removed == 3
    assert store.list_memories() == [unrelated]
    assert read_rows(database, "SELECT memory_id FROM memory_postings") == [
        (unrelated.memory_id,),
        (unrelated.memory_id,),
    ]
    assert read_rows(database, "SELECT memory_id FROM memory_embeddings") == [
        (unrelated.memory_id,)
    ]
    assert read_rows(database, "SELECT proposal_hash FROM memory_rejections") == [
        (
            memory_proposal_hash(
                unrelated.memory_type,
                unrelated.memory_key,
                unrelated.content,
            ),
        )
    ]
    assert read_rows(database, "SELECT status FROM memories") == [("active",)]


@pytest.mark.parametrize(
    ("mutation", "parameters"),
    [
        ("UPDATE memories SET memory_id = ?", ("invalid-id",)),
        ("UPDATE memories SET memory_type = ?", ("private-invalid-type",)),
        ("UPDATE memories SET status = ?", ("private-invalid-status",)),
        ("UPDATE memories SET created_at = ?", ("private-invalid-time",)),
        ("UPDATE memories SET updated_at = ?", ("2026-07-18T09:30:00",)),
        ("UPDATE memories SET content_hash = ?", ("private-invalid-hash",)),
        ("UPDATE memories SET source_hash = ?", (123,)),
        ("UPDATE memories SET supersedes_id = ?", ("private-invalid-id",)),
        ("UPDATE memory_embeddings SET vector = ?", ("private-not-a-blob",)),
        ("UPDATE memory_embeddings SET vector = ?", (b"private",)),
    ],
)
def test_list_safely_maps_corrupt_row_metadata_and_vectors(
    tmp_path: Path,
    mutation: str,
    parameters: tuple[object, ...],
) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    secret = "private row content"
    store.remember(MemoryType.GOAL, "goal.agent", secret)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(mutation, parameters)

    with pytest.raises(MemoryStorageError) as captured:
        store.list_memories()

    assert secret not in str(captured.value)
    assert "private" not in str(captured.value)


def test_rejects_unsupported_schema_version_without_data_leakage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    make_store(database)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE schema_metadata SET version = 2")

    with pytest.raises(MemoryStorageError, match="version 2"):
        make_store(database)


@pytest.mark.parametrize(
    ("singleton", "version"),
    [
        (1.5, 1),
        (1, 1.5),
        ("1", 1),
        (1, "1"),
    ],
)
def test_rejects_noninteger_schema_metadata_values(
    tmp_path: Path,
    singleton: object,
    version: object,
) -> None:
    database = tmp_path / "memory.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE schema_metadata(singleton, version)")
        connection.execute(
            "INSERT INTO schema_metadata(singleton, version) VALUES (?, ?)",
            (singleton, version),
        )

    with pytest.raises(MemoryStorageError, match="metadata is invalid"):
        make_store(database)


def test_pending_filter_maps_corrupt_live_hash_to_safe_storage_error(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    secret = "private active content"
    store.remember(MemoryType.GOAL, "goal.agent", secret)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE memories SET content_hash = ?",
            ("private-invalid-hash",),
        )
    before = snapshot_memory_tables(database)

    with pytest.raises(MemoryStorageError) as captured:
        store.create_pending(
            (MemoryProposal(MemoryType.GOAL, "goal.other", "Other goal"),),
            source_hash=memory_source_hash("source"),
        )

    assert "private" not in str(captured.value)
    assert snapshot_memory_tables(database) == before


def test_pending_filter_maps_mismatched_live_content_hash_to_safe_storage_error(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    secret = "private active content"
    store.remember(MemoryType.GOAL, "goal.agent", secret)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE memories SET content_hash = ?",
            (memory_content_hash("different content"),),
        )
    before = snapshot_memory_tables(database)

    with pytest.raises(MemoryStorageError) as captured:
        store.create_pending(
            (MemoryProposal(MemoryType.GOAL, "goal.agent", secret),),
            source_hash=memory_source_hash("source"),
        )

    assert secret not in str(captured.value)
    assert snapshot_memory_tables(database) == before


def test_activation_maps_corrupt_conflict_id_to_safe_storage_error(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    store = make_store(database)
    store.remember(MemoryType.GOAL, "goal.agent", "First goal")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE memories SET memory_id = ?",
            ("private-invalid-id",),
        )
    before = snapshot_memory_tables(database)

    with pytest.raises(MemoryStorageError) as captured:
        store.remember(MemoryType.GOAL, "goal.agent", "Second goal")

    assert "private" not in str(captured.value)
    assert snapshot_memory_tables(database) == before


def test_schema_initialization_rolls_back_partial_creation(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE memories(existing_value TEXT)")

    with pytest.raises(MemoryStorageError):
        make_store(database)

    tables = {
        row[0]
        for row in read_rows(
            database,
            "SELECT name FROM sqlite_master WHERE type = 'table'",
        )
    }
    assert tables == {"memories"}

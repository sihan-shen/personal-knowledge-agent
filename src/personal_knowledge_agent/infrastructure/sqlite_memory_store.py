"""Persistent SQLite storage for confirmation-gated long-term memory."""

from __future__ import annotations

import re
import sqlite3
import struct
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import fsum, isfinite, log
from pathlib import Path

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
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
    make_memory_id,
    memory_content_hash,
    memory_proposal_hash,
)
from personal_knowledge_agent.domain.protocols import EmbeddingModel
from personal_knowledge_agent.infrastructure.mixed_tokenizer import (
    MixedLanguageTokenizer,
)
from personal_knowledge_agent.infrastructure.retrieval_fusion import (
    reciprocal_rank_scores,
)

_SCHEMA_VERSION = 1
_BM25_K1 = 1.5
_BM25_B = 0.75
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_MEMORY_ID_PATTERN = re.compile(r"mem_[0-9a-f]{32}")
_MEMORY_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_SCHEMA = """
BEGIN IMMEDIATE;

CREATE TABLE schema_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL
);

CREATE TABLE memories (
    memory_id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL CHECK (
        memory_type IN ('preference', 'goal', 'project', 'learning_progress')
    ),
    memory_key TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'superseded')),
    supersedes_id TEXT REFERENCES memories(memory_id) DEFERRABLE INITIALLY DEFERRED,
    source_hash TEXT,
    token_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX idx_memory_live_duplicate
ON memories(memory_type, memory_key, content_hash)
WHERE status IN ('pending', 'active');

CREATE INDEX idx_memories_status_type_key
ON memories(status, memory_type, memory_key);

CREATE TABLE memory_postings (
    token TEXT NOT NULL,
    memory_id TEXT NOT NULL
        REFERENCES memories(memory_id) ON DELETE CASCADE,
    frequency INTEGER NOT NULL,
    PRIMARY KEY (token, memory_id)
);

CREATE INDEX idx_memory_postings_token ON memory_postings(token);

CREATE TABLE memory_embeddings (
    memory_id TEXT PRIMARY KEY
        REFERENCES memories(memory_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE INDEX idx_memory_embeddings_model_id
ON memory_embeddings(model_id);

CREATE TABLE memory_rejections (
    proposal_hash TEXT PRIMARY KEY,
    rejected_at TEXT NOT NULL
);

INSERT INTO schema_metadata(singleton, version) VALUES (1, 1);

COMMIT;
"""


def utc_now() -> datetime:
    """Return the current timezone-aware UTC instant."""
    return datetime.now(UTC)


def _pack_vector(values: Sequence[float]) -> bytes:
    try:
        return struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error) as error:
        raise EmbeddingError("Embedding values cannot be stored as float32") from error


def _unpack_vector(value: bytes, *, dimensions: int) -> tuple[float, ...]:
    if dimensions <= 0:
        raise MemoryStorageError("Stored memory embedding dimension is invalid")
    if len(value) != dimensions * 4:
        raise MemoryStorageError(
            "Stored memory embedding byte length does not match its dimension"
        )
    try:
        values = tuple(struct.unpack(f"<{dimensions}f", value))
    except struct.error as error:
        raise MemoryStorageError("Stored memory embedding vector is invalid") from error
    if not all(isfinite(item) for item in values):
        raise MemoryStorageError("Stored memory embedding vector is invalid")
    return values


@dataclass(frozen=True, slots=True)
class _PreparedEmbedding:
    model_id: str
    dimensions: int
    vector: bytes


@dataclass(frozen=True, slots=True)
class _PreparedMemory:
    proposal: MemoryProposal
    embedding: _PreparedEmbedding


@dataclass(frozen=True, slots=True)
class _ReplacementNode:
    memory_id: str
    memory_type: MemoryType
    memory_key: str
    status: MemoryStatus
    supersedes_id: str | None


class SQLiteMemoryStore:
    """Persist memories and lifecycle transitions in a dedicated SQLite database."""

    def __init__(
        self,
        database_path: Path,
        embedding_model: EmbeddingModel,
        tokenizer: MixedLanguageTokenizer | None = None,
        *,
        clock: Callable[[], datetime] = utc_now,
        id_factory: Callable[[], str] = make_memory_id,
    ) -> None:
        self._database_path = database_path.expanduser().resolve()
        self._embedding_model = embedding_model
        self._tokenizer = tokenizer or MixedLanguageTokenizer()
        self._clock = clock
        self._id_factory = id_factory
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MemoryStorageError(
                "Could not create memory database directory"
            ) from error
        self._initialize_schema()

    def create_pending(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        source_hash: str,
    ) -> tuple[MemoryRecord, ...]:
        """Persist nonduplicate, nonrejected proposals as one pending batch."""
        self._validate_hash(source_hash, "source_hash")
        normalized = self._normalize_proposals(proposals)
        if not normalized:
            return ()

        with closing(self._open()) as connection:
            try:
                filtered = self._filter_pending(connection, normalized)
            except sqlite3.Error as error:
                raise MemoryStorageError(
                    "Could not inspect pending memories"
                ) from error
            except (TypeError, ValueError, OverflowError) as error:
                raise MemoryStorageError("Stored memory data is invalid") from error
        if not filtered:
            return ()

        prepared = self._prepare_memories(filtered)
        created_at = self._now()
        records: list[MemoryRecord] = []
        with closing(self._open()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                still_allowed = self._filter_pending(
                    connection,
                    [item.proposal for item in prepared],
                )
                allowed_hashes = {
                    memory_proposal_hash(
                        proposal.memory_type,
                        proposal.memory_key,
                        proposal.content,
                    )
                    for proposal in still_allowed
                }
                for item in prepared:
                    proposal_hash = memory_proposal_hash(
                        item.proposal.memory_type,
                        item.proposal.memory_key,
                        item.proposal.content,
                    )
                    if proposal_hash not in allowed_hashes:
                        continue
                    record = self._make_record(
                        item.proposal,
                        status=MemoryStatus.PENDING,
                        supersedes_id=None,
                        source_hash=source_hash,
                        created_at=created_at,
                    )
                    self._insert_memory(connection, record, item.embedding)
                    records.append(record)
                connection.commit()
            except sqlite3.Error as error:
                connection.rollback()
                raise MemoryStorageError(
                    "Could not persist pending memory batch"
                ) from error
            except (TypeError, ValueError, OverflowError) as error:
                connection.rollback()
                raise MemoryStorageError("Stored memory data is invalid") from error
            except Exception:
                connection.rollback()
                raise
        return tuple(records)

    def remember(
        self,
        memory_type: MemoryType,
        memory_key: str,
        content: str,
        *,
        action: MemoryActivationAction = MemoryActivationAction.ACTIVATE,
        replace_id: str | None = None,
    ) -> MemoryRecord:
        """Create one explicit active memory under atomic conflict rules."""
        proposal = self._normalize_proposal(memory_type, memory_key, content)
        prepared = self._prepare_memories((proposal,))[0]
        timestamp = self._now()

        with closing(self._open()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                supersedes_id = self._validate_activation(
                    connection,
                    proposal,
                    action=action,
                    replace_id=replace_id,
                    timestamp=timestamp,
                )
                record = self._make_record(
                    proposal,
                    status=MemoryStatus.ACTIVE,
                    supersedes_id=supersedes_id,
                    source_hash=None,
                    created_at=timestamp,
                )
                self._insert_memory(connection, record, prepared.embedding)
                connection.commit()
                return record
            except sqlite3.Error as error:
                connection.rollback()
                raise MemoryStorageError("Could not persist active memory") from error
            except (TypeError, ValueError, OverflowError) as error:
                connection.rollback()
                raise MemoryStorageError("Stored memory data is invalid") from error
            except Exception:
                connection.rollback()
                raise

    def approve(
        self,
        candidate_id: str,
        *,
        action: MemoryActivationAction,
        replace_id: str | None = None,
    ) -> MemoryRecord:
        """Activate one pending candidate."""
        timestamp = self._now()
        with closing(self._open()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT memory_id, memory_type, memory_key, content, status
                    FROM memories
                    WHERE memory_id = ?
                    """,
                    (candidate_id,),
                ).fetchone()
                if row is None:
                    raise MemoryNotFoundError("Memory candidate was not found")
                status = MemoryStatus(self._require_stored_str(row, "status"))
                if status is not MemoryStatus.PENDING:
                    raise InvalidMemoryError("Memory candidate is no longer pending")
                proposal = self._proposal_from_row(row)
                supersedes_id = self._validate_activation(
                    connection,
                    proposal,
                    action=action,
                    replace_id=replace_id,
                    timestamp=timestamp,
                )
                connection.execute(
                    """
                    UPDATE memories
                    SET status = 'active', supersedes_id = ?, updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (supersedes_id, timestamp.isoformat(), candidate_id),
                )
                approved = self._record_by_id(connection, candidate_id)
                connection.commit()
                return approved
            except sqlite3.Error as error:
                connection.rollback()
                raise MemoryStorageError(
                    "Could not approve memory candidate"
                ) from error
            except (TypeError, ValueError, OverflowError) as error:
                connection.rollback()
                raise MemoryStorageError("Stored memory data is invalid") from error
            except Exception:
                connection.rollback()
                raise

    def reject(self, candidate_id: str) -> None:
        """Reject one pending candidate and retain only its fingerprint."""
        timestamp = self._now()
        with closing(self._open()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT memory_id, memory_type, memory_key, content, status
                    FROM memories
                    WHERE memory_id = ?
                    """,
                    (candidate_id,),
                ).fetchone()
                if row is None:
                    raise MemoryNotFoundError("Memory candidate was not found")
                status = MemoryStatus(self._require_stored_str(row, "status"))
                if status is not MemoryStatus.PENDING:
                    raise InvalidMemoryError("Only a pending memory can be rejected")
                proposal = self._proposal_from_row(row)
                proposal_hash = memory_proposal_hash(
                    proposal.memory_type,
                    proposal.memory_key,
                    proposal.content,
                )
                connection.execute(
                    """
                    INSERT INTO memory_rejections(proposal_hash, rejected_at)
                    VALUES (?, ?)
                    ON CONFLICT(proposal_hash) DO UPDATE
                    SET rejected_at = excluded.rejected_at
                    """,
                    (proposal_hash, timestamp.isoformat()),
                )
                connection.execute(
                    "DELETE FROM memories WHERE memory_id = ?",
                    (candidate_id,),
                )
                connection.commit()
            except sqlite3.Error as error:
                connection.rollback()
                raise MemoryStorageError("Could not reject memory candidate") from error
            except (TypeError, ValueError, OverflowError) as error:
                connection.rollback()
                raise MemoryStorageError("Stored memory data is invalid") from error
            except Exception:
                connection.rollback()
                raise

    def forget(self, memory_id: str) -> int:
        """Delete one complete replacement chain."""
        with closing(self._open()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if (
                    connection.execute(
                        "SELECT 1 FROM memories WHERE memory_id = ?",
                        (memory_id,),
                    ).fetchone()
                    is None
                ):
                    raise MemoryNotFoundError("Memory was not found")

                rows = connection.execute(
                    """
                    WITH RECURSIVE replacement_chain(memory_id) AS (
                        SELECT ?
                        UNION
                        SELECT memories.supersedes_id
                        FROM memories
                        JOIN replacement_chain
                            ON memories.memory_id = replacement_chain.memory_id
                        WHERE memories.supersedes_id IS NOT NULL
                        UNION
                        SELECT memories.memory_id
                        FROM memories
                        JOIN replacement_chain
                            ON memories.supersedes_id = replacement_chain.memory_id
                    )
                    SELECT
                        memory_id,
                        memory_type,
                        memory_key,
                        content,
                        status,
                        supersedes_id
                    FROM memories
                    WHERE memory_id IN (SELECT memory_id FROM replacement_chain)
                    """,
                    (memory_id,),
                ).fetchall()
                chain_ids, proposal_hashes = self._validate_replacement_component(rows)
                placeholders = ", ".join("?" for _ in chain_ids)
                connection.execute(
                    f"DELETE FROM memory_postings WHERE memory_id IN ({placeholders})",
                    chain_ids,
                )
                connection.execute(
                    "DELETE FROM memory_embeddings "
                    f"WHERE memory_id IN ({placeholders})",
                    chain_ids,
                )
                connection.execute(
                    f"DELETE FROM memories WHERE memory_id IN ({placeholders})",
                    chain_ids,
                )
                if proposal_hashes:
                    hash_placeholders = ", ".join("?" for _ in proposal_hashes)
                    connection.execute(
                        "DELETE FROM memory_rejections "
                        f"WHERE proposal_hash IN ({hash_placeholders})",
                        proposal_hashes,
                    )
                connection.commit()
                return len(chain_ids)
            except sqlite3.Error as error:
                connection.rollback()
                raise MemoryStorageError("Could not forget memory chain") from error
            except (TypeError, ValueError, OverflowError) as error:
                connection.rollback()
                raise MemoryStorageError("Stored memory data is invalid") from error
            except Exception:
                connection.rollback()
                raise

    def list_memories(
        self,
        *,
        status: MemoryStatus | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        """Return optionally filtered memories, newest first."""
        if status is not None and not isinstance(status, MemoryStatus):
            raise InvalidMemoryError("status must be a MemoryStatus")
        if memory_type is not None and not isinstance(memory_type, MemoryType):
            raise InvalidMemoryError("memory_type must be a MemoryType")

        clauses: list[str] = []
        parameters: list[str] = []
        if status is not None:
            clauses.append("m.status = ?")
            parameters.append(status.value)
        if memory_type is not None:
            clauses.append("m.memory_type = ?")
            parameters.append(memory_type.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT
                m.memory_id,
                m.memory_type,
                m.memory_key,
                m.content,
                m.content_hash,
                m.status,
                m.supersedes_id,
                m.source_hash,
                m.created_at,
                m.updated_at,
                e.model_id AS embedding_model_id,
                e.dimensions AS embedding_dimensions,
                e.vector AS embedding_vector
            FROM memories AS m
            LEFT JOIN memory_embeddings AS e ON e.memory_id = m.memory_id
            {where}
            ORDER BY m.created_at DESC, m.memory_id DESC
        """
        with closing(self._open()) as connection:
            try:
                rows = connection.execute(query, parameters).fetchall()
                return [self._record_from_row(row) for row in rows]
            except sqlite3.Error as error:
                raise MemoryStorageError("Could not list stored memories") from error
            except (TypeError, ValueError, OverflowError) as error:
                raise MemoryStorageError("Stored memory data is invalid") from error

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
    ) -> list[MemorySearchResult]:
        """Return hybrid-ranked active memories matching a query."""
        query_tokens = list(dict.fromkeys(self._tokenizer.tokenize(query)))
        if not query_tokens:
            raise InvalidMemoryError("Query contains no searchable terms")
        if not 1 <= limit <= 20:
            raise InvalidMemoryError("Memory search limit must be between 1 and 20")
        if memory_type is not None and not isinstance(memory_type, MemoryType):
            raise InvalidMemoryError("memory_type must be a MemoryType")

        type_clause = " AND m.memory_type = ?" if memory_type is not None else ""
        type_parameters = [memory_type.value] if memory_type is not None else []
        with closing(self._open()) as connection:
            try:
                connection.execute("BEGIN")
                statistics = connection.execute(
                    "SELECT COUNT(*), COALESCE(AVG(m.token_count), 0) "
                    "FROM memories AS m WHERE m.status = 'active'" + type_clause,
                    type_parameters,
                ).fetchone()
                if statistics is None:
                    raise MemoryStorageError("Stored memory statistics are invalid")
                memory_count = int(statistics[0])
                average_length = float(statistics[1])
                if memory_count == 0:
                    return []

                candidate_limit = max(20, limit * 4)
                lexical_ids = self._lexical_memory_ids(
                    connection,
                    query_tokens,
                    memory_count=memory_count,
                    average_length=average_length,
                    memory_type=memory_type,
                    limit=candidate_limit,
                )
                semantic_ids, semantic_scores = self._semantic_memory_ranking(
                    connection,
                    query,
                    memory_type=memory_type,
                    limit=candidate_limit,
                )
                fused = reciprocal_rank_scores(
                    lexical_ids,
                    semantic_ids,
                    limit=limit,
                )
                return [
                    MemorySearchResult(
                        memory=self._record_by_id(connection, memory_id),
                        score=score,
                        semantic_similarity=semantic_scores[memory_id],
                    )
                    for memory_id, score in fused
                ]
            except sqlite3.Error as error:
                raise MemoryStorageError("Could not search stored memories") from error
            except (TypeError, ValueError, OverflowError) as error:
                raise MemoryStorageError("Stored memory data is invalid") from error

    def _lexical_memory_ids(
        self,
        connection: sqlite3.Connection,
        query_tokens: list[str],
        *,
        memory_count: int,
        average_length: float,
        memory_type: MemoryType | None,
        limit: int,
    ) -> list[str]:
        placeholders = ", ".join("?" for _ in query_tokens)
        type_clause = " AND m.memory_type = ?" if memory_type is not None else ""
        parameters = [*query_tokens]
        if memory_type is not None:
            parameters.append(memory_type.value)
        rows = connection.execute(
            f"""
            SELECT p.token, p.frequency, m.memory_id, m.token_count
            FROM memory_postings AS p
            JOIN memories AS m ON m.memory_id = p.memory_id
            WHERE p.token IN ({placeholders})
                AND m.status = 'active'
                {type_clause}
            """,
            parameters,
        ).fetchall()
        document_frequencies = Counter(
            self._require_stored_str(row, "token") for row in rows
        )
        scores: dict[str, float] = {}
        safe_average = average_length if average_length > 0 else 1.0
        for row in rows:
            token = self._require_stored_str(row, "token")
            memory_id = self._require_stored_memory_id(row, "memory_id")
            frequency = row["frequency"]
            document_length = row["token_count"]
            if type(frequency) is not int or frequency <= 0:
                raise MemoryStorageError("Stored memory posting frequency is invalid")
            if type(document_length) is not int or document_length < 0:
                raise MemoryStorageError("Stored memory token count is invalid")
            document_frequency = document_frequencies[token]
            inverse_frequency = log(
                1
                + (memory_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + _BM25_K1 * (
                1 - _BM25_B + _BM25_B * document_length / safe_average
            )
            scores[memory_id] = scores.get(memory_id, 0.0) + (
                inverse_frequency * frequency * (_BM25_K1 + 1) / denominator
            )

        return [
            memory_id
            for memory_id, score in sorted(
                scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:limit]
            if score > 0
        ]

    def _semantic_memory_ranking(
        self,
        connection: sqlite3.Connection,
        query: str,
        *,
        memory_type: MemoryType | None,
        limit: int,
    ) -> tuple[list[str], dict[str, float]]:
        query_embedding = self._embedding_model.embed_query(query)
        if query_embedding.model_id != self._embedding_model.model_id:
            raise EmbeddingError(
                "Query embedding model identifier does not match configured model"
            )

        type_clause = " AND m.memory_type = ?" if memory_type is not None else ""
        parameters = [memory_type.value] if memory_type is not None else []
        rows = connection.execute(
            """
            SELECT
                m.memory_id,
                e.model_id,
                e.dimensions,
                e.vector
            FROM memories AS m
            LEFT JOIN memory_embeddings AS e ON e.memory_id = m.memory_id
            WHERE m.status = 'active'
            """
            + type_clause,
            parameters,
        ).fetchall()

        scores: dict[str, float] = {}
        query_dimensions = len(query_embedding.values)
        for row in rows:
            memory_id = self._require_stored_memory_id(row, "memory_id")
            model_id = row["model_id"]
            if (
                not isinstance(model_id, str)
                or model_id != self._embedding_model.model_id
            ):
                raise MemoryStorageError(
                    "Stored memory embedding model is incompatible"
                )
            dimensions = row["dimensions"]
            if type(dimensions) is not int or dimensions != query_dimensions:
                raise MemoryStorageError(
                    "Stored memory embedding dimension is incompatible"
                )
            vector_value = row["vector"]
            if not isinstance(vector_value, (bytes, bytearray, memoryview)):
                raise MemoryStorageError("Stored memory embedding vector is not a BLOB")
            vector = _unpack_vector(bytes(vector_value), dimensions=dimensions)
            score = fsum(
                query_value * stored_value
                for query_value, stored_value in zip(
                    query_embedding.values,
                    vector,
                    strict=True,
                )
            )
            scores[memory_id] = score

        ranked = sorted(
            ((memory_id, score) for memory_id, score in scores.items() if score > 0),
            key=lambda item: (-item[1], item[0]),
        )
        return [memory_id for memory_id, _ in ranked[:limit]], scores

    def _initialize_schema(self) -> None:
        with closing(self._open()) as connection:
            try:
                has_metadata = (
                    connection.execute(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'schema_metadata'"
                    ).fetchone()
                    is not None
                )
                if not has_metadata:
                    connection.executescript(_SCHEMA)
                    connection.commit()
                    return

                rows = connection.execute(
                    "SELECT singleton, version FROM schema_metadata"
                ).fetchall()
                if len(rows) != 1:
                    raise MemoryStorageError("Memory schema metadata is invalid")
                singleton = rows[0]["singleton"]
                version = rows[0]["version"]
                if (
                    type(singleton) is not int
                    or singleton != 1
                    or type(version) is not int
                ):
                    raise MemoryStorageError("Memory schema metadata is invalid")
                if version != _SCHEMA_VERSION:
                    raise MemoryStorageError(
                        f"Unsupported memory schema version {version}; "
                        f"expected {_SCHEMA_VERSION}"
                    )
            except sqlite3.Error as error:
                connection.rollback()
                raise MemoryStorageError(
                    "Could not initialize memory database"
                ) from error
            except (TypeError, ValueError, OverflowError) as error:
                raise MemoryStorageError("Memory schema metadata is invalid") from error

    def _open(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self._database_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as error:
            raise MemoryStorageError("Could not open memory database") from error

    @staticmethod
    def _normalize_proposal(
        memory_type: MemoryType,
        memory_key: str,
        content: str,
    ) -> MemoryProposal:
        try:
            return MemoryProposal(memory_type, memory_key, content)
        except (TypeError, ValueError) as error:
            raise InvalidMemoryError(str(error)) from error

    def _normalize_proposals(
        self,
        proposals: Sequence[MemoryProposal],
    ) -> tuple[MemoryProposal, ...]:
        normalized: list[MemoryProposal] = []
        seen: set[str] = set()
        for proposal in proposals:
            if not isinstance(proposal, MemoryProposal):
                raise InvalidMemoryError("proposals must contain MemoryProposal values")
            candidate = self._normalize_proposal(
                proposal.memory_type,
                proposal.memory_key,
                proposal.content,
            )
            fingerprint = memory_proposal_hash(
                candidate.memory_type,
                candidate.memory_key,
                candidate.content,
            )
            if fingerprint not in seen:
                normalized.append(candidate)
                seen.add(fingerprint)
        return tuple(normalized)

    @classmethod
    def _filter_pending(
        cls,
        connection: sqlite3.Connection,
        proposals: Sequence[MemoryProposal],
    ) -> tuple[MemoryProposal, ...]:
        live: set[tuple[str, str, str]] = set()
        for row in connection.execute(
            "SELECT memory_type, memory_key, content, content_hash FROM memories "
            "WHERE status IN ('pending', 'active')"
        ).fetchall():
            memory_type = MemoryType(cls._require_stored_str(row, "memory_type"))
            memory_key = cls._require_stored_str(row, "memory_key")
            if _MEMORY_KEY_PATTERN.fullmatch(memory_key) is None:
                raise ValueError("stored memory_key is invalid")
            content = cls._require_stored_str(row, "content")
            content_hash = cls._require_stored_hash(row, "content_hash")
            if content_hash != memory_content_hash(content):
                raise ValueError("stored content_hash does not match content")
            live.add((memory_type.value, memory_key, content_hash))

        rejected = {
            cls._require_stored_hash(row, "proposal_hash")
            for row in connection.execute(
                "SELECT proposal_hash FROM memory_rejections"
            ).fetchall()
        }
        return tuple(
            proposal
            for proposal in proposals
            if (
                proposal.memory_type.value,
                proposal.memory_key,
                memory_content_hash(proposal.content),
            )
            not in live
            and memory_proposal_hash(
                proposal.memory_type,
                proposal.memory_key,
                proposal.content,
            )
            not in rejected
        )

    def _prepare_memories(
        self,
        proposals: Sequence[MemoryProposal],
    ) -> tuple[_PreparedMemory, ...]:
        texts = [proposal.content for proposal in proposals]
        embeddings = self._embedding_model.embed_documents(texts)
        if len(embeddings) != len(proposals):
            raise EmbeddingError(
                f"Embedding model returned {len(embeddings)} vectors for "
                f"{len(proposals)} memories"
            )

        dimensions: set[int] = set()
        prepared: list[_PreparedMemory] = []
        for proposal, embedding in zip(proposals, embeddings, strict=True):
            if embedding.model_id != self._embedding_model.model_id:
                raise EmbeddingError(
                    "Embedding model identifier does not match configured model"
                )
            dimension = len(embedding.values)
            dimensions.add(dimension)
            prepared.append(
                _PreparedMemory(
                    proposal,
                    _PreparedEmbedding(
                        embedding.model_id,
                        dimension,
                        _pack_vector(embedding.values),
                    ),
                )
            )
        if len(dimensions) > 1:
            raise EmbeddingError("Memory embeddings must have one common dimension")
        return tuple(prepared)

    def _make_record(
        self,
        proposal: MemoryProposal,
        *,
        status: MemoryStatus,
        supersedes_id: str | None,
        source_hash: str | None,
        created_at: datetime,
    ) -> MemoryRecord:
        try:
            memory_id = self._id_factory()
            return MemoryRecord(
                memory_id=memory_id,
                memory_type=proposal.memory_type,
                memory_key=proposal.memory_key,
                content=proposal.content,
                content_hash=memory_content_hash(proposal.content),
                status=status,
                supersedes_id=supersedes_id,
                source_hash=source_hash,
                created_at=created_at,
                updated_at=created_at,
            )
        except (TypeError, ValueError) as error:
            raise MemoryStorageError(
                "Memory ID factory returned invalid data"
            ) from error

    def _insert_memory(
        self,
        connection: sqlite3.Connection,
        record: MemoryRecord,
        embedding: _PreparedEmbedding,
    ) -> None:
        frequencies = Counter(self._tokenizer.tokenize(record.content))
        connection.execute(
            """
            INSERT INTO memories(
                memory_id,
                memory_type,
                memory_key,
                content,
                content_hash,
                status,
                supersedes_id,
                source_hash,
                token_count,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.memory_id,
                record.memory_type.value,
                record.memory_key,
                record.content,
                record.content_hash,
                record.status.value,
                record.supersedes_id,
                record.source_hash,
                sum(frequencies.values()),
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
            ),
        )
        connection.executemany(
            "INSERT INTO memory_postings(token, memory_id, frequency) VALUES (?, ?, ?)",
            (
                (token, record.memory_id, frequency)
                for token, frequency in frequencies.items()
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_embeddings(memory_id, model_id, dimensions, vector)
            VALUES (?, ?, ?, ?)
            """,
            (
                record.memory_id,
                embedding.model_id,
                embedding.dimensions,
                embedding.vector,
            ),
        )

    def _validate_activation(
        self,
        connection: sqlite3.Connection,
        proposal: MemoryProposal,
        *,
        action: MemoryActivationAction,
        replace_id: str | None,
        timestamp: datetime,
    ) -> str | None:
        if not isinstance(action, MemoryActivationAction):
            raise InvalidMemoryError("action must be a MemoryActivationAction")
        if action is MemoryActivationAction.REPLACE:
            if replace_id is None or not replace_id.strip():
                raise InvalidMemoryError("REPLACE requires a non-blank replace_id")
        elif replace_id is not None:
            raise InvalidMemoryError(f"{action.name} does not accept replace_id")

        conflicts = [
            self._require_stored_memory_id(row, "memory_id")
            for row in connection.execute(
                "SELECT memory_id FROM memories "
                "WHERE memory_type = ? AND memory_key = ? AND status = 'active' "
                "ORDER BY memory_id",
                (proposal.memory_type.value, proposal.memory_key),
            ).fetchall()
        ]
        if action is MemoryActivationAction.ACTIVATE:
            if conflicts:
                joined = ", ".join(conflicts)
                raise MemoryConflictError(
                    f"Active memory conflict; choose REPLACE or KEEP_BOTH. "
                    f"Active IDs: {joined}"
                )
            return None
        if action is MemoryActivationAction.KEEP_BOTH:
            if not conflicts:
                raise MemoryConflictError(
                    "KEEP_BOTH requires an active memory in the same slot"
                )
            return None

        assert replace_id is not None
        if replace_id not in conflicts:
            raise MemoryConflictError(
                "Replacement target must be active in the same memory slot"
            )
        connection.execute(
            "UPDATE memories SET status = 'superseded', updated_at = ? "
            "WHERE memory_id = ?",
            (timestamp.isoformat(), replace_id),
        )
        return replace_id

    @classmethod
    def _validate_replacement_component(
        cls,
        rows: Sequence[sqlite3.Row],
    ) -> tuple[list[str], list[str]]:
        error_message = "Stored memory replacement lifecycle is invalid"
        nodes: dict[str, _ReplacementNode] = {}
        proposal_hashes: list[str] = []
        for row in rows:
            memory_id = cls._require_stored_memory_id(row, "memory_id")
            memory_type = MemoryType(cls._require_stored_str(row, "memory_type"))
            memory_key = cls._require_stored_str(row, "memory_key")
            if _MEMORY_KEY_PATTERN.fullmatch(memory_key) is None:
                raise MemoryStorageError(error_message)
            status = MemoryStatus(cls._require_stored_str(row, "status"))
            supersedes_id = cls._optional_stored_str(row, "supersedes_id")
            if supersedes_id is not None:
                if _MEMORY_ID_PATTERN.fullmatch(supersedes_id) is None:
                    raise MemoryStorageError(error_message)
            if memory_id in nodes:
                raise MemoryStorageError(error_message)
            nodes[memory_id] = _ReplacementNode(
                memory_id=memory_id,
                memory_type=memory_type,
                memory_key=memory_key,
                status=status,
                supersedes_id=supersedes_id,
            )
            proposal_hashes.append(
                memory_proposal_hash(
                    memory_type,
                    memory_key,
                    cls._require_stored_str(row, "content"),
                )
            )

        if not nodes:
            raise MemoryStorageError(error_message)

        replacement_counts = dict.fromkeys(nodes, 0)
        edge_count = 0
        for node in nodes.values():
            if node.supersedes_id is None:
                continue
            edge_count += 1
            replaced = nodes.get(node.supersedes_id)
            if replaced is None:
                raise MemoryStorageError(error_message)
            if (
                node.memory_type,
                node.memory_key,
            ) != (
                replaced.memory_type,
                replaced.memory_key,
            ):
                raise MemoryStorageError(error_message)
            if (
                node.status not in (MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED)
                or replaced.status is not MemoryStatus.SUPERSEDED
            ):
                raise MemoryStorageError(error_message)
            replacement_counts[replaced.memory_id] += 1
            if replacement_counts[replaced.memory_id] > 1:
                raise MemoryStorageError(error_message)

        if len(nodes) == 1:
            node = next(iter(nodes.values()))
            if edge_count != 0 or node.status is MemoryStatus.SUPERSEDED:
                raise MemoryStorageError(error_message)
            return [node.memory_id], proposal_hashes

        if edge_count != len(nodes) - 1:
            raise MemoryStorageError(error_message)
        heads = [
            node for node in nodes.values() if replacement_counts[node.memory_id] == 0
        ]
        if len(heads) != 1 or heads[0].status is not MemoryStatus.ACTIVE:
            raise MemoryStorageError(error_message)
        if any(
            node.status is not MemoryStatus.SUPERSEDED
            for node in nodes.values()
            if node.memory_id != heads[0].memory_id
        ):
            raise MemoryStorageError(error_message)

        ordered_ids: list[str] = []
        current: _ReplacementNode | None = heads[0]
        while current is not None:
            if current.memory_id in ordered_ids:
                raise MemoryStorageError(error_message)
            ordered_ids.append(current.memory_id)
            current = (
                nodes[current.supersedes_id]
                if current.supersedes_id is not None
                else None
            )
        if len(ordered_ids) != len(nodes):
            raise MemoryStorageError(error_message)
        return ordered_ids, proposal_hashes

    def _record_from_row(self, row: sqlite3.Row) -> MemoryRecord:
        model_id = row["embedding_model_id"]
        dimensions = row["embedding_dimensions"]
        vector = row["embedding_vector"]
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("stored embedding model is invalid")
        if not isinstance(dimensions, int):
            raise ValueError("stored embedding dimension is invalid")
        if not isinstance(vector, (bytes, bytearray, memoryview)):
            raise ValueError("stored embedding vector is not a BLOB")
        _unpack_vector(bytes(vector), dimensions=dimensions)

        return MemoryRecord(
            memory_id=self._require_stored_memory_id(row, "memory_id"),
            memory_type=MemoryType(self._require_stored_str(row, "memory_type")),
            memory_key=self._require_stored_str(row, "memory_key"),
            content=self._require_stored_str(row, "content"),
            content_hash=self._require_stored_str(row, "content_hash"),
            status=MemoryStatus(self._require_stored_str(row, "status")),
            supersedes_id=self._optional_stored_str(row, "supersedes_id"),
            source_hash=self._optional_stored_str(row, "source_hash"),
            created_at=datetime.fromisoformat(
                self._require_stored_str(row, "created_at")
            ),
            updated_at=datetime.fromisoformat(
                self._require_stored_str(row, "updated_at")
            ),
        )

    def _record_by_id(
        self,
        connection: sqlite3.Connection,
        memory_id: str,
    ) -> MemoryRecord:
        row = connection.execute(
            """
            SELECT
                m.memory_id,
                m.memory_type,
                m.memory_key,
                m.content,
                m.content_hash,
                m.status,
                m.supersedes_id,
                m.source_hash,
                m.created_at,
                m.updated_at,
                e.model_id AS embedding_model_id,
                e.dimensions AS embedding_dimensions,
                e.vector AS embedding_vector
            FROM memories AS m
            LEFT JOIN memory_embeddings AS e ON e.memory_id = m.memory_id
            WHERE m.memory_id = ?
            """,
            (memory_id,),
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError("Memory was not found")
        return self._record_from_row(row)

    def _proposal_from_row(self, row: sqlite3.Row) -> MemoryProposal:
        return MemoryProposal(
            MemoryType(self._require_stored_str(row, "memory_type")),
            self._require_stored_str(row, "memory_key"),
            self._require_stored_str(row, "content"),
        )

    @staticmethod
    def _require_stored_str(row: sqlite3.Row, field_name: str) -> str:
        value = row[field_name]
        if not isinstance(value, str):
            raise ValueError(f"stored {field_name} is not text")
        return value

    @classmethod
    def _require_stored_hash(cls, row: sqlite3.Row, field_name: str) -> str:
        value = cls._require_stored_str(row, field_name)
        if _HASH_PATTERN.fullmatch(value) is None:
            raise ValueError(f"stored {field_name} is invalid")
        return value

    @classmethod
    def _require_stored_memory_id(cls, row: sqlite3.Row, field_name: str) -> str:
        value = cls._require_stored_str(row, field_name)
        if _MEMORY_ID_PATTERN.fullmatch(value) is None:
            raise ValueError(f"stored {field_name} is invalid")
        return value

    @classmethod
    def _optional_stored_str(
        cls,
        row: sqlite3.Row,
        field_name: str,
    ) -> str | None:
        value = row[field_name]
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"stored {field_name} is not text")
        return value

    def _now(self) -> datetime:
        try:
            value = self._clock()
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError
            return value
        except (AttributeError, TypeError, ValueError) as error:
            raise MemoryStorageError(
                "Memory clock must return a timezone-aware UTC datetime"
            ) from error

    @staticmethod
    def _validate_hash(value: str, field_name: str) -> None:
        if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
            raise InvalidMemoryError(f"{field_name} must be a lowercase SHA-256 hash")

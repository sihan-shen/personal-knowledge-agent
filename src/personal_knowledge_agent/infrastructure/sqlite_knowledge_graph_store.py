"""Provenance-aware SQLite persistence and local queries for knowledge graphs."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import UTC, datetime
from math import log
from pathlib import Path

from personal_knowledge_agent.domain.errors import (
    GraphEmptyError,
    GraphEntityNotFoundError,
    GraphStorageError,
    InvalidGraphError,
)
from personal_knowledge_agent.domain.graph import (
    EntityNeighborhood,
    EntitySearchResult,
    EntityType,
    GraphBuildResult,
    GraphChunkExtraction,
    GraphEntity,
    GraphNeighbor,
    GraphProvenance,
    GraphRelation,
    GraphSourceDocument,
    RelationType,
    make_entity_id,
    make_relation_id,
    normalize_entity_name,
)
from personal_knowledge_agent.infrastructure.mixed_tokenizer import (
    MixedLanguageTokenizer,
)

_SCHEMA_VERSION = "1"
_BM25_K1 = 1.5
_BM25_B = 0.75

_SCHEMA = """
CREATE TABLE schema_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version TEXT NOT NULL
);

CREATE TABLE graph_documents (
    source TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    built_at TEXT NOT NULL
);

CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    UNIQUE (entity_type, normalized_name)
);

CREATE TABLE entity_mentions (
    source TEXT NOT NULL REFERENCES graph_documents(source) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    chunk_position INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    mention_name TEXT NOT NULL,
    PRIMARY KEY (source, chunk_id, entity_id)
);

CREATE TABLE relations (
    relation_id TEXT PRIMARY KEY,
    relation_type TEXT NOT NULL,
    source_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    target_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    UNIQUE (relation_type, source_entity_id, target_entity_id)
);

CREATE TABLE relation_evidence (
    relation_id TEXT NOT NULL REFERENCES relations(relation_id) ON DELETE CASCADE,
    source TEXT NOT NULL REFERENCES graph_documents(source) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    chunk_position INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    evidence_quote TEXT NOT NULL,
    PRIMARY KEY (relation_id, source, chunk_id, evidence_quote)
);

CREATE TABLE entity_postings (
    token TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    frequency INTEGER NOT NULL CHECK (frequency > 0),
    PRIMARY KEY (token, entity_id)
);

CREATE INDEX idx_entity_postings_token ON entity_postings(token);
CREATE INDEX idx_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX idx_relation_evidence_relation ON relation_evidence(relation_id);
CREATE INDEX idx_relations_source ON relations(source_entity_id);
CREATE INDEX idx_relations_target ON relations(target_entity_id);

INSERT INTO schema_metadata(singleton, version) VALUES (1, '1');
"""

type _MentionRow = tuple[str, str, int, str, str, str]
type _EvidenceRow = tuple[str, str, str, int, str, str]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SQLiteKnowledgeGraphStore:
    """Persist exact graph identities and source-scoped provenance atomically."""

    def __init__(
        self,
        database_path: Path,
        tokenizer: MixedLanguageTokenizer | None = None,
        *,
        clock: Callable[[], datetime] = _utc_now,
        create_if_missing: bool = True,
    ) -> None:
        self._database_path = database_path.expanduser().resolve()
        self._tokenizer = tokenizer or MixedLanguageTokenizer()
        self._clock = clock
        if create_if_missing:
            try:
                self._database_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise GraphStorageError(
                    "Could not create graph database directory"
                ) from error
            self._initialize_schema()
        elif self._database_path.is_file():
            self._initialize_schema()

    def replace_document(
        self,
        source_document: GraphSourceDocument,
        extractions: Sequence[GraphChunkExtraction],
    ) -> GraphBuildResult:
        """Replace one source contribution without disturbing shared graph facts."""
        prepared = self._prepare_replacement(source_document, tuple(extractions))
        source = str(source_document.source)
        try:
            with closing(self._open()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM graph_documents WHERE source = ?", (source,)
                )
                connection.execute(
                    "INSERT INTO graph_documents("
                    "source, document_id, content_hash, built_at"
                    ") "
                    "VALUES (?, ?, ?, ?)",
                    (
                        source,
                        source_document.document_id,
                        source_document.content_hash,
                        self._clock().astimezone(UTC).isoformat(),
                    ),
                )
                self._write_entities(connection, prepared.entities, prepared.mentions)
                self._write_relations(connection, prepared.relations, prepared.evidence)
                self._remove_orphans(connection)
                self._recompute_display_names(connection)
                self._rebuild_postings(connection)
                connection.commit()
        except sqlite3.Error as error:
            raise GraphStorageError(
                "Could not replace graph document contribution"
            ) from error
        except (TypeError, ValueError, OverflowError) as error:
            raise GraphStorageError("Graph replacement data is invalid") from error

        return GraphBuildResult(
            source=source,
            chunks=len(extractions),
            entities=len(prepared.document_entity_ids),
            relations=len(prepared.document_relation_ids),
            evidence=len(prepared.evidence),
        )

    def search_entities(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[EntitySearchResult]:
        """Return deterministic local BM25 entity matches."""
        tokens = tuple(dict.fromkeys(self._tokenizer.tokenize(query)))
        if not tokens:
            raise InvalidGraphError("Graph search query contains no searchable terms")
        try:
            with closing(self._open()) as connection:
                statistics = connection.execute(
                    "SELECT COUNT(*), COALESCE(AVG(token_count), 0) FROM entities"
                ).fetchone()
                if statistics is None:
                    raise GraphStorageError("Graph entity statistics are invalid")
                entity_count = self._require_int(statistics[0], "entity count")
                average_length = float(statistics[1])
                if entity_count == 0:
                    raise GraphEmptyError(
                        "Knowledge graph is empty. Build a document graph first."
                    )
                scores = self._bm25_scores(
                    connection, tokens, entity_count, average_length
                )
                if not scores:
                    return []
                placeholders = ", ".join("?" for _ in scores)
                rows = connection.execute(
                    "SELECT entity_id, entity_type, display_name FROM entities "
                    f"WHERE entity_id IN ({placeholders})",
                    tuple(scores),
                ).fetchall()
                results = [
                    self._search_result_from_row(
                        row,
                        scores,
                    )
                    for row in rows
                ]
        except sqlite3.Error as error:
            raise GraphStorageError("Could not search graph entities") from error
        except (TypeError, ValueError, OverflowError) as error:
            raise GraphStorageError("Stored graph data is invalid") from error

        return sorted(results, key=lambda item: (-item.score, item.entity_id))[:limit]

    def get_neighborhood(self, entity_id: str) -> EntityNeighborhood:
        """Return a complete one-hop neighborhood and every evidence row."""
        try:
            with closing(self._open()) as connection:
                entity_row = connection.execute(
                    "SELECT entity_id, entity_type, display_name FROM entities "
                    "WHERE entity_id = ?",
                    (entity_id,),
                ).fetchone()
                if entity_row is None:
                    entity_count_row = connection.execute(
                        "SELECT COUNT(*) FROM entities"
                    ).fetchone()
                    if entity_count_row is None:
                        raise GraphStorageError("Graph entity statistics are invalid")
                    if self._require_int(entity_count_row[0], "entity count") == 0:
                        raise GraphEmptyError(
                            "Knowledge graph is empty. Build a document graph first."
                        )
                    raise GraphEntityNotFoundError("Graph entity was not found")
                entity = self._entity_from_row(entity_row)
                outgoing = self._neighbors(connection, entity_id, outgoing=True)
                incoming = self._neighbors(connection, entity_id, outgoing=False)
        except sqlite3.Error as error:
            raise GraphStorageError("Could not read graph neighborhood") from error
        except (TypeError, ValueError, OverflowError) as error:
            raise GraphStorageError("Stored graph data is invalid") from error
        return EntityNeighborhood(entity, outgoing, incoming)

    def _initialize_schema(self) -> None:
        try:
            with closing(self._open(create=True)) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                if not tables:
                    connection.executescript(_SCHEMA)
                    return
                required = {
                    "schema_metadata",
                    "graph_documents",
                    "entities",
                    "entity_mentions",
                    "relations",
                    "relation_evidence",
                    "entity_postings",
                }
                if not required.issubset(tables):
                    raise GraphStorageError(
                        "Existing graph database schema is incompatible; "
                        "remove and rebuild it"
                    )
                row = connection.execute(
                    "SELECT version FROM schema_metadata WHERE singleton = 1"
                ).fetchone()
                if row is None or row[0] != _SCHEMA_VERSION:
                    raise GraphStorageError(
                        "Graph database schema version is incompatible"
                    )
        except sqlite3.Error as error:
            raise GraphStorageError("Could not initialize graph database") from error

    def _open(self, *, create: bool = False) -> sqlite3.Connection:
        if not create and not self._database_path.is_file():
            raise GraphEmptyError(
                "Knowledge graph is empty. Build a document graph first."
            )
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _prepare_replacement(
        self,
        source_document: GraphSourceDocument,
        extractions: tuple[GraphChunkExtraction, ...],
    ) -> _PreparedReplacement:
        source_chunks = {chunk.chunk_id: chunk for chunk in source_document.chunks}
        if len(extractions) != len(source_chunks):
            raise GraphStorageError(
                "Graph extraction count does not match indexed chunks"
            )

        entities: dict[str, GraphEntity] = {}
        mentions: list[_MentionRow] = []
        relations: dict[str, tuple[RelationType, str, str]] = {}
        evidence: list[_EvidenceRow] = []
        source = str(source_document.source)
        seen_chunks: set[str] = set()
        document_entity_ids: set[str] = set()
        document_relation_ids: set[str] = set()

        for extraction in extractions:
            chunk = source_chunks.get(extraction.chunk.chunk_id)
            if chunk != extraction.chunk or chunk.chunk_id in seen_chunks:
                raise GraphStorageError(
                    "Graph extraction does not match indexed source chunks"
                )
            seen_chunks.add(chunk.chunk_id)
            chunk_entity_ids = {entity.entity_id for entity in extraction.entities}
            for entity in extraction.entities:
                normalized = normalize_entity_name(entity.display_name)
                if not normalized or entity.entity_id != make_entity_id(
                    entity.entity_type, normalized
                ):
                    raise GraphStorageError("Graph entity identity is invalid")
                existing = entities.get(entity.entity_id)
                if (
                    existing is not None
                    and existing.entity_type is not entity.entity_type
                ):
                    raise GraphStorageError(
                        "Graph entity identity conflicts within source"
                    )
                entities.setdefault(entity.entity_id, entity)
                mentions.append(
                    (
                        source,
                        chunk.chunk_id,
                        chunk.position,
                        json.dumps(chunk.heading_path, ensure_ascii=False),
                        entity.entity_id,
                        entity.display_name,
                    )
                )
                document_entity_ids.add(entity.entity_id)

            for item in extraction.relation_evidence:
                relation = item.relation
                if (
                    relation.source_entity_id not in chunk_entity_ids
                    or relation.target_entity_id not in chunk_entity_ids
                ):
                    raise GraphStorageError(
                        "Graph relation endpoint is absent from its chunk"
                    )
                expected_relation_id = make_relation_id(
                    relation.relation_type,
                    relation.source_entity_id,
                    relation.target_entity_id,
                )
                if relation.relation_id != expected_relation_id:
                    raise GraphStorageError("Graph relation identity is invalid")
                relations.setdefault(
                    relation.relation_id,
                    (
                        relation.relation_type,
                        relation.source_entity_id,
                        relation.target_entity_id,
                    ),
                )
                evidence.append(
                    (
                        relation.relation_id,
                        source,
                        chunk.chunk_id,
                        chunk.position,
                        json.dumps(chunk.heading_path, ensure_ascii=False),
                        item.evidence_quote,
                    )
                )
                document_relation_ids.add(relation.relation_id)

        if seen_chunks != set(source_chunks):
            raise GraphStorageError(
                "Graph extraction does not cover every indexed chunk"
            )
        return _PreparedReplacement(
            entities,
            mentions,
            relations,
            evidence,
            document_entity_ids,
            document_relation_ids,
        )

    def _write_entities(
        self,
        connection: sqlite3.Connection,
        entities: dict[str, GraphEntity],
        mentions: list[_MentionRow],
    ) -> None:
        for entity in entities.values():
            normalized = normalize_entity_name(entity.display_name)
            tokens = self._tokenizer.tokenize(f"{entity.display_name} {normalized}")
            if not tokens:
                raise GraphStorageError("Graph entity contains no searchable terms")
            connection.execute(
                "INSERT INTO entities("
                "entity_id, entity_type, normalized_name, display_name, token_count"
                ") "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(entity_id) DO UPDATE SET "
                "entity_type = excluded.entity_type, "
                "normalized_name = excluded.normalized_name",
                (
                    entity.entity_id,
                    entity.entity_type.value,
                    normalized,
                    entity.display_name,
                    len(tokens),
                ),
            )
        for (
            source,
            chunk_id,
            position,
            heading_path,
            entity_id,
            mention_name,
        ) in mentions:
            connection.execute(
                "INSERT INTO entity_mentions("
                "source, chunk_id, chunk_position, heading_path, entity_id, "
                "mention_name"
                ") "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source, chunk_id, position, heading_path, entity_id, mention_name),
            )

    def _write_relations(
        self,
        connection: sqlite3.Connection,
        relations: dict[str, tuple[RelationType, str, str]],
        evidence: list[_EvidenceRow],
    ) -> None:
        for relation_id, (relation_type, source_id, target_id) in relations.items():
            connection.execute(
                "INSERT INTO relations("
                "relation_id, relation_type, source_entity_id, target_entity_id"
                ") "
                "VALUES (?, ?, ?, ?) ON CONFLICT(relation_id) DO NOTHING",
                (relation_id, relation_type.value, source_id, target_id),
            )
        for relation_id, source, chunk_id, position, heading_path, quote in evidence:
            connection.execute(
                "INSERT INTO relation_evidence("
                "relation_id, source, chunk_id, chunk_position, heading_path, "
                "evidence_quote"
                ") "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (relation_id, source, chunk_id, position, heading_path, quote),
            )

    def _remove_orphans(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "DELETE FROM relations WHERE NOT EXISTS ("
            "SELECT 1 FROM relation_evidence "
            "WHERE relation_evidence.relation_id = relations.relation_id"
            ")"
        )
        connection.execute(
            "DELETE FROM entities WHERE NOT EXISTS ("
            "SELECT 1 FROM entity_mentions "
            "WHERE entity_mentions.entity_id = entities.entity_id"
            ") AND NOT EXISTS ("
            "SELECT 1 FROM relations "
            "WHERE relations.source_entity_id = entities.entity_id "
            "OR relations.target_entity_id = entities.entity_id"
            ")"
        )

    def _recompute_display_names(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE entities SET display_name = ("
            "SELECT MIN(mention_name) FROM entity_mentions "
            "WHERE entity_mentions.entity_id = entities.entity_id)"
        )

    def _rebuild_postings(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM entity_postings")
        rows = connection.execute(
            "SELECT entity_id, display_name, normalized_name FROM entities"
        ).fetchall()
        for row in rows:
            entity_id = self._require_text(row["entity_id"], "entity_id")
            display_name = self._require_text(row["display_name"], "display_name")
            normalized_name = self._require_text(
                row["normalized_name"], "normalized_name"
            )
            frequencies = Counter(
                self._tokenizer.tokenize(f"{display_name} {normalized_name}")
            )
            token_count = sum(frequencies.values())
            if token_count <= 0:
                raise GraphStorageError("Graph entity contains no searchable terms")
            connection.execute(
                "UPDATE entities SET token_count = ? WHERE entity_id = ?",
                (token_count, entity_id),
            )
            connection.executemany(
                "INSERT INTO entity_postings(token, entity_id, frequency) "
                "VALUES (?, ?, ?)",
                [
                    (token, entity_id, frequency)
                    for token, frequency in frequencies.items()
                ],
            )

    def _bm25_scores(
        self,
        connection: sqlite3.Connection,
        tokens: tuple[str, ...],
        entity_count: int,
        average_length: float,
    ) -> dict[str, float]:
        scores: defaultdict[str, float] = defaultdict(float)
        for token in tokens:
            row = connection.execute(
                "SELECT COUNT(*) FROM entity_postings WHERE token = ?", (token,)
            ).fetchone()
            if row is None:
                raise GraphStorageError("Graph posting statistics are invalid")
            document_frequency = self._require_int(row[0], "document frequency")
            if document_frequency == 0:
                continue
            inverse_frequency = log(
                1
                + (entity_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            rows = connection.execute(
                "SELECT posting.entity_id, posting.frequency, entity.token_count "
                "FROM entity_postings AS posting JOIN entities AS entity "
                "ON entity.entity_id = posting.entity_id WHERE posting.token = ?",
                (token,),
            ).fetchall()
            for item in rows:
                entity_id = self._require_text(item["entity_id"], "entity_id")
                frequency = self._require_int(item["frequency"], "posting frequency")
                length = self._require_int(item["token_count"], "token_count")
                denominator = frequency + _BM25_K1 * (
                    1 - _BM25_B + _BM25_B * length / average_length
                )
                scores[entity_id] += (
                    inverse_frequency * frequency * (_BM25_K1 + 1) / denominator
                )
        return dict(scores)

    def _neighbors(
        self,
        connection: sqlite3.Connection,
        entity_id: str,
        *,
        outgoing: bool,
    ) -> tuple[GraphNeighbor, ...]:
        endpoint = "source_entity_id" if outgoing else "target_entity_id"
        neighbor = "target_entity_id" if outgoing else "source_entity_id"
        rows = connection.execute(
            "SELECT relation.relation_id, relation.relation_type, "
            "relation.source_entity_id, relation.target_entity_id, "
            "neighbor.entity_id, neighbor.entity_type, neighbor.display_name, "
            "evidence.source, "
            "evidence.chunk_id, "
            "evidence.chunk_position, evidence.heading_path, evidence.evidence_quote "
            "FROM relations AS relation "
            "JOIN entities AS neighbor ON neighbor.entity_id = relation."
            + neighbor
            + " JOIN relation_evidence AS evidence "
            "ON evidence.relation_id = relation.relation_id "
            "WHERE relation."
            + endpoint
            + " = ? ORDER BY relation.relation_type, neighbor.display_name, "
            "neighbor.entity_id, "
            "evidence.source, evidence.chunk_position, evidence.evidence_quote",
            (entity_id,),
        ).fetchall()
        grouped: dict[tuple[str, str], list[GraphProvenance]] = {}
        entities: dict[tuple[str, str], tuple[EntityType, str, RelationType]] = {}
        for row in rows:
            relation_id = self._require_text(row["relation_id"], "relation_id")
            neighbor_id = self._require_text(row["entity_id"], "entity_id")
            relation = GraphRelation(
                relation_id,
                RelationType(self._require_text(row["relation_type"], "relation_type")),
                self._require_text(row["source_entity_id"], "source_entity_id"),
                self._require_text(row["target_entity_id"], "target_entity_id"),
            )
            neighbor_entity = GraphEntity(
                neighbor_id,
                EntityType(self._require_text(row["entity_type"], "entity_type")),
                self._require_text(row["display_name"], "display_name"),
            )
            key = (relation_id, neighbor_id)
            heading_path = self._decode_heading(row["heading_path"])
            grouped.setdefault(key, []).append(
                GraphProvenance(
                    self._require_text(row["source"], "source"),
                    self._require_text(row["chunk_id"], "chunk_id"),
                    self._require_int(row["chunk_position"], "chunk_position"),
                    heading_path,
                    self._require_text(row["evidence_quote"], "evidence_quote"),
                )
            )
            entities[key] = (
                neighbor_entity.entity_type,
                neighbor_entity.display_name,
                relation.relation_type,
            )
        return tuple(
            GraphNeighbor(
                neighbor_id,
                entities[(relation_id, neighbor_id)][1],
                entities[(relation_id, neighbor_id)][0],
                entities[(relation_id, neighbor_id)][2],
                tuple(grouped[(relation_id, neighbor_id)]),
            )
            for relation_id, neighbor_id in grouped
        )

    def _entity_from_row(self, row: sqlite3.Row) -> GraphEntity:
        entity_id = self._require_text(row["entity_id"], "entity_id")
        entity_type = EntityType(self._require_text(row["entity_type"], "entity_type"))
        display_name = self._require_text(row["display_name"], "display_name")
        if entity_id != make_entity_id(entity_type, display_name):
            raise ValueError("stored entity identity is invalid")
        return GraphEntity(entity_id, entity_type, display_name)

    def _search_result_from_row(
        self,
        row: sqlite3.Row,
        scores: dict[str, float],
    ) -> EntitySearchResult:
        entity = self._entity_from_row(row)
        return EntitySearchResult(
            entity.entity_id,
            entity.entity_type,
            entity.display_name,
            scores[entity.entity_id],
        )

    @staticmethod
    def _decode_heading(value: object) -> tuple[str, ...]:
        if not isinstance(value, str):
            raise ValueError("heading_path is invalid")
        decoded = json.loads(value)
        if not isinstance(decoded, list) or not all(
            isinstance(item, str) for item in decoded
        ):
            raise ValueError("heading_path is invalid")
        return tuple(decoded)

    @staticmethod
    def _require_text(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"stored {field_name} is invalid")
        return value

    @staticmethod
    def _require_int(value: object, field_name: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"stored {field_name} is invalid")
        return value


class _PreparedReplacement:
    def __init__(
        self,
        entities: dict[str, GraphEntity],
        mentions: list[_MentionRow],
        relations: dict[str, tuple[RelationType, str, str]],
        evidence: list[_EvidenceRow],
        document_entity_ids: set[str],
        document_relation_ids: set[str],
    ) -> None:
        self.entities = entities
        self.mentions = mentions
        self.relations = relations
        self.evidence = evidence
        self.document_entity_ids = document_entity_ids
        self.document_relation_ids = document_relation_ids

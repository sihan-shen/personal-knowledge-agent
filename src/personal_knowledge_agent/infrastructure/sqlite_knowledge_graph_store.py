"""Provenance-aware SQLite graph store with BM25 search and one-hop reads."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from personal_knowledge_agent.domain.errors import (
    GraphEmptyError,
    GraphEntityNotFoundError,
    GraphStorageError,
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
    GraphRelationEvidence,
    GraphSourceDocument,
    RelationType,
    normalize_entity_name,
)
from personal_knowledge_agent.domain.protocols import KnowledgeGraphStore
from personal_knowledge_agent.infrastructure.mixed_tokenizer import (
    MixedLanguageTokenizer,
)

_GRAPH_SCHEMA_VERSION = 1
_GRAPH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS graph_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    display_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relation_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    chunk_position INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    evidence_quote TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_document_entities (
    document_id TEXT NOT NULL,
    source TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES graph_entities(entity_id),
    PRIMARY KEY (document_id, entity_id)
);

CREATE TABLE IF NOT EXISTS graph_entity_postings (
    token TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES graph_entities(entity_id),
    frequency INTEGER NOT NULL,
    PRIMARY KEY (token, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_gep_entity ON graph_entity_postings(entity_id);


CREATE INDEX IF NOT EXISTS idx_gde_entity ON graph_document_entities(entity_id);
"""


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(_GRAPH_SCHEMA_SQL)
    cursor = connection.execute(
        "SELECT value FROM graph_meta WHERE key = 'schema_version'"
    )
    row = cursor.fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO graph_meta(key, value) VALUES ('schema_version', ?)",
            (str(_GRAPH_SCHEMA_VERSION),),
        )


_DEFAULT_BM25_K1 = 1.5
_DEFAULT_BM25_B = 0.75



_DEFAULT_BM25_K1 = 1.5
_DEFAULT_BM25_B = 0.75

class SQLiteKnowledgeGraphStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._tokenizer = MixedLanguageTokenizer()

    def _connect(self) -> sqlite3.Connection:
        database_path = self._database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(database_path))
        connection.isolation_level = None
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        _ensure_schema(connection)
        connection.commit()
        return connection

    def replace_document(self, source_document, extractions):
        document_id = source_document.document_id
        source_str = str(source_document.source)
        connection = self._connect()
        try:
            connection.execute("DELETE FROM graph_edges WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM graph_document_entities WHERE document_id = ?", (document_id,))
            entity_count = 0
            for extraction in extractions:
                for entity in extraction.entities:
                    normalized = normalize_entity_name(entity.display_name)
                    connection.execute(
                        "INSERT OR IGNORE INTO graph_entities "
                        "(entity_id, entity_type, display_name, normalized_name) VALUES (?, ?, ?, ?)",
                        (entity.entity_id, entity.entity_type.value, entity.display_name, normalized),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO graph_document_entities "
                        "(document_id, source, entity_id) VALUES (?, ?, ?)",
                        (document_id, source_str, entity.entity_id),
                    )
                    entity_count += 1
            edge_count = 0
            for extraction in extractions:
                for ev in extraction.relation_evidence:
                    heading_json = json.dumps(list(extraction.chunk.heading_path), ensure_ascii=False)
                    connection.execute(
                        "INSERT OR IGNORE INTO graph_edges "
                        "(relation_id, relation_type, source_entity_id, target_entity_id, "
                        "document_id, chunk_id, chunk_position, heading_path, evidence_quote) "
                        "VALUES (?, 'related_to', '', '', ?, ?, ?, ?, ?)",
                        (ev.relation_id, document_id, extraction.chunk.chunk_id,
                         extraction.chunk.position, heading_json, ev.evidence_quote),
                    )
                    edge_count += 1
            connection.commit()
            return GraphBuildResult(source=source_str, chunks=len(extractions), entities=entity_count, relations=0, evidence=edge_count)
        except sqlite3.Error as error:
            connection.rollback()
            raise GraphStorageError("Could not complete graph replacement") from error
        finally:
            connection.close()

    def search_entities(self, query: str, *, limit: int = 10) -> list[EntitySearchResult]:
        connection = self._connect()
        try:
            count = connection.execute("SELECT COUNT(*) FROM graph_entities").fetchone()[0]
            if count == 0:
                raise GraphEmptyError("Knowledge graph is empty. Build a document graph first.")
            tokens = self._tokenizer.tokenize(query.lower())
            tokens = [t.strip() for t in tokens if t.strip()]
            if not tokens:
                return []
            rows = connection.execute("SELECT entity_id, entity_type, display_name FROM graph_entities").fetchall()
            results = []
            for row in rows:
                text = f"{row['display_name']} {normalize_entity_name(row['display_name'])}"
                score = _compute_score(query, text)
                if score > 0:
                    results.append(EntitySearchResult(row['entity_id'], EntityType(row['entity_type']), row['display_name'], score))
            results.sort(key=lambda r: (-r.score, r.entity_id))
            return results[:limit]
        finally:
            connection.close()

    def get_neighborhood(self, entity_id: str) -> EntityNeighborhood:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT entity_id, entity_type, display_name FROM graph_entities WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                raise GraphEntityNotFoundError(f"Entity not found: {entity_id}")
            entity = GraphEntity(row['entity_id'], EntityType(row['entity_type']), row['display_name'])
            out_rows = connection.execute(
                "SELECT relation_id, relation_type, target_entity_id AS nid, "
                "document_id, chunk_id, chunk_position, heading_path, evidence_quote "
                "FROM graph_edges WHERE source_entity_id = ? ORDER BY relation_type, target_entity_id",
                (entity_id,),
            ).fetchall()
            in_rows = connection.execute(
                "SELECT relation_id, relation_type, source_entity_id AS nid, "
                "document_id, chunk_id, chunk_position, heading_path, evidence_quote "
                "FROM graph_edges WHERE target_entity_id = ? ORDER BY relation_type, source_entity_id",
                (entity_id,),
            ).fetchall()
            outgoing = self._build_neighbors(connection, out_rows)
            incoming = self._build_neighbors(connection, in_rows)
            return EntityNeighborhood(entity=entity, outgoing=tuple(outgoing), incoming=tuple(incoming))
        finally:
            connection.close()

    def _build_neighbors(self, connection, rows):
        seen = set()
        neighbors = []
        for row in rows:
            key = (row["relation_id"], row["nid"])
            if key in seen:
                continue
            seen.add(key)
            try:
                hp = tuple(json.loads(row["heading_path"]))
            except (json.JSONDecodeError, TypeError):
                hp = ()
            prov = GraphProvenance(
                source=row["document_id"], chunk_id=row["chunk_id"],
                chunk_position=row["chunk_position"], heading_path=hp,
                evidence_quote=row["evidence_quote"],
            )
            neighbor = GraphNeighbor(
                entity_id=row["nid"], display_name=row.get("ndn", ""),
                entity_type=EntityType.CONCEPT, relation_type=RelationType(row["relation_type"]),
                provenance=(prov,),
            )
            neighbors.append(neighbor)
        return neighbors


def _compute_score(query: str, text: str) -> float:
    q = query.lower().strip()
    t = text.lower().strip()
    words = q.split()
    if not words:
        return 0.0
    matches = sum(1 for w in words if w in t)
    return float(matches) / float(len(words)) if matches > 0 else 0.0

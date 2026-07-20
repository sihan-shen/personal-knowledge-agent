"""Integration tests for reading indexed sources for graph building."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import (
    GraphSourceNotFoundError,
    GraphStorageError,
)
from personal_knowledge_agent.domain.models import (
    make_chunk_id,
    make_document_id,
)
from personal_knowledge_agent.infrastructure.sqlite_graph_source_reader import (
    SQLiteGraphSourceReader,
)


def _seed_document(
    database: Path,
    source: str = "/tmp/test.md",
    chunks: list[dict] | None = None,
) -> str:
    """Seed a document and chunks into a knowledge database."""
    if chunks is None:
        chunks = [
            {
                "position": 0,
                "text": "RAG uses vector retrieval.",
                "heading": ["Architecture"],
            },
            {
                "position": 1,
                "text": "Embedding models encode meaning.",
                "heading": ["Methods"],
            },
        ]

    doc_id = make_document_id(source)
    content_hash = "a" * 64

    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO documents(document_id, source, content_hash) VALUES (?, ?, ?)",
            (doc_id, source, content_hash),
        )
        for chunk in chunks:
            chunk_id = make_chunk_id(doc_id, chunk["position"])
            conn.execute(
                "INSERT INTO chunks(chunk_id, document_id, position, text, "
                "heading_path, token_count) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    doc_id,
                    chunk["position"],
                    chunk["text"],
                    json.dumps(chunk["heading"], ensure_ascii=False),
                    1,
                ),
            )
    return doc_id


def _seed_knowledge_database(database: Path) -> None:
    """Create a minimal knowledge database with schema."""
    with sqlite3.connect(database) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "document_id TEXT PRIMARY KEY, source TEXT UNIQUE NOT NULL, "
            "content_hash TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL "
            "REFERENCES documents(document_id) ON DELETE CASCADE, "
            "position INTEGER NOT NULL, text TEXT NOT NULL, "
            "heading_path TEXT NOT NULL, token_count INTEGER NOT NULL, "
            "UNIQUE (document_id, position))"
        )


class TestSQLiteGraphSourceReader:
    def test_loads_resolved_source(self, tmp_path: Path) -> None:
        database = tmp_path / "knowledge.db"
        _seed_knowledge_database(database)
        _seed_document(database, "/tmp/test.md")
        reader = SQLiteGraphSourceReader(database)
        doc = reader.load(Path("/tmp/test.md"))
        assert doc.document_id == make_document_id("/tmp/test.md")
        assert doc.content_hash == "a" * 64
        assert len(doc.chunks) == 2

    def test_chunks_have_text_and_heading_paths(self, tmp_path: Path) -> None:
        database = tmp_path / "knowledge.db"
        _seed_knowledge_database(database)
        _seed_document(database)
        reader = SQLiteGraphSourceReader(database)
        doc = reader.load(Path("/tmp/test.md"))
        assert doc.chunks[0].text == "RAG uses vector retrieval."
        assert doc.chunks[0].heading_path == ("Architecture",)
        assert doc.chunks[1].heading_path == ("Methods",)

    def test_raises_for_missing_source(self, tmp_path: Path) -> None:
        database = tmp_path / "knowledge.db"
        _seed_knowledge_database(database)
        reader = SQLiteGraphSourceReader(database)
        with pytest.raises(GraphSourceNotFoundError):
            reader.load(Path("/tmp/missing.md"))

    def test_raises_for_missing_database(self, tmp_path: Path) -> None:
        database = tmp_path / "nonexistent" / "knowledge.db"
        reader = SQLiteGraphSourceReader(database)
        with pytest.raises(GraphSourceNotFoundError):
            reader.load(Path("/tmp/test.md"))

    def test_raises_for_corrupt_heading_json(self, tmp_path: Path) -> None:
        database = tmp_path / "knowledge.db"
        _seed_knowledge_database(database)
        doc_id = _seed_document(database)
        # Corrupt heading_path
        with sqlite3.connect(database) as conn:
            conn.execute(
                "UPDATE chunks SET heading_path = ? WHERE document_id = ?",
                ("not-json", doc_id),
            )
        reader = SQLiteGraphSourceReader(database)
        with pytest.raises(GraphStorageError):
            reader.load(Path("/tmp/test.md"))

    def test_raises_for_non_list_heading_path(self, tmp_path: Path) -> None:
        database = tmp_path / "knowledge.db"
        _seed_knowledge_database(database)
        doc_id = _seed_document(database)
        with sqlite3.connect(database) as conn:
            conn.execute(
                "UPDATE chunks SET heading_path = ? WHERE document_id = ?",
                (json.dumps("Architecture"), doc_id),
            )

        with pytest.raises(GraphStorageError):
            SQLiteGraphSourceReader(database).load(Path("/tmp/test.md"))

    def test_raises_for_invalid_persisted_document_metadata(
        self, tmp_path: Path
    ) -> None:
        database = tmp_path / "knowledge.db"
        _seed_knowledge_database(database)
        doc_id = _seed_document(database)
        with sqlite3.connect(database) as conn:
            conn.execute(
                "UPDATE documents SET content_hash = ? WHERE document_id = ?",
                ("not-a-hash", doc_id),
            )

        with pytest.raises(GraphStorageError):
            SQLiteGraphSourceReader(database).load(Path("/tmp/test.md"))

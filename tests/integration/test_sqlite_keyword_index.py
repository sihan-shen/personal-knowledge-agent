"""Integration tests for persistent SQLite keyword indexing."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import IndexStorageError
from personal_knowledge_agent.domain.models import (
    Chunk,
    Document,
    make_chunk_id,
    make_document_id,
)
from personal_knowledge_agent.infrastructure.sqlite_keyword_index import (
    SQLiteKeywordIndex,
)


def make_document(source: str = "/notes/rag.md", content: str = "RAG") -> Document:
    return Document(make_document_id(source), source, content)


def make_chunks(document: Document, *texts: str) -> list[Chunk]:
    return [
        Chunk(
            make_chunk_id(document.document_id, position),
            document.document_id,
            position,
            text,
            ("RAG",),
        )
        for position, text in enumerate(texts)
    ]


def read_rows(database: Path, query: str) -> list[tuple[object, ...]]:
    with sqlite3.connect(database) as connection:
        return connection.execute(query).fetchall()


def test_initializes_schema_version_one(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "knowledge.db"

    SQLiteKeywordIndex(database)

    assert read_rows(database, "PRAGMA user_version") == [(1,)]
    tables = {row[0] for row in read_rows(database, "SELECT name FROM sqlite_master")}
    assert {"documents", "chunks", "postings"} <= tables


def test_persists_document_across_index_instances(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    chunks = make_chunks(document, "retrieval evidence")

    assert SQLiteKeywordIndex(database).replace_document(document, chunks) is False
    assert SQLiteKeywordIndex(database).replace_document(document, chunks) is True


def test_replaces_old_chunks_without_duplicates(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    index = SQLiteKeywordIndex(database)
    document = make_document()

    index.replace_document(document, make_chunks(document, "old text", "old second"))
    replaced = index.replace_document(document, make_chunks(document, "new text"))

    assert replaced is True
    assert read_rows(database, "SELECT text FROM chunks") == [("new text",)]
    assert read_rows(database, "SELECT COUNT(*) FROM documents") == [(1,)]


def test_indexes_heading_and_body_term_frequencies(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    chunk = Chunk(
        make_chunk_id(document.document_id, 0),
        document.document_id,
        0,
        "RAG rag retrieval",
        ("Retrieval",),
    )

    SQLiteKeywordIndex(database).replace_document(document, [chunk])

    postings = read_rows(
        database,
        "SELECT token, frequency FROM postings ORDER BY token",
    )
    assert postings == [("rag", 2), ("retrieval", 2)]


def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 2")

    with pytest.raises(IndexStorageError, match="version 2"):
        SQLiteKeywordIndex(database)


def test_failed_replacement_rolls_back_previous_document(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    index = SQLiteKeywordIndex(database)
    document = make_document()
    original = make_chunks(document, "original text")
    index.replace_document(document, original)
    duplicate = Chunk(
        original[0].chunk_id,
        document.document_id,
        1,
        "duplicate id",
    )

    with pytest.raises(IndexStorageError):
        index.replace_document(document, [original[0], duplicate])

    assert read_rows(database, "SELECT text FROM chunks") == [("original text",)]

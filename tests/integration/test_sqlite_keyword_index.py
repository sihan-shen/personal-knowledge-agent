"""Integration tests for persistent SQLite keyword indexing."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import (
    IndexStorageError,
    InvalidQueryError,
    KnowledgeBaseEmptyError,
)
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


def test_search_ranks_higher_term_frequency_first(tmp_path: Path) -> None:
    index = SQLiteKeywordIndex(tmp_path / "knowledge.db")
    document = make_document()
    index.replace_document(
        document,
        make_chunks(
            document,
            "retrieval retrieval evidence",
            "retrieval overview with extra words",
        ),
    )

    results = index.search("retrieval")

    assert [result.chunk.position for result in results] == [0, 1]
    assert results[0].score > results[1].score > 0


def test_search_supports_chinese_and_mixed_queries(tmp_path: Path) -> None:
    index = SQLiteKeywordIndex(tmp_path / "knowledge.db")
    relevant = make_document("/notes/rag.md", "RAG")
    distractor = make_document("/notes/coffee.md", "coffee")
    index.replace_document(relevant, make_chunks(relevant, "RAG 使用检索增强生成"))
    index.replace_document(distractor, make_chunks(distractor, "咖啡冲煮需要控制水温"))

    chinese = index.search("检索增强")
    mixed = index.search("RAG 检索")

    assert chinese[0].source == relevant.source
    assert mixed[0].source == relevant.source


def test_search_indexes_heading_terms(tmp_path: Path) -> None:
    index = SQLiteKeywordIndex(tmp_path / "knowledge.db")
    document = make_document()
    chunk = Chunk(
        make_chunk_id(document.document_id, 0),
        document.document_id,
        0,
        "Tools have explicit contracts.",
        ("Agent", "Tools"),
    )
    index.replace_document(document, [chunk])

    assert index.search("agent")[0].chunk.heading_path == ("Agent", "Tools")


def test_search_applies_limit_and_stable_tie_break(tmp_path: Path) -> None:
    index = SQLiteKeywordIndex(tmp_path / "knowledge.db")
    document = make_document()
    chunks = make_chunks(document, "same", "same", "same")
    index.replace_document(document, chunks)

    results = index.search("same", limit=2)

    expected = sorted(chunk.chunk_id for chunk in chunks)[:2]
    assert [result.chunk.chunk_id for result in results] == expected


def test_search_persists_after_reopening(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    SQLiteKeywordIndex(database).replace_document(
        document,
        make_chunks(document, "persistent retrieval"),
    )

    results = SQLiteKeywordIndex(database).search("persistent")

    assert results[0].source == document.source


def test_search_rejects_query_without_tokens(tmp_path: Path) -> None:
    index = SQLiteKeywordIndex(tmp_path / "knowledge.db")

    with pytest.raises(InvalidQueryError, match="searchable terms"):
        index.search("，。！？")


def test_search_rejects_empty_knowledge_base(tmp_path: Path) -> None:
    index = SQLiteKeywordIndex(tmp_path / "knowledge.db")

    with pytest.raises(KnowledgeBaseEmptyError, match="empty"):
        index.search("rag")


def test_search_returns_empty_list_when_no_terms_match(tmp_path: Path) -> None:
    index = SQLiteKeywordIndex(tmp_path / "knowledge.db")
    document = make_document()
    index.replace_document(document, make_chunks(document, "retrieval evidence"))

    assert index.search("gardening") == []

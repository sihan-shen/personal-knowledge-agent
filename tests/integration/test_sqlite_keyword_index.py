"""Integration tests for persistent SQLite keyword indexing."""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Sequence
from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import (
    EmbeddingError,
    IndexStorageError,
    InvalidQueryError,
    KnowledgeBaseEmptyError,
)
from personal_knowledge_agent.domain.models import (
    Chunk,
    Document,
    Embedding,
    SearchMode,
    make_chunk_id,
    make_document_id,
)
from personal_knowledge_agent.domain.protocols import EmbeddingModel
from personal_knowledge_agent.infrastructure.sqlite_keyword_index import (
    SQLiteKeywordIndex,
    _unpack_vector,
)
from tests.helpers import FakeEmbedder


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


def make_index(
    database: Path,
    embedder: FakeEmbedder | None = None,
) -> SQLiteKeywordIndex:
    return SQLiteKeywordIndex(
        database,
        embedder or FakeEmbedder({}, default_vector=(1.0, 0.0)),
    )


def test_initializes_schema_version_two(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "knowledge.db"

    make_index(database)

    assert read_rows(database, "PRAGMA user_version") == [(2,)]
    tables = {row[0] for row in read_rows(database, "SELECT name FROM sqlite_master")}
    assert {"documents", "chunks", "postings", "embeddings"} <= tables


def test_migrates_v1_database_without_losing_lexical_rows(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                source TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(document_id)
                    ON DELETE CASCADE,
                position INTEGER NOT NULL,
                text TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                UNIQUE (document_id, position)
            );
            CREATE TABLE postings (
                token TEXT NOT NULL,
                chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id)
                    ON DELETE CASCADE,
                frequency INTEGER NOT NULL,
                PRIMARY KEY (token, chunk_id)
            );
            INSERT INTO documents VALUES ('doc', '/old.md', 'hash');
            INSERT INTO chunks VALUES ('chunk', 'doc', 0, 'retrieval', '[]', 1);
            INSERT INTO postings VALUES ('retrieval', 'chunk', 1);
            PRAGMA user_version = 1;
            """
        )

    index = make_index(database)

    assert read_rows(database, "PRAGMA user_version") == [(2,)]
    assert read_rows(database, "SELECT text FROM chunks") == [("retrieval",)]
    assert index.search("retrieval")[0].source == "/old.md"


def test_persists_document_across_index_instances(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    chunks = make_chunks(document, "retrieval evidence")

    assert make_index(database).replace_document(document, chunks) is False
    assert make_index(database).replace_document(document, chunks) is True


def test_replaces_old_chunks_without_duplicates(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    index = make_index(database)
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

    make_index(database).replace_document(document, [chunk])

    postings = read_rows(
        database,
        "SELECT token, frequency FROM postings ORDER BY token",
    )
    assert postings == [("rag", 2), ("retrieval", 2)]


def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 3")

    with pytest.raises(IndexStorageError, match="version 3"):
        make_index(database)


def test_failed_replacement_rolls_back_previous_document(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    index = make_index(database)
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
    assert read_rows(database, "SELECT COUNT(*) FROM embeddings") == [(1,)]


def test_persists_little_endian_float32_embedding_per_chunk(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    chunk = make_chunks(document, "retrieval evidence")[0]
    passage = "RAG\nretrieval evidence"
    embedder = FakeEmbedder({passage: (3.0, 4.0)}, model_id="fake-e5")

    make_index(database, embedder).replace_document(document, [chunk])

    rows = read_rows(
        database,
        "SELECT chunk_id, model_id, dimensions, vector FROM embeddings",
    )
    assert rows[0][:3] == (chunk.chunk_id, "fake-e5", 2)
    vector = rows[0][3]
    assert isinstance(vector, bytes)
    assert struct.unpack("<2f", vector) == pytest.approx((0.6, 0.8))
    assert embedder.document_calls == [[passage]]


def test_vector_unpack_rejects_invalid_blob_length() -> None:
    with pytest.raises(IndexStorageError, match="byte length"):
        _unpack_vector(b"\x00\x00", dimensions=2)


def test_replacement_removes_old_embedding_rows(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    index = make_index(database)
    old_chunks = make_chunks(document, "old first", "old second")
    new_chunk = make_chunks(document, "new only")[0]

    index.replace_document(document, old_chunks)
    index.replace_document(document, [new_chunk])

    assert read_rows(database, "SELECT chunk_id FROM embeddings") == [
        (new_chunk.chunk_id,)
    ]


def test_embedding_failure_preserves_previously_stored_document(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    original = make_chunks(document, "original text")
    make_index(database).replace_document(document, original)
    failing = make_index(database, FakeEmbedder({}))

    with pytest.raises(EmbeddingError, match="No fake embedding"):
        failing.replace_document(document, make_chunks(document, "replacement"))

    assert read_rows(database, "SELECT text FROM chunks") == [("original text",)]
    assert read_rows(database, "SELECT chunk_id FROM embeddings") == [
        (original[0].chunk_id,)
    ]


@pytest.mark.parametrize(
    ("embedder", "message"),
    [
        (WrongCountEmbedder(), "0 vectors for 1 chunks"),
        (WrongModelIdEmbedder(), "identifier"),
    ],
)
def test_rejects_incompatible_embedding_batches_before_writing(
    tmp_path: Path,
    embedder: EmbeddingModel,
    message: str,
) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()

    with pytest.raises(EmbeddingError, match=message):
        SQLiteKeywordIndex(database, embedder).replace_document(
            document,
            make_chunks(document, "text"),
        )

    assert read_rows(database, "SELECT COUNT(*) FROM documents") == [(0,)]


def test_rejects_mixed_embedding_dimensions_before_writing(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    chunks = make_chunks(document, "first", "second")
    embedder = FakeEmbedder(
        {
            "RAG\nfirst": (1.0,),
            "RAG\nsecond": (1.0, 0.0),
        }
    )

    with pytest.raises(EmbeddingError, match="common dimension"):
        make_index(database, embedder).replace_document(document, chunks)

    assert read_rows(database, "SELECT COUNT(*) FROM documents") == [(0,)]


def test_search_ranks_higher_term_frequency_first(tmp_path: Path) -> None:
    index = make_index(tmp_path / "knowledge.db")
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
    index = make_index(tmp_path / "knowledge.db")
    relevant = make_document("/notes/rag.md", "RAG")
    distractor = make_document("/notes/coffee.md", "coffee")
    index.replace_document(relevant, make_chunks(relevant, "RAG 使用检索增强生成"))
    index.replace_document(distractor, make_chunks(distractor, "咖啡冲煮需要控制水温"))

    chinese = index.search("检索增强")
    mixed = index.search("RAG 检索")

    assert chinese[0].source == relevant.source
    assert mixed[0].source == relevant.source


def test_search_indexes_heading_terms(tmp_path: Path) -> None:
    index = make_index(tmp_path / "knowledge.db")
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
    index = make_index(tmp_path / "knowledge.db")
    document = make_document()
    chunks = make_chunks(document, "same", "same", "same")
    index.replace_document(document, chunks)

    results = index.search("same", limit=2)

    expected = sorted(chunk.chunk_id for chunk in chunks)[:2]
    assert [result.chunk.chunk_id for result in results] == expected


def test_search_persists_after_reopening(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    make_index(database).replace_document(
        document,
        make_chunks(document, "persistent retrieval"),
    )

    results = make_index(database).search("persistent")

    assert results[0].source == document.source


def test_search_rejects_query_without_tokens(tmp_path: Path) -> None:
    index = make_index(tmp_path / "knowledge.db")

    with pytest.raises(InvalidQueryError, match="searchable terms"):
        index.search("，。！？")


def test_search_rejects_empty_knowledge_base(tmp_path: Path) -> None:
    index = make_index(tmp_path / "knowledge.db")

    with pytest.raises(KnowledgeBaseEmptyError, match="empty"):
        index.search("rag")


def test_search_returns_empty_list_when_no_terms_match(tmp_path: Path) -> None:
    index = make_index(tmp_path / "knowledge.db")
    document = make_document()
    index.replace_document(document, make_chunks(document, "retrieval evidence"))

    assert index.search("gardening") == []


def test_semantic_search_orders_positive_cosine_scores(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    chunks = make_chunks(document, "closest", "partial", "opposite")
    embedder = FakeEmbedder(
        {
            "RAG\nclosest": (1.0, 0.0),
            "RAG\npartial": (1.0, 1.0),
            "RAG\nopposite": (-1.0, 0.0),
            "meaning": (1.0, 0.0),
        }
    )
    index = make_index(database, embedder)
    index.replace_document(document, chunks)

    results = index.search("meaning", mode=SearchMode.SEMANTIC)

    assert [result.chunk.text for result in results] == ["closest", "partial"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(2**-0.5)
    assert embedder.query_calls == ["meaning"]


def test_semantic_search_returns_empty_when_cosine_is_not_positive(
    tmp_path: Path,
) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    embedder = FakeEmbedder(
        {
            "RAG\nunrelated": (0.0, 1.0),
            "meaning": (1.0, 0.0),
        }
    )
    index = make_index(database, embedder)
    index.replace_document(document, make_chunks(document, "unrelated"))

    assert index.search("meaning", mode=SearchMode.SEMANTIC) == []


def test_semantic_search_rejects_stored_model_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    embedder = FakeEmbedder({"RAG\ntext": (1.0, 0.0), "meaning": (1.0, 0.0)})
    index = make_index(database, embedder)
    index.replace_document(document, make_chunks(document, "text"))
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE embeddings SET model_id = 'old-model'")

    with pytest.raises(IndexStorageError, match="Re-ingest.*old-model"):
        index.search("meaning", mode=SearchMode.SEMANTIC)


def test_semantic_search_breaks_equal_scores_by_chunk_id(tmp_path: Path) -> None:
    database = tmp_path / "knowledge.db"
    document = make_document()
    chunks = make_chunks(document, "first", "second")
    embedder = FakeEmbedder(
        {
            "RAG\nfirst": (1.0, 0.0),
            "RAG\nsecond": (1.0, 0.0),
            "meaning": (1.0, 0.0),
        }
    )
    index = make_index(database, embedder)
    index.replace_document(document, chunks)

    results = index.search("meaning", mode=SearchMode.SEMANTIC)

    assert [result.chunk.chunk_id for result in results] == sorted(
        chunk.chunk_id for chunk in chunks
    )


def test_empty_semantic_search_does_not_call_embedding_model(tmp_path: Path) -> None:
    embedder = FakeEmbedder({"meaning": (1.0, 0.0)})
    index = make_index(tmp_path / "knowledge.db", embedder)

    with pytest.raises(KnowledgeBaseEmptyError):
        index.search("meaning", mode=SearchMode.SEMANTIC)

    assert embedder.query_calls == []

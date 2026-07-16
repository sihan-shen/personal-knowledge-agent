"""Persistent SQLite inverted index for lexical retrieval."""

from __future__ import annotations

import json
import sqlite3
import struct
from collections import Counter
from collections.abc import Sequence
from contextlib import closing
from hashlib import sha256
from math import log
from pathlib import Path

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
    SearchResult,
)
from personal_knowledge_agent.domain.protocols import EmbeddingModel
from personal_knowledge_agent.infrastructure.mixed_tokenizer import (
    MixedLanguageTokenizer,
)

_SCHEMA_VERSION = 2
_BM25_K1 = 1.5
_BM25_B = 0.75
_SCHEMA = """
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    source TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL
);

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL
        REFERENCES documents(document_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    UNIQUE (document_id, position)
);

CREATE TABLE postings (
    token TEXT NOT NULL,
    chunk_id TEXT NOT NULL
        REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    frequency INTEGER NOT NULL,
    PRIMARY KEY (token, chunk_id)
);

CREATE INDEX idx_postings_token ON postings(token);

CREATE TABLE embeddings (
    chunk_id TEXT PRIMARY KEY
        REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE INDEX idx_embeddings_model_id ON embeddings(model_id);
"""

_EMBEDDINGS_SCHEMA = """
CREATE TABLE embeddings (
    chunk_id TEXT PRIMARY KEY
        REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE INDEX idx_embeddings_model_id ON embeddings(model_id);
"""


def _pack_vector(values: Sequence[float]) -> bytes:
    try:
        return struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error) as error:
        raise EmbeddingError("Embedding values cannot be stored as float32") from error


def _unpack_vector(value: bytes, *, dimensions: int) -> tuple[float, ...]:
    if dimensions <= 0:
        raise IndexStorageError("Stored embedding dimension must be positive")
    expected_length = dimensions * 4
    if len(value) != expected_length:
        raise IndexStorageError(
            "Stored embedding byte length does not match its dimension"
        )
    try:
        return tuple(struct.unpack(f"<{dimensions}f", value))
    except struct.error as error:
        raise IndexStorageError("Stored embedding vector is invalid") from error


class SQLiteKeywordIndex:
    """Persist chunks and their mixed-language term frequencies."""

    def __init__(
        self,
        database_path: Path,
        embedding_model: EmbeddingModel,
        tokenizer: MixedLanguageTokenizer | None = None,
    ) -> None:
        self._database_path = database_path.expanduser().resolve()
        self._embedding_model = embedding_model
        self._tokenizer = tokenizer or MixedLanguageTokenizer()
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise IndexStorageError(
                f"Could not create index directory: {self._database_path.parent}"
            ) from error
        self._initialize_schema()

    def replace_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> bool:
        """Atomically replace all indexed data for one source."""
        for chunk in chunks:
            if chunk.document_id != document.document_id:
                raise IndexStorageError(
                    f"Chunk {chunk.chunk_id} belongs to a different document"
                )

        embeddings = self._prepare_embeddings(chunks)

        with closing(self._open()) as connection:
            try:
                connection.execute("BEGIN")
                existed = (
                    connection.execute(
                        "SELECT 1 FROM documents WHERE source = ?",
                        (document.source,),
                    ).fetchone()
                    is not None
                )
                if existed:
                    connection.execute(
                        "DELETE FROM documents WHERE source = ?",
                        (document.source,),
                    )

                connection.execute(
                    """
                    INSERT INTO documents(document_id, source, content_hash)
                    VALUES (?, ?, ?)
                    """,
                    (
                        document.document_id,
                        document.source,
                        sha256(document.content.encode("utf-8")).hexdigest(),
                    ),
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    self._insert_chunk(connection, chunk)
                    self._insert_embedding(connection, chunk.chunk_id, embedding)
                connection.commit()
            except sqlite3.Error as error:
                connection.rollback()
                raise IndexStorageError(
                    f"Could not replace document in index: {document.source}"
                ) from error

        return existed

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Return positive-scoring chunks ordered by BM25 relevance."""
        query_tokens = list(dict.fromkeys(self._tokenizer.tokenize(query)))
        if not query_tokens:
            raise InvalidQueryError("Query contains no searchable terms")

        with closing(self._open()) as connection:
            try:
                statistics = connection.execute(
                    "SELECT COUNT(*), COALESCE(AVG(token_count), 0) FROM chunks"
                ).fetchone()
                chunk_count = int(statistics[0])
                average_length = float(statistics[1])
                if chunk_count == 0:
                    raise KnowledgeBaseEmptyError(
                        "Knowledge base is empty. Ingest a document first."
                    )

                rows = self._matching_postings(connection, query_tokens)
                if not rows:
                    return []
                return self._rank_rows(
                    rows,
                    chunk_count=chunk_count,
                    average_length=average_length,
                    limit=limit,
                )
            except (
                sqlite3.Error,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as error:
                raise IndexStorageError(
                    f"Could not search knowledge index: {self._database_path}"
                ) from error

    @staticmethod
    def _matching_postings(
        connection: sqlite3.Connection,
        query_tokens: list[str],
    ) -> list[sqlite3.Row]:
        placeholders = ", ".join("?" for _ in query_tokens)
        query = f"""
            SELECT
                p.token,
                p.frequency,
                c.chunk_id,
                c.document_id,
                c.position,
                c.text,
                c.heading_path,
                c.token_count,
                d.source
            FROM postings AS p
            JOIN chunks AS c ON c.chunk_id = p.chunk_id
            JOIN documents AS d ON d.document_id = c.document_id
            WHERE p.token IN ({placeholders})
        """
        return list(connection.execute(query, query_tokens).fetchall())

    @staticmethod
    def _rank_rows(
        rows: list[sqlite3.Row],
        *,
        chunk_count: int,
        average_length: float,
        limit: int,
    ) -> list[SearchResult]:
        document_frequencies = Counter(str(row["token"]) for row in rows)
        scores: dict[str, float] = {}
        chunks: dict[str, tuple[Chunk, str]] = {}
        safe_average = average_length if average_length > 0 else 1.0

        for row in rows:
            token = str(row["token"])
            chunk_id = str(row["chunk_id"])
            frequency = int(row["frequency"])
            document_length = int(row["token_count"])
            document_frequency = document_frequencies[token]
            inverse_frequency = log(
                1
                + (chunk_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + _BM25_K1 * (
                1 - _BM25_B + _BM25_B * document_length / safe_average
            )
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                inverse_frequency * frequency * (_BM25_K1 + 1) / denominator
            )

            if chunk_id not in chunks:
                heading_value = json.loads(str(row["heading_path"]))
                if not isinstance(heading_value, list) or not all(
                    isinstance(item, str) for item in heading_value
                ):
                    raise ValueError("Stored heading_path is not a string list")
                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=str(row["document_id"]),
                    position=int(row["position"]),
                    text=str(row["text"]),
                    heading_path=tuple(heading_value),
                )
                chunks[chunk_id] = (chunk, str(row["source"]))

        ranked = [
            SearchResult(
                chunk=chunks[chunk_id][0], source=chunks[chunk_id][1], score=score
            )
            for chunk_id, score in scores.items()
            if score > 0
        ]
        ranked.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
        return ranked[:limit]

    def _insert_chunk(
        self,
        connection: sqlite3.Connection,
        chunk: Chunk,
    ) -> None:
        searchable_text = self._searchable_text(chunk)
        frequencies = Counter(self._tokenizer.tokenize(searchable_text))
        connection.execute(
            """
            INSERT INTO chunks(
                chunk_id, document_id, position, text, heading_path, token_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.position,
                chunk.text,
                json.dumps(chunk.heading_path, ensure_ascii=False),
                sum(frequencies.values()),
            ),
        )
        connection.executemany(
            "INSERT INTO postings(token, chunk_id, frequency) VALUES (?, ?, ?)",
            (
                (token, chunk.chunk_id, frequency)
                for token, frequency in frequencies.items()
            ),
        )

    def _prepare_embeddings(self, chunks: Sequence[Chunk]) -> list[Embedding]:
        texts = [self._searchable_text(chunk) for chunk in chunks]
        embeddings = self._embedding_model.embed_documents(texts)
        if len(embeddings) != len(chunks):
            raise EmbeddingError(
                f"Embedding model returned {len(embeddings)} vectors for "
                f"{len(chunks)} chunks"
            )

        dimensions: set[int] = set()
        for embedding in embeddings:
            if embedding.model_id != self._embedding_model.model_id:
                raise EmbeddingError(
                    "Embedding model identifier does not match configured model"
                )
            dimensions.add(len(embedding.values))
            _pack_vector(embedding.values)
        if len(dimensions) > 1:
            raise EmbeddingError("Chunk embeddings must have one common dimension")
        return embeddings

    @staticmethod
    def _searchable_text(chunk: Chunk) -> str:
        return f"{' '.join(chunk.heading_path)}\n{chunk.text}"

    @staticmethod
    def _insert_embedding(
        connection: sqlite3.Connection,
        chunk_id: str,
        embedding: Embedding,
    ) -> None:
        connection.execute(
            """
            INSERT INTO embeddings(chunk_id, model_id, dimensions, vector)
            VALUES (?, ?, ?, ?)
            """,
            (
                chunk_id,
                embedding.model_id,
                len(embedding.values),
                _pack_vector(embedding.values),
            ),
        )

    def _initialize_schema(self) -> None:
        with closing(self._open()) as connection:
            try:
                row = connection.execute("PRAGMA user_version").fetchone()
                version = int(row[0]) if row is not None else 0
                if version == 0:
                    connection.executescript(_SCHEMA)
                    connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                    connection.commit()
                elif version == 1:
                    connection.executescript(_EMBEDDINGS_SCHEMA)
                    connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                    connection.commit()
                elif version != _SCHEMA_VERSION:
                    raise IndexStorageError(
                        f"Unsupported knowledge index version {version}; "
                        f"expected {_SCHEMA_VERSION}"
                    )
            except sqlite3.Error as error:
                raise IndexStorageError(
                    f"Could not initialize knowledge index: {self._database_path}"
                ) from error

    def _open(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self._database_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as error:
            raise IndexStorageError(
                f"Could not open knowledge index: {self._database_path}"
            ) from error

"""Persistent SQLite inverted index for lexical retrieval."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Sequence
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from personal_knowledge_agent.domain.errors import IndexStorageError
from personal_knowledge_agent.domain.models import Chunk, Document
from personal_knowledge_agent.infrastructure.mixed_tokenizer import (
    MixedLanguageTokenizer,
)

_SCHEMA_VERSION = 1
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
"""


class SQLiteKeywordIndex:
    """Persist chunks and their mixed-language term frequencies."""

    def __init__(
        self,
        database_path: Path,
        tokenizer: MixedLanguageTokenizer | None = None,
    ) -> None:
        self._database_path = database_path.expanduser().resolve()
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
                for chunk in chunks:
                    self._insert_chunk(connection, chunk)
                connection.commit()
            except sqlite3.Error as error:
                connection.rollback()
                raise IndexStorageError(
                    f"Could not replace document in index: {document.source}"
                ) from error

        return existed

    def _insert_chunk(
        self,
        connection: sqlite3.Connection,
        chunk: Chunk,
    ) -> None:
        searchable_text = f"{' '.join(chunk.heading_path)}\n{chunk.text}"
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

    def _initialize_schema(self) -> None:
        with closing(self._open()) as connection:
            try:
                row = connection.execute("PRAGMA user_version").fetchone()
                version = int(row[0]) if row is not None else 0
                if version == 0:
                    connection.executescript(_SCHEMA)
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

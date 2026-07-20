"""Read one indexed source document from the knowledge database for graph building."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from personal_knowledge_agent.domain.errors import (
    GraphSourceNotFoundError,
    GraphStorageError,
)
from personal_knowledge_agent.domain.graph import GraphSourceDocument
from personal_knowledge_agent.domain.models import Chunk


class SQLiteGraphSourceReader:
    """Read one indexed source document without E5 or Markdown parsing.

    Resolves source paths (relative/absolute) and returns the current
    document metadata and ordered chunks from the knowledge database.
    """

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def load(self, source: Path) -> GraphSourceDocument:
        """Load the current source document and its chunks."""
        try:
            resolved = source.resolve()
        except OSError as error:
            raise GraphStorageError("Could not resolve graph source path") from error
        if not self._database_path.is_file():
            raise GraphSourceNotFoundError(
                f"Knowledge database not found: {self._database_path}"
            )

        try:
            with closing(sqlite3.connect(str(self._database_path))) as connection:
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                source_str = str(resolved)
                cursor.execute(
                    "SELECT document_id, content_hash FROM documents WHERE source = ?",
                    (source_str,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise GraphSourceNotFoundError(
                        f"Source not found in knowledge index: {source_str}"
                    )

                document_id = row["document_id"]
                content_hash = row["content_hash"]
                cursor.execute(
                    "SELECT chunk_id, position, text, heading_path "
                    "FROM chunks WHERE document_id = ? ORDER BY position ASC",
                    (document_id,),
                )
                chunk_rows = cursor.fetchall()

                chunks: list[Chunk] = []
                for chunk_row in chunk_rows:
                    heading_path = self._decode_heading_path(chunk_row["heading_path"])
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_row["chunk_id"],
                            document_id=document_id,
                            position=chunk_row["position"],
                            text=chunk_row["text"],
                            heading_path=heading_path,
                        )
                    )

                return GraphSourceDocument(
                    document_id=document_id,
                    source=resolved,
                    content_hash=content_hash,
                    chunks=chunks,
                )

        except GraphSourceNotFoundError:
            raise
        except (sqlite3.Error, TypeError, ValueError) as error:
            raise GraphStorageError(
                "Could not read graph source from knowledge index"
            ) from error

    @staticmethod
    def _decode_heading_path(value: object) -> tuple[str, ...]:
        if not isinstance(value, str):
            raise ValueError("heading_path is not text")
        decoded = json.loads(value)
        if not isinstance(decoded, list) or not all(
            isinstance(item, str) for item in decoded
        ):
            raise ValueError("heading_path is invalid")
        return tuple(decoded)

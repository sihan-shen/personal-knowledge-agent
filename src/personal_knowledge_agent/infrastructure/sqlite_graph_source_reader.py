"""Read one indexed source document from the knowledge database for graph building."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from personal_knowledge_agent.domain.errors import (
    GraphSourceNotFoundError,
    GraphStorageError,
)
from personal_knowledge_agent.domain.graph import GraphSourceDocument
from personal_knowledge_agent.domain.models import Chunk, make_chunk_id
from personal_knowledge_agent.domain.protocols import GraphSourceReader


class SQLiteGraphSourceReader:
    """Read one indexed source document without E5 or Markdown parsing.

    Resolves source paths (relative/absolute) and returns the current
    document metadata and ordered chunks from the knowledge database.
    """

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def load(self, source: Path) -> GraphSourceDocument:
        """Load the current source document and its chunks."""
        resolved = source.resolve()
        if not self._database_path.is_file():
            raise GraphSourceNotFoundError(
                f"Knowledge database not found: {self._database_path}"
            )

        try:
            connection = sqlite3.connect(str(self._database_path))
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()

            # Look up document by both the original and resolved source
            source_str = str(resolved)
            cursor.execute(
                "SELECT document_id, source, content_hash FROM documents "
                "WHERE source = ?",
                (source_str,),
            )
            row = cursor.fetchone()

            if row is None:
                connection.close()
                raise GraphSourceNotFoundError(
                    f"Source not found in knowledge index: {source_str}"
                )

            document_id = row["document_id"]
            content_hash = row["content_hash"]

            # Read chunks ordered by position
            cursor.execute(
                "SELECT chunk_id, position, text, heading_path "
                "FROM chunks WHERE document_id = ? "
                "ORDER BY position ASC",
                (document_id,),
            )
            chunk_rows = cursor.fetchall()
            connection.close()

            chunks: list[Chunk] = []
            for cr in chunk_rows:
                heading_raw = cr["heading_path"]
                try:
                    heading_path = tuple(json.loads(heading_raw))
                except (json.JSONDecodeError, TypeError) as error:
                    raise GraphStorageError(
                        "Could not read graph source from knowledge index"
                    ) from error

                if not all(isinstance(h, str) for h in heading_path):
                    raise GraphStorageError(
                        "Could not read graph source from knowledge index"
                    )

                chunk = Chunk(
                    chunk_id=cr["chunk_id"],
                    document_id=document_id,
                    position=cr["position"],
                    text=cr["text"],
                    heading_path=heading_path,
                )
                chunks.append(chunk)

            return GraphSourceDocument(
                document_id=document_id,
                source=resolved,
                content_hash=content_hash,
                chunks=chunks,
            )

        except GraphSourceNotFoundError:
            raise
        except sqlite3.Error as error:
            raise GraphStorageError(
                "Could not read graph source from knowledge index"
            ) from error

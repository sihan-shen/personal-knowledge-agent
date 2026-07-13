"""Load UTF-8 Markdown documents from local paths."""

from __future__ import annotations

from pathlib import Path

from personal_knowledge_agent.domain.errors import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from personal_knowledge_agent.domain.models import Document, make_document_id

_SUPPORTED_SUFFIXES = {".md", ".markdown"}


class MarkdownLoader:
    """Load and validate one local Markdown document."""

    def load(self, path: Path) -> Document:
        """Read a Markdown file as UTF-8 and return its domain record."""
        resolved = path.expanduser().resolve()
        if resolved.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise UnsupportedDocumentError(
                f"Only Markdown files (.md, .markdown) are supported: {resolved}"
            )
        if not resolved.is_file():
            raise DocumentReadError(f"Path is not a readable file: {resolved}")

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise DocumentReadError(
                f"Document is not valid UTF-8: {resolved}"
            ) from error
        except OSError as error:
            raise DocumentReadError(f"Could not read document: {resolved}") from error

        if not content.strip():
            raise EmptyDocumentError(f"Document is empty: {resolved}")

        source = str(resolved)
        return Document(make_document_id(source), source, content)

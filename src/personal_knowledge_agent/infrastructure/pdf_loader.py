"""Load extractable text from local PDF documents."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from personal_knowledge_agent.domain.errors import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from personal_knowledge_agent.domain.models import Document, make_document_id


class PdfLoader:
    """Load text-based PDF pages into heading-structured documents."""

    def __init__(self, reader_factory: Callable[[Path], PdfReader] = PdfReader) -> None:
        self._reader_factory = reader_factory

    def load(self, path: Path) -> Document:
        """Extract nonempty pages while preserving their one-based page numbers."""
        resolved = path.expanduser().resolve()
        if resolved.suffix.lower() != ".pdf":
            raise UnsupportedDocumentError(
                f"Only PDF files (.pdf) are supported: {resolved}"
            )
        if not resolved.is_file():
            raise DocumentReadError(f"Path is not a readable file: {resolved}")

        try:
            reader = self._reader_factory(resolved)
            if reader.is_encrypted:
                raise DocumentReadError(
                    f"PDF is encrypted and cannot be ingested: {resolved}"
                )
            sections = [
                f"## Page {number}\n\n{text}"
                for number, page in enumerate(reader.pages, start=1)
                if (text := (page.extract_text() or "").strip())
            ]
        except DocumentReadError:
            raise
        except (OSError, PdfReadError) as error:
            raise DocumentReadError(
                f"Could not read PDF document: {resolved}"
            ) from error

        if not sections:
            raise EmptyDocumentError(
                f"PDF has no extractable text; OCR is required before ingestion: {resolved}"
            )

        source = str(resolved)
        return Document(make_document_id(source), source, "\n\n".join(sections))

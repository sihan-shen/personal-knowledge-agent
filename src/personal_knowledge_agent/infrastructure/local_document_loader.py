"""Dispatch supported local document paths to their format loaders."""

from __future__ import annotations

from pathlib import Path

from personal_knowledge_agent.domain.errors import UnsupportedDocumentError
from personal_knowledge_agent.domain.models import Document
from personal_knowledge_agent.domain.protocols import DocumentLoader
from personal_knowledge_agent.infrastructure.markdown_loader import MarkdownLoader
from personal_knowledge_agent.infrastructure.pdf_loader import PdfLoader


class LocalDocumentLoader:
    """Load supported local Markdown or PDF documents by filename suffix."""

    def __init__(
        self,
        *,
        markdown_loader: DocumentLoader | None = None,
        pdf_loader: DocumentLoader | None = None,
    ) -> None:
        self._markdown_loader = markdown_loader or MarkdownLoader()
        self._pdf_loader = pdf_loader or PdfLoader()

    def load(self, path: Path) -> Document:
        """Select the loader without exposing a format switch in the CLI."""
        suffix = path.expanduser().suffix.lower()
        if suffix in {".md", ".markdown"}:
            return self._markdown_loader.load(path)
        if suffix == ".pdf":
            return self._pdf_loader.load(path)
        raise UnsupportedDocumentError(
            f"Only Markdown (.md, .markdown) and PDF (.pdf) are supported: {path}"
        )

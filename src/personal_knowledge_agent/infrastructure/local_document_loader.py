"""Dispatch supported local document paths and public web URLs to loaders."""

from __future__ import annotations

from pathlib import Path

from personal_knowledge_agent.domain.errors import UnsupportedDocumentError
from personal_knowledge_agent.domain.models import Document
from personal_knowledge_agent.domain.protocols import DocumentLoader
from personal_knowledge_agent.infrastructure.markdown_loader import MarkdownLoader
from personal_knowledge_agent.infrastructure.pdf_loader import PdfLoader
from personal_knowledge_agent.infrastructure.web_loader import WebLoader


class LocalDocumentLoader:
    """Load supported local Markdown/PDF documents or one public web page."""

    def __init__(
        self,
        *,
        markdown_loader: DocumentLoader[Path] | None = None,
        pdf_loader: DocumentLoader[Path] | None = None,
        web_loader: DocumentLoader[str] | None = None,
    ) -> None:
        self._markdown_loader = markdown_loader or MarkdownLoader()
        self._pdf_loader = pdf_loader or PdfLoader()
        self._web_loader = web_loader or WebLoader()

    def load(self, source: Path | str) -> Document:
        """Select the loader without exposing a format switch in the CLI."""
        if isinstance(source, str):
            return self._web_loader.load(source)

        path = source
        suffix = path.expanduser().suffix.lower()
        if suffix in {".md", ".markdown"}:
            return self._markdown_loader.load(path)
        if suffix == ".pdf":
            return self._pdf_loader.load(path)
        raise UnsupportedDocumentError(
            f"Only Markdown (.md, .markdown) and PDF (.pdf) are supported: {path}"
        )

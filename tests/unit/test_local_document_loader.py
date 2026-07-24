"""Unit tests for local document format dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import UnsupportedDocumentError
from personal_knowledge_agent.domain.models import Document
from personal_knowledge_agent.infrastructure.local_document_loader import (
    LocalDocumentLoader,
)


class RecordingLoader:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def load(self, path: Path) -> Document:
        self.paths.append(path)
        return Document("document", str(path), "content")


def test_routes_pdf_and_markdown_suffixes_to_their_loaders(tmp_path: Path) -> None:
    markdown = RecordingLoader()
    pdf = RecordingLoader()
    loader = LocalDocumentLoader(markdown_loader=markdown, pdf_loader=pdf)

    loader.load(tmp_path / "note.md")
    loader.load(tmp_path / "paper.pdf")

    assert markdown.paths == [tmp_path / "note.md"]
    assert pdf.paths == [tmp_path / "paper.pdf"]


def test_routes_http_url_to_web_loader_without_converting_it_to_a_path() -> None:
    web = RecordingWebLoader()
    loader = LocalDocumentLoader(web_loader=web)

    loader.load("https://public.example/articles/rag?language=en")

    assert web.urls == ["https://public.example/articles/rag?language=en"]


def test_rejects_unknown_local_document_suffix(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedDocumentError):
        LocalDocumentLoader().load(tmp_path / "note.txt")


class RecordingWebLoader:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def load(self, url: str) -> Document:
        self.urls.append(url)
        return Document("web-document", url, "content")

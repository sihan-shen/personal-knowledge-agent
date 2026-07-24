"""Unit tests for local text-based PDF ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf.errors import PdfReadError

from personal_knowledge_agent.domain.errors import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from personal_knowledge_agent.infrastructure.pdf_loader import PdfLoader


class FakePage:
    """Expose deterministic text without requiring a fixture PDF."""

    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class FakeReader:
    """Minimal PdfReader-compatible fake for loader behavior tests."""

    def __init__(self, pages: list[FakePage], *, encrypted: bool = False) -> None:
        self.pages = pages
        self.is_encrypted = encrypted


def _pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "rag.pdf"
    path.write_bytes(b"%PDF-placeholder")
    return path


def test_loads_nonempty_pages_with_page_headings(tmp_path: Path) -> None:
    loader = PdfLoader(
        reader_factory=lambda _: FakeReader(
            [FakePage("First page"), FakePage("Second page")]
        )
    )

    document = loader.load(_pdf_path(tmp_path))

    assert document.source == str(_pdf_path(tmp_path).resolve())
    assert document.content == "## Page 1\n\nFirst page\n\n## Page 2\n\nSecond page"


def test_skips_blank_pages_without_renumbering(tmp_path: Path) -> None:
    loader = PdfLoader(
        reader_factory=lambda _: FakeReader(
            [FakePage("First page"), FakePage("  \n"), FakePage("Third page")]
        )
    )

    document = loader.load(_pdf_path(tmp_path))

    assert document.content == "## Page 1\n\nFirst page\n\n## Page 3\n\nThird page"


def test_rejects_non_pdf_suffix(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("text", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError):
        PdfLoader().load(path)


def test_rejects_missing_pdf_path(tmp_path: Path) -> None:
    with pytest.raises(DocumentReadError):
        PdfLoader().load(tmp_path / "missing.pdf")


def test_rejects_encrypted_pdf(tmp_path: Path) -> None:
    loader = PdfLoader(reader_factory=lambda _: FakeReader([], encrypted=True))

    with pytest.raises(DocumentReadError, match="encrypted"):
        loader.load(_pdf_path(tmp_path))


def test_maps_parser_errors_to_document_read_error(tmp_path: Path) -> None:
    def raise_read_error(_: Path) -> FakeReader:
        raise PdfReadError("invalid structure")

    loader = PdfLoader(reader_factory=raise_read_error)

    with pytest.raises(DocumentReadError):
        loader.load(_pdf_path(tmp_path))


def test_explains_ocr_requirement_for_textless_pdf(tmp_path: Path) -> None:
    loader = PdfLoader(reader_factory=lambda _: FakeReader([FakePage(None)]))

    with pytest.raises(EmptyDocumentError, match="OCR"):
        loader.load(_pdf_path(tmp_path))

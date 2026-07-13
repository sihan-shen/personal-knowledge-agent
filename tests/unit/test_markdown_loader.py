"""Tests for safe Markdown loading."""

from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from personal_knowledge_agent.domain.models import make_document_id
from personal_knowledge_agent.infrastructure.markdown_loader import MarkdownLoader


def test_loads_utf8_markdown_with_resolved_source(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# RAG\n\n检索增强生成", encoding="utf-8")

    document = MarkdownLoader().load(path)

    resolved = str(path.resolve())
    assert document.source == resolved
    assert document.document_id == make_document_id(resolved)
    assert document.content == "# RAG\n\n检索增强生成"


def test_accepts_markdown_extension_case_insensitively(tmp_path: Path) -> None:
    path = tmp_path / "notes.MARKDOWN"
    path.write_text("Agent notes", encoding="utf-8")

    assert MarkdownLoader().load(path).content == "Agent notes"


def test_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(DocumentReadError, match="not a readable file"):
        MarkdownLoader().load(tmp_path / "missing.md")


def test_rejects_directory_input(tmp_path: Path) -> None:
    path = tmp_path / "folder.md"
    path.mkdir()

    with pytest.raises(DocumentReadError, match="not a readable file"):
        MarkdownLoader().load(path)


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("notes", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError, match="Markdown"):
        MarkdownLoader().load(path)


def test_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(DocumentReadError, match="UTF-8"):
        MarkdownLoader().load(path)


def test_rejects_whitespace_only_document(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(EmptyDocumentError, match="empty"):
        MarkdownLoader().load(path)

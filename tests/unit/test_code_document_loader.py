"""Tests for explicit local code-document loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import DocumentReadError, EmptyDocumentError
from personal_knowledge_agent.infrastructure.code_document_loader import (
    CodeDocumentLoader,
)


def test_python_declarations_become_page_like_headings(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    source.write_text("class Agent:\n    pass\n\ndef answer():\n    return 1\n")

    document = CodeDocumentLoader().load(source)

    assert "# example.py" in document.content
    assert "## Agent" in document.content
    assert "## answer" in document.content


def test_text_document_gets_file_heading(tmp_path: Path) -> None:
    source = tmp_path / "config.toml"
    source.write_text("[tool]\nname = 'pka'\n")

    assert CodeDocumentLoader().load(source).content.startswith("# config.toml")


def test_rejects_binary_and_empty_documents(tmp_path: Path) -> None:
    binary = tmp_path / "data.json"
    binary.write_bytes(b"{\x00}")
    with pytest.raises(DocumentReadError):
        CodeDocumentLoader().load(binary)

    empty = tmp_path / "empty.txt"
    empty.write_text("   ")
    with pytest.raises(EmptyDocumentError):
        CodeDocumentLoader().load(empty)

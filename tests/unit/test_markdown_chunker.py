"""Tests for heading-aware Markdown chunking."""

import pytest

from personal_knowledge_agent.domain.models import Document, make_document_id
from personal_knowledge_agent.infrastructure.markdown_chunker import MarkdownChunker


def make_document(content: str) -> Document:
    source = "/notes/example.md"
    return Document(make_document_id(source), source, content)


def test_tracks_nested_heading_paths() -> None:
    document = make_document(
        "# RAG\n\nOverview.\n\n## Retrieval\n\nEvidence selection."
    )

    chunks = MarkdownChunker().split(document)

    assert [chunk.heading_path for chunk in chunks] == [
        ("RAG",),
        ("RAG", "Retrieval"),
    ]
    assert [chunk.text for chunk in chunks] == ["Overview.", "Evidence selection."]


def test_aggregates_paragraphs_below_target() -> None:
    document = make_document("# Notes\n\nFirst paragraph.\n\nSecond paragraph.")

    chunks = MarkdownChunker(target_chars=100).split(document)

    assert len(chunks) == 1
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."


def test_splits_before_next_block_exceeds_target() -> None:
    document = make_document("# Notes\n\nalpha alpha\n\nbeta beta")

    chunks = MarkdownChunker(target_chars=15).split(document)

    assert [chunk.text for chunk in chunks] == ["alpha alpha", "beta beta"]


def test_preserves_fenced_code_as_one_block() -> None:
    document = make_document(
        "# Code\n\nBefore.\n\n```python\n# not a heading\n\nprint('x')\n```\n\nAfter."
    )

    chunks = MarkdownChunker(target_chars=20).split(document)

    code_chunks = [chunk for chunk in chunks if "```python" in chunk.text]
    assert len(code_chunks) == 1
    assert "# not a heading\n\nprint('x')" in code_chunks[0].text
    assert code_chunks[0].heading_path == ("Code",)


def test_keeps_oversized_paragraph_intact() -> None:
    paragraph = "A" * 40

    chunks = MarkdownChunker(target_chars=10).split(make_document(paragraph))

    assert [chunk.text for chunk in chunks] == [paragraph]


def test_empty_sections_do_not_create_chunks() -> None:
    document = make_document("# Empty\n\n## Full\n\nActual content.")

    chunks = MarkdownChunker().split(document)

    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Empty", "Full")


def test_preserves_literal_hash_at_end_of_heading() -> None:
    chunks = MarkdownChunker().split(make_document("# C#\n\nLanguage notes."))

    assert chunks[0].heading_path == ("C#",)


def test_output_is_deterministic_and_has_no_overlap() -> None:
    document = make_document("# Notes\n\none one\n\ntwo two\n\nthree three")
    chunker = MarkdownChunker(target_chars=12)

    first = chunker.split(document)
    second = chunker.split(document)

    assert first == second
    combined = "\n".join(chunk.text for chunk in first)
    assert combined.count("one one") == 1
    assert combined.count("two two") == 1
    assert [chunk.position for chunk in first] == list(range(len(first)))


def test_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError, match="target_chars"):
        MarkdownChunker(target_chars=0)

"""Tests for retrieval domain models."""

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    ChatMessage,
    Chunk,
    Document,
    Embedding,
    Evidence,
    GroundedAnswer,
    IngestionResult,
    MessageRole,
    ModelOutput,
    SearchMode,
    SearchResult,
    make_chunk_id,
    make_document_id,
)


def test_document_id_is_stable_and_source_specific() -> None:
    assert make_document_id("/notes/rag.md") == make_document_id("/notes/rag.md")
    assert make_document_id("/notes/rag.md") != make_document_id("/notes/agent.md")


def test_chunk_id_is_stable_and_position_specific() -> None:
    document_id = make_document_id("/notes/rag.md")

    assert make_chunk_id(document_id, 0) == make_chunk_id(document_id, 0)
    assert make_chunk_id(document_id, 0) != make_chunk_id(document_id, 1)


def test_models_are_frozen() -> None:
    document = Document("doc-id", "/notes/rag.md", "RAG notes")

    with pytest.raises(FrozenInstanceError):
        document.content = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("value", ["", "   "])
def test_document_rejects_blank_identifiers(value: str) -> None:
    with pytest.raises(ValueError, match="document_id"):
        Document(value, "/notes/rag.md", "RAG notes")


def test_chunk_rejects_negative_position() -> None:
    with pytest.raises(ValueError, match="position"):
        Chunk("chunk-id", "doc-id", -1, "text")


@pytest.mark.parametrize("value", ["", "\n\t"])
def test_chunk_rejects_blank_text(value: str) -> None:
    with pytest.raises(ValueError, match="text"):
        Chunk("chunk-id", "doc-id", 0, value)


def test_result_records_keep_retrieval_metadata() -> None:
    chunk = Chunk("chunk-id", "doc-id", 0, "RAG text", ("RAG",))

    result = SearchResult(chunk, "/notes/rag.md", 1.5)
    ingestion = IngestionResult("/notes/rag.md", 1, False)

    assert result.chunk == chunk
    assert result.score == 1.5
    assert ingestion.chunk_count == 1


def test_embedding_is_immutable_and_preserves_vector_values() -> None:
    embedding = Embedding("test-model", (0.6, 0.8))

    assert embedding.model_id == "test-model"
    assert embedding.values == (0.6, 0.8)
    with pytest.raises(FrozenInstanceError):
        embedding.model_id = "other-model"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("model_id", "values", "message"),
    [
        ("   ", (1.0,), "model_id"),
        ("test-model", (), "values"),
        ("test-model", (nan,), "finite"),
        ("test-model", (inf,), "finite"),
    ],
)
def test_embedding_rejects_invalid_values(
    model_id: str,
    values: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Embedding(model_id, values)


def test_search_modes_are_stable_cli_values() -> None:
    assert [mode.value for mode in SearchMode] == ["lexical", "semantic", "hybrid"]


def test_message_roles_and_answer_statuses_are_stable_values() -> None:
    assert [role.value for role in MessageRole] == ["system", "user", "assistant"]
    assert [status.value for status in AnswerStatus] == [
        "answered",
        "insufficient_evidence",
    ]


def test_chat_message_is_immutable_and_preserves_content() -> None:
    message = ChatMessage(MessageRole.SYSTEM, "Answer only from evidence.")

    assert message.role is MessageRole.SYSTEM
    assert message.content == "Answer only from evidence."
    with pytest.raises(FrozenInstanceError):
        message.content = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_chat_message_rejects_blank_content(content: str) -> None:
    with pytest.raises(ValueError, match="content"):
        ChatMessage(MessageRole.USER, content)


def test_evidence_preserves_local_retrieval_metadata() -> None:
    evidence = Evidence(
        "S1",
        "chunk-id",
        "/notes/rag.md",
        ("RAG", "Retrieval"),
        "retrieval evidence",
        0.75,
    )

    assert evidence.citation_id == "S1"
    assert evidence.chunk_id == "chunk-id"
    assert evidence.source == "/notes/rag.md"
    assert evidence.heading_path == ("RAG", "Retrieval")
    assert evidence.text == "retrieval evidence"
    assert evidence.retrieval_score == 0.75


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("citation_id", ("", "chunk", "/notes.md", "text")),
        ("chunk_id", ("S1", " ", "/notes.md", "text")),
        ("source", ("S1", "chunk", "\n", "text")),
        ("text", ("S1", "chunk", "/notes.md", "\t")),
    ],
)
def test_evidence_rejects_blank_required_text(
    field: str,
    values: tuple[str, str, str, str],
) -> None:
    citation_id, chunk_id, source, text = values

    with pytest.raises(ValueError, match=field):
        Evidence(citation_id, chunk_id, source, (), text, 1.0)


@pytest.mark.parametrize("score", [nan, inf])
def test_evidence_rejects_non_finite_score(score: float) -> None:
    with pytest.raises(ValueError, match="retrieval_score"):
        Evidence("S1", "chunk", "/notes.md", (), "text", score)


@pytest.mark.parametrize(
    ("model_id", "content", "finish_reason", "field"),
    [
        ("", "{}", "stop", "model_id"),
        ("model", " ", "stop", "content"),
        ("model", "{}", "\n", "finish_reason"),
    ],
)
def test_model_output_rejects_blank_required_text(
    model_id: str,
    content: str,
    finish_reason: str,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        ModelOutput(model_id, content, finish_reason)


def test_grounded_answer_is_immutable_and_preserves_all_evidence() -> None:
    evidence = Evidence("S1", "chunk", "/notes.md", (), "text", 0.5)
    answer = GroundedAnswer(
        AnswerStatus.ANSWERED,
        "Supported answer. [S1]",
        ("S1",),
        (evidence,),
        "model-id",
    )

    assert answer.evidence == (evidence,)
    assert answer.model_id == "model-id"
    with pytest.raises(FrozenInstanceError):
        answer.answer = "changed"  # type: ignore[misc]


def test_grounded_answer_rejects_blank_answer() -> None:
    with pytest.raises(ValueError, match="answer"):
        GroundedAnswer(AnswerStatus.INSUFFICIENT_EVIDENCE, " ", (), (), None)

"""Tests for strict structured answers with document and memory citations."""

from datetime import UTC, datetime

import pytest

from personal_knowledge_agent.domain.errors import (
    InvalidAnswerSchemaError,
    InvalidCitationError,
)
from personal_knowledge_agent.domain.memory import (
    MemoryEvidence,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    memory_content_hash,
)
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    Evidence,
    ModelOutput,
)
from personal_knowledge_agent.infrastructure.structured_contextual_answer_parser import (  # noqa: E501
    parse_contextual_answer,
)

_NOW = datetime(2026, 7, 18, tzinfo=UTC)


def contexts() -> tuple[tuple[Evidence, ...], tuple[MemoryEvidence, ...]]:
    document = Evidence("S1", "chunk", "/notes/rag.md", (), "RAG evidence", 0.9)
    content = "Prefer concise answers"
    record = MemoryRecord(
        "mem_00000000000000000000000000000001",
        MemoryType.PREFERENCE,
        "answer_style",
        content,
        memory_content_hash(content),
        MemoryStatus.ACTIVE,
        None,
        None,
        _NOW,
        _NOW,
    )
    return (document,), (MemoryEvidence("M1", record, 0.02),)


def output(content: str) -> ModelOutput:
    return ModelOutput("test-model", content, "stop")


@pytest.mark.parametrize(
    ("body", "citations"),
    [
        ("RAG uses evidence. [S1]", '["S1"]'),
        ("You prefer concise answers. [M1]", '["M1"]'),
        ("Use RAG [S1] and stay concise [M1].", '["S1","M1"]'),
    ],
)
def test_parser_accepts_s_m_and_mixed_answers(body: str, citations: str) -> None:
    documents, memories = contexts()

    answer = parse_contextual_answer(
        output(f'{{"status":"answered","answer":"{body}","citation_ids":{citations}}}'),
        documents,
        memories,
    )

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.citation_ids == tuple(
        item.strip('"') for item in citations[1:-1].split(",")
    )
    assert answer.evidence == documents
    assert answer.model_id == "test-model"


def test_parser_accepts_insufficient_without_citations() -> None:
    documents, memories = contexts()
    answer = parse_contextual_answer(
        output(
            '{"status":"insufficient_evidence","answer":"Not enough.",'
            '"citation_ids":[]}'
        ),
        documents,
        memories,
    )
    assert answer.citation_ids == ()


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '```json\n{"status":"answered"}\n```',
        "[]",
        '{"status":"answered","answer":"x","citation_ids":[],"extra":1}',
        '{"status":"answered","status":"insufficient_evidence","answer":"x","citation_ids":[]}',
        '{"status":1,"answer":"x","citation_ids":[]}',
        '{"status":"answered","answer":" ","citation_ids":[]}',
    ],
)
def test_parser_rejects_invalid_schema(content: str) -> None:
    documents, memories = contexts()
    with pytest.raises(InvalidAnswerSchemaError):
        parse_contextual_answer(output(content), documents, memories)


@pytest.mark.parametrize(
    "content",
    [
        '{"status":"answered","answer":"No marker","citation_ids":[]}',
        '{"status":"answered","answer":"Unknown [M2]","citation_ids":["M2"]}',
        '{"status":"answered","answer":"Injected [S99] [M99]",'
        '"citation_ids":["S99","M99"]}',
        '{"status":"answered","answer":"Bad [M0]","citation_ids":["M0"]}',
        '{"status":"answered","answer":"Memory [M1] source [S1]",'
        '"citation_ids":["S1","M1"]}',
        '{"status":"answered","answer":"Memory [M1]","citation_ids":["M1","M1"]}',
        '{"status":"insufficient_evidence","answer":"No [M1]","citation_ids":[]}',
    ],
)
def test_parser_rejects_invalid_citations(content: str) -> None:
    documents, memories = contexts()
    with pytest.raises(InvalidCitationError):
        parse_contextual_answer(output(content), documents, memories)

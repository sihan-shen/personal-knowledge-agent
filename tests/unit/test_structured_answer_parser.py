"""Tests for strict grounded-answer JSON and citation validation."""

import json

import pytest

from personal_knowledge_agent.domain.errors import (
    InvalidAnswerSchemaError,
    InvalidCitationError,
)
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    Evidence,
    ModelOutput,
)
from personal_knowledge_agent.infrastructure.structured_answer_parser import (
    parse_grounded_answer,
)


def make_evidence(citation_id: str) -> Evidence:
    return Evidence(
        citation_id,
        f"chunk-{citation_id}",
        f"/notes/{citation_id}.md",
        ("Notes", citation_id),
        f"evidence {citation_id}",
        0.5,
    )


def output(payload: object, model_id: str = "provider-model") -> ModelOutput:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return ModelOutput(model_id, content, "stop")


def test_parses_answered_output_and_preserves_all_supplied_evidence() -> None:
    evidence = (make_evidence("S1"), make_evidence("S2"), make_evidence("S3"))
    model_output = output(
        {
            "status": "answered",
            "answer": "First fact. [S1] Second fact. [S2] Repeat. [S1]",
            "citation_ids": ["S1", "S2"],
        }
    )

    answer = parse_grounded_answer(model_output, evidence)

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.answer == "First fact. [S1] Second fact. [S2] Repeat. [S1]"
    assert answer.citation_ids == ("S1", "S2")
    assert answer.evidence == evidence
    assert answer.model_id == "provider-model"


def test_parses_insufficient_evidence_output() -> None:
    answer = parse_grounded_answer(
        output(
            {
                "status": "insufficient_evidence",
                "answer": "The knowledge base does not contain enough evidence.",
                "citation_ids": [],
            }
        ),
        (make_evidence("S1"),),
    )

    assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.citation_ids == ()
    assert answer.model_id == "provider-model"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "valid JSON"),
        ("```json\n{}\n```", "valid JSON"),
        ([], "JSON object"),
        ({"status": "answered", "answer": "x [S1]"}, "exactly"),
        (
            {
                "status": "answered",
                "answer": "x [S1]",
                "citation_ids": ["S1"],
                "extra": True,
            },
            "exactly",
        ),
        (
            {"status": 1, "answer": "x [S1]", "citation_ids": ["S1"]},
            "status must be a string",
        ),
        (
            {"status": "answered", "answer": 1, "citation_ids": ["S1"]},
            "answer must be a string",
        ),
        (
            {"status": "unknown", "answer": "x [S1]", "citation_ids": ["S1"]},
            "status",
        ),
        (
            {"status": "answered", "answer": "x [S1]", "citation_ids": "S1"},
            "citation_ids must be a list",
        ),
        (
            {"status": "answered", "answer": "x [S1]", "citation_ids": [1]},
            "citation_ids must contain strings",
        ),
        (
            {"status": "answered", "answer": "   ", "citation_ids": ["S1"]},
            "answer must not be blank",
        ),
    ],
)
def test_rejects_invalid_answer_schema(payload: object, message: str) -> None:
    with pytest.raises(InvalidAnswerSchemaError, match=message):
        parse_grounded_answer(output(payload), (make_evidence("S1"),))


@pytest.mark.parametrize(
    ("answer", "citation_ids", "message"),
    [
        ("No citation.", [], "at least one citation"),
        ("Unknown. [S99]", ["S99"], "unknown citation"),
        ("Duplicate. [S1]", ["S1", "S1"], "unique"),
        ("Body only. [S1] [S2]", ["S1"], "match"),
        ("Declared only. [S1]", ["S1", "S2"], "match"),
        ("Wrong order. [S1] [S2]", ["S2", "S1"], "match"),
    ],
)
def test_rejects_invalid_answered_citations(
    answer: str,
    citation_ids: list[str],
    message: str,
) -> None:
    payload = {
        "status": "answered",
        "answer": answer,
        "citation_ids": citation_ids,
    }

    with pytest.raises(InvalidCitationError, match=message):
        parse_grounded_answer(
            output(payload),
            (make_evidence("S1"), make_evidence("S2")),
        )


@pytest.mark.parametrize(
    ("answer", "citation_ids"),
    [
        ("Cannot answer. [S1]", []),
        ("Cannot answer.", ["S1"]),
    ],
)
def test_insufficient_evidence_must_not_contain_citations(
    answer: str,
    citation_ids: list[str],
) -> None:
    payload = {
        "status": "insufficient_evidence",
        "answer": answer,
        "citation_ids": citation_ids,
    }

    with pytest.raises(InvalidCitationError, match="must not cite"):
        parse_grounded_answer(output(payload), (make_evidence("S1"),))

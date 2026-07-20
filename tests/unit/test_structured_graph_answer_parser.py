"""Tests for strict graph-grounded answer JSON and citation validation."""

from __future__ import annotations

import json

import pytest

from personal_knowledge_agent.domain.errors import (
    InvalidAnswerSchemaError,
    InvalidCitationError,
)
from personal_knowledge_agent.domain.graph import (
    EntityType,
    GraphProvenance,
    RelationType,
)
from personal_knowledge_agent.domain.graph_answer import GraphEvidence
from personal_knowledge_agent.domain.models import AnswerStatus, ModelOutput
from personal_knowledge_agent.infrastructure.structured_graph_answer_parser import (
    parse_graph_grounded_answer,
)


def evidence(citation_id: str) -> GraphEvidence:
    return GraphEvidence(
        citation_id,
        EntityType.METHOD,
        f"RAG {citation_id}",
        RelationType.USES,
        EntityType.TECHNOLOGY,
        f"Vector Retrieval {citation_id}",
        GraphProvenance(
            f"/notes/{citation_id}.md",
            f"chunk-{citation_id}",
            int(citation_id[1:]),
            ("Graph", citation_id),
            f"evidence {citation_id}",
        ),
    )


def output(payload: object) -> ModelOutput:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return ModelOutput("fake-provider-model", content, "stop")


def test_parses_answered_output_and_selects_cited_evidence_in_order() -> None:
    answer = parse_graph_grounded_answer(
        output(
            {
                "status": "answered",
                "answer": "First relation. [G2] Second relation. [G1] Again. [G2]",
                "citation_ids": ["G2", "G1"],
            }
        ),
        (evidence("G1"), evidence("G2"), evidence("G3")),
    )

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.citation_ids == ("G2", "G1")
    assert tuple(item.citation_id for item in answer.evidence) == ("G2", "G1")
    assert answer.model_id == "fake-provider-model"


def test_parses_model_refusal_with_no_citations_or_evidence() -> None:
    answer = parse_graph_grounded_answer(
        output(
            {
                "status": "insufficient_evidence",
                "answer": "The graph does not contain enough relationship evidence.",
                "citation_ids": [],
            }
        ),
        (evidence("G1"),),
    )

    assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.citation_ids == ()
    assert answer.evidence == ()
    assert answer.model_id == "fake-provider-model"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "valid JSON"),
        ("```json\\n{}\\n```", "valid JSON"),
        ([], "JSON object"),
        (
            '{"status":"answered","status":"insufficient_evidence",'
            '"answer":"x","citation_ids":[]}',
            "duplicate field",
        ),
        ({"status": "answered", "answer": "x [G1]"}, "exactly"),
        (
            {
                "status": "answered",
                "answer": "x [G1]",
                "citation_ids": ["G1"],
                "extra": True,
            },
            "exactly",
        ),
        ({"status": 1, "answer": "x [G1]", "citation_ids": ["G1"]}, "status"),
        (
            {"status": "answered", "answer": 1, "citation_ids": ["G1"]},
            "answer",
        ),
        (
            {"status": "answered", "answer": "   ", "citation_ids": ["G1"]},
            "blank",
        ),
        (
            {"status": "answered", "answer": "x [G1]", "citation_ids": "G1"},
            "list",
        ),
        (
            {"status": "answered", "answer": "x [G1]", "citation_ids": [1]},
            "strings",
        ),
        (
            {"status": "unsupported", "answer": "x [G1]", "citation_ids": ["G1"]},
            "Unsupported",
        ),
    ],
)
def test_rejects_invalid_answer_schema(payload: object, message: str) -> None:
    with pytest.raises(InvalidAnswerSchemaError, match=message):
        parse_graph_grounded_answer(output(payload), (evidence("G1"),))


@pytest.mark.parametrize(
    ("answer_text", "citation_ids", "message"),
    [
        ("No citation.", [], "needs a citation"),
        ("Unknown. [G99]", ["G99"], "unknown citation"),
        ("Duplicate. [G1]", ["G1", "G1"], "unique"),
        ("Body only. [G1] [G2]", ["G1"], "match"),
        ("Declared only. [G1]", ["G1", "G2"], "match"),
        ("Wrong order. [G1] [G2]", ["G2", "G1"], "first-use"),
        ("Document evidence. [S1]", ["G1"], "only valid G evidence"),
        ("Memory evidence. [M1]", ["G1"], "only valid G evidence"),
        (
            "Declared document evidence. [S1]",
            ["S1"],
            "only valid G evidence",
        ),
        (
            "Declared memory evidence. [M1]",
            ["M1"],
            "only valid G evidence",
        ),
    ],
)
def test_rejects_invalid_answered_citations(
    answer_text: str,
    citation_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(InvalidCitationError, match=message):
        parse_graph_grounded_answer(
            output(
                {
                    "status": "answered",
                    "answer": answer_text,
                    "citation_ids": citation_ids,
                }
            ),
            (evidence("G1"), evidence("G2")),
        )


@pytest.mark.parametrize("malformed_marker", ["[Gx]", "[G1 ]", "[G[1]]"])
def test_answered_output_rejects_malformed_graph_family_body_marker(
    malformed_marker: str,
) -> None:
    with pytest.raises(InvalidCitationError, match="only valid G evidence"):
        parse_graph_grounded_answer(
            output(
                {
                    "status": "answered",
                    "answer": f"Supported. [G1] Also malformed. {malformed_marker}",
                    "citation_ids": ["G1"],
                }
            ),
            (evidence("G1"),),
        )


@pytest.mark.parametrize("malformed_marker", ["[Gx]", "[G1 ]", "[G[1]]"])
def test_insufficient_output_rejects_malformed_graph_family_body_marker(
    malformed_marker: str,
) -> None:
    with pytest.raises(InvalidCitationError, match="only valid G evidence"):
        parse_graph_grounded_answer(
            output(
                {
                    "status": "insufficient_evidence",
                    "answer": f"Cannot answer. {malformed_marker}",
                    "citation_ids": [],
                }
            ),
            (evidence("G1"),),
        )


def test_answered_output_ignores_non_citation_bracketed_prose() -> None:
    answer = parse_graph_grounded_answer(
        output(
            {
                "status": "answered",
                "answer": "Supported. [G1] [additional context]",
                "citation_ids": ["G1"],
            }
        ),
        (evidence("G1"),),
    )

    assert answer.citation_ids == ("G1",)


@pytest.mark.parametrize(
    ("answer_text", "citation_ids", "message"),
    [
        ("Cannot answer. [G1]", [], "must not cite"),
        ("Cannot answer. [S1]", [], "only valid G evidence"),
        ("Cannot answer. [M1]", [], "only valid G evidence"),
        ("Cannot answer.", ["G1"], "must not cite"),
    ],
)
def test_insufficient_evidence_rejects_declared_or_body_citations(
    answer_text: str,
    citation_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(InvalidCitationError, match=message):
        parse_graph_grounded_answer(
            output(
                {
                    "status": "insufficient_evidence",
                    "answer": answer_text,
                    "citation_ids": citation_ids,
                }
            ),
            (evidence("G1"),),
        )

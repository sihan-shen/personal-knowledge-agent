"""Tests for immutable, provenance-grounded graph answer values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from personal_knowledge_agent.domain.graph import (
    EntityType,
    GraphProvenance,
    RelationType,
)
from personal_knowledge_agent.domain.graph_answer import (
    GraphEvidence,
    GraphGroundedAnswer,
)
from personal_knowledge_agent.domain.models import AnswerStatus


def evidence(citation_id: str = "G1") -> GraphEvidence:
    return GraphEvidence(
        citation_id=citation_id,
        source_entity_type=EntityType.METHOD,
        source_entity_name="RAG",
        relation_type=RelationType.USES,
        target_entity_type=EntityType.TECHNOLOGY,
        target_entity_name="Vector Retrieval",
        provenance=GraphProvenance(
            "/private/rag.md",
            "chunk-private",
            0,
            ("RAG", "Retrieval"),
            "RAG uses vector retrieval.",
        ),
    )


def test_graph_evidence_is_immutable_and_keeps_one_provenance() -> None:
    item = evidence()

    assert item.citation_id == "G1"
    assert item.relation_type is RelationType.USES
    assert item.provenance.evidence_quote == "RAG uses vector retrieval."
    with pytest.raises(FrozenInstanceError):
        item.source_entity_name = "tampered"  # type: ignore[misc]


def test_answered_graph_result_requires_matching_cited_evidence() -> None:
    item = evidence()

    result = GraphGroundedAnswer(
        AnswerStatus.ANSWERED,
        "RAG uses vector retrieval. [G1]",
        ("G1",),
        (item,),
        "fake-model",
    )

    assert result.evidence == (item,)


def test_local_refusal_has_no_citations_evidence_or_model() -> None:
    result = GraphGroundedAnswer(
        AnswerStatus.INSUFFICIENT_EVIDENCE,
        "The knowledge graph does not contain enough relationship evidence.",
        (),
        (),
        None,
    )

    assert result.model_id is None


@pytest.mark.parametrize("field_name", ["source_entity_name", "target_entity_name"])
def test_graph_evidence_rejects_blank_entity_names(field_name: str) -> None:
    values: dict[str, object] = {
        "citation_id": "G1",
        "source_entity_type": EntityType.METHOD,
        "source_entity_name": "RAG",
        "relation_type": RelationType.USES,
        "target_entity_type": EntityType.TECHNOLOGY,
        "target_entity_name": "Vector Retrieval",
        "provenance": GraphProvenance(
            "/private/rag.md",
            "chunk-private",
            0,
            (),
            "RAG uses vector retrieval.",
        ),
    }
    values[field_name] = "  "

    with pytest.raises(ValueError, match=f"{field_name} must not be blank"):
        GraphEvidence(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("citation_id", ["G0", "G13", "G01", "S1", "g1", "G1 "])
def test_graph_evidence_rejects_citation_ids_outside_the_graph_range(
    citation_id: str,
) -> None:
    with pytest.raises(ValueError, match="citation_id must be between G1 and G12"):
        evidence(citation_id)


def test_graph_answer_rejects_blank_answer() -> None:
    with pytest.raises(ValueError, match="answer must not be blank"):
        GraphGroundedAnswer(AnswerStatus.INSUFFICIENT_EVIDENCE, " ", (), (), None)


def test_answered_graph_result_rejects_duplicate_citations() -> None:
    item = evidence()

    with pytest.raises(ValueError, match="citation_ids must be unique"):
        GraphGroundedAnswer(
            AnswerStatus.ANSWERED,
            "RAG uses vector retrieval. [G1]",
            ("G1", "G1"),
            (item, item),
            "fake-model",
        )


def test_answered_graph_result_rejects_mismatched_citations() -> None:
    with pytest.raises(ValueError, match="evidence must match citation_ids in order"):
        GraphGroundedAnswer(
            AnswerStatus.ANSWERED,
            "RAG uses vector retrieval. [G1]",
            ("G2",),
            (evidence(),),
            "fake-model",
        )


def test_answered_graph_result_requires_model_id() -> None:
    with pytest.raises(ValueError, match="answered graph result requires model_id"):
        GraphGroundedAnswer(
            AnswerStatus.ANSWERED,
            "RAG uses vector retrieval. [G1]",
            ("G1",),
            (evidence(),),
            None,
        )


def test_insufficient_graph_result_rejects_evidence() -> None:
    with pytest.raises(
        ValueError,
        match="insufficient graph result must not contain evidence",
    ):
        GraphGroundedAnswer(
            AnswerStatus.INSUFFICIENT_EVIDENCE,
            "The knowledge graph does not contain enough relationship evidence.",
            ("G1",),
            (evidence(),),
            None,
        )


def test_graph_answer_rejects_blank_model_id() -> None:
    with pytest.raises(ValueError, match="model_id must not be blank"):
        GraphGroundedAnswer(
            AnswerStatus.ANSWERED,
            "RAG uses vector retrieval. [G1]",
            ("G1",),
            (evidence(),),
            "  ",
        )

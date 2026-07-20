"""Parse strict structured answers grounded in graph relation evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from personal_knowledge_agent.domain.errors import (
    InvalidAnswerSchemaError,
    InvalidCitationError,
)
from personal_knowledge_agent.domain.graph_answer import (
    GraphEvidence,
    GraphGroundedAnswer,
)
from personal_knowledge_agent.domain.models import AnswerStatus, ModelOutput

_REQUIRED_FIELDS = {"status", "answer", "citation_ids"}
_CITATION_LIKE_PATTERN = re.compile(r"\[((?:G|S|M)[^\]]*)\]")
_VALID_GRAPH_CITATION_PATTERN = re.compile(r"G\d+")
_INVALID_GRAPH_CITATION_MESSAGE = "Graph answers may cite only valid G evidence"


def parse_graph_grounded_answer(
    output: ModelOutput,
    evidence: Sequence[GraphEvidence],
) -> GraphGroundedAnswer:
    """Require exact graph-answer JSON and citations valid for this request."""
    payload = _decode_payload(output.content)
    if set(payload) != _REQUIRED_FIELDS:
        raise InvalidAnswerSchemaError(
            "Structured answer must contain exactly status, answer, and citation_ids"
        )

    status_value = payload["status"]
    answer = payload["answer"]
    citation_values = payload["citation_ids"]
    if not isinstance(status_value, str):
        raise InvalidAnswerSchemaError("status must be a string")
    if not isinstance(answer, str):
        raise InvalidAnswerSchemaError("answer must be a string")
    if not answer.strip():
        raise InvalidAnswerSchemaError("answer must not be blank")
    if not isinstance(citation_values, list):
        raise InvalidAnswerSchemaError("citation_ids must be a list")
    if not all(isinstance(item, str) for item in citation_values):
        raise InvalidAnswerSchemaError("citation_ids must contain strings")

    try:
        status = AnswerStatus(status_value)
    except ValueError as error:
        raise InvalidAnswerSchemaError(
            f"Unsupported answer status: {status_value}"
        ) from error

    citation_ids = tuple(citation_values)
    body_labels = _CITATION_LIKE_PATTERN.findall(answer)
    if any(
        _VALID_GRAPH_CITATION_PATTERN.fullmatch(item) is None for item in body_labels
    ):
        raise InvalidCitationError(_INVALID_GRAPH_CITATION_MESSAGE)
    body_graph_citations = _ordered_unique(body_labels)
    if status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        if citation_ids or body_graph_citations:
            raise InvalidCitationError(
                "An insufficient-evidence graph response must not cite evidence"
            )
        return GraphGroundedAnswer(status, answer, (), (), output.model_id)

    selected_evidence = _validate_answered_citations(
        citation_ids,
        body_graph_citations,
        evidence,
    )
    return GraphGroundedAnswer(
        status,
        answer,
        citation_ids,
        selected_evidence,
        output.model_id,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise InvalidAnswerSchemaError(
                "Structured answer must not contain a duplicate field"
            )
        payload[key] = value
    return payload


def _decode_payload(content: str) -> dict[str, object]:
    try:
        payload = json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise InvalidAnswerSchemaError(
            "Model output must be valid JSON without a code fence"
        ) from error
    if not isinstance(payload, dict):
        raise InvalidAnswerSchemaError("Structured answer must be a JSON object")
    if not all(isinstance(key, str) for key in payload):
        raise InvalidAnswerSchemaError("Structured answer keys must be strings")
    return payload


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _validate_answered_citations(
    citation_ids: tuple[str, ...],
    body_graph_citations: tuple[str, ...],
    evidence: Sequence[GraphEvidence],
) -> tuple[GraphEvidence, ...]:
    if not citation_ids:
        raise InvalidCitationError("An answered graph response needs a citation")
    if len(set(citation_ids)) != len(citation_ids):
        raise InvalidCitationError("citation_ids must be unique")
    if any(
        _VALID_GRAPH_CITATION_PATTERN.fullmatch(item) is None for item in citation_ids
    ):
        raise InvalidCitationError(_INVALID_GRAPH_CITATION_MESSAGE)
    if body_graph_citations != citation_ids:
        raise InvalidCitationError(
            "Answer body citations must match citation_ids in first-use order"
        )

    evidence_by_id = {item.citation_id: item for item in evidence}
    for citation_id in citation_ids:
        if citation_id not in evidence_by_id:
            raise InvalidCitationError(
                f"Answer contains unknown citation: {citation_id}"
            )
    return tuple(evidence_by_id[citation_id] for citation_id in citation_ids)

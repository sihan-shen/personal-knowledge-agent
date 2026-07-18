"""Parse strict structured answers with document and memory citations."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from personal_knowledge_agent.domain.errors import (
    InvalidAnswerSchemaError,
    InvalidCitationError,
)
from personal_knowledge_agent.domain.memory import MemoryEvidence
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    Evidence,
    GroundedAnswer,
    ModelOutput,
)

_REQUIRED_FIELDS = {"status", "answer", "citation_ids"}
_CITATION_PATTERN = re.compile(r"\[((?:S|M)[1-9]\d*)\]")


def parse_contextual_answer(
    output: ModelOutput,
    evidence: Sequence[Evidence],
    memories: Sequence[MemoryEvidence],
) -> GroundedAnswer:
    """Validate exact JSON and citations against both request-local contexts."""
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
    body_citations = tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
    if status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        if citation_ids or body_citations:
            raise InvalidCitationError(
                "An insufficient-evidence answer must not cite context"
            )
    else:
        _validate_answered_citations(
            citation_ids,
            body_citations,
            evidence,
            memories,
        )
    return GroundedAnswer(
        status,
        answer,
        citation_ids,
        tuple(evidence),
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


def _validate_answered_citations(
    citation_ids: tuple[str, ...],
    body_citations: tuple[str, ...],
    evidence: Sequence[Evidence],
    memories: Sequence[MemoryEvidence],
) -> None:
    if not citation_ids:
        raise InvalidCitationError("An answered response needs at least one citation")
    if len(set(citation_ids)) != len(citation_ids):
        raise InvalidCitationError("citation_ids must be unique")
    if body_citations != citation_ids:
        raise InvalidCitationError(
            "Answer body citations must match citation_ids in first-use order"
        )
    known_ids = {
        *(item.citation_id for item in evidence),
        *(item.citation_id for item in memories),
    }
    unknown = [item for item in citation_ids if item not in known_ids]
    if unknown:
        raise InvalidCitationError(f"Answer contains unknown citation: {unknown[0]}")

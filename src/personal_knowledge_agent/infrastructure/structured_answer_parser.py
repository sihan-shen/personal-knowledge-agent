"""Parse and validate evidence-grounded structured model output."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from personal_knowledge_agent.domain.errors import (
    InvalidAnswerSchemaError,
    InvalidCitationError,
)
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    Evidence,
    GroundedAnswer,
    ModelOutput,
)

_REQUIRED_FIELDS = {"status", "answer", "citation_ids"}
_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def parse_grounded_answer(
    output: ModelOutput,
    evidence: Sequence[Evidence],
) -> GroundedAnswer:
    """Require the exact answer schema and citations valid for this request."""
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
    body_citations = _ordered_unique(_CITATION_PATTERN.findall(answer))
    if status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        if citation_ids or body_citations:
            raise InvalidCitationError(
                "An insufficient-evidence answer must not cite evidence"
            )
    else:
        _validate_answered_citations(citation_ids, body_citations, evidence)

    return GroundedAnswer(
        status=status,
        answer=answer,
        citation_ids=citation_ids,
        evidence=tuple(evidence),
        model_id=output.model_id,
    )


def _decode_payload(content: str) -> dict[str, object]:
    try:
        payload = json.loads(content)
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
    body_citations: tuple[str, ...],
    evidence: Sequence[Evidence],
) -> None:
    if not citation_ids:
        raise InvalidCitationError("An answered response needs at least one citation")
    if len(set(citation_ids)) != len(citation_ids):
        raise InvalidCitationError("citation_ids must be unique")
    if body_citations != citation_ids:
        raise InvalidCitationError(
            "Answer body citations must match citation_ids in first-use order"
        )

    known_ids = {item.citation_id for item in evidence}
    unknown_ids = [item for item in citation_ids if item not in known_ids]
    if unknown_ids:
        raise InvalidCitationError(
            f"Answer contains unknown citation: {unknown_ids[0]}"
        )

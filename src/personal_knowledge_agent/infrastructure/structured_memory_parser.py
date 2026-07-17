"""Strict parser for provider-neutral structured memory output."""

from __future__ import annotations

import json
from typing import Any

from personal_knowledge_agent.domain.errors import InvalidMemorySchemaError
from personal_knowledge_agent.domain.memory import (
    MAX_MEMORY_CANDIDATES,
    MemoryProposal,
    MemoryType,
)
from personal_knowledge_agent.domain.models import ModelOutput

_ROOT_FIELDS = {"candidates"}
_CANDIDATE_FIELDS = {"type", "key", "content"}


def _build_unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise InvalidMemorySchemaError(
                "Memory extractor output must not contain a duplicate field"
            )
        payload[key] = value
    return payload


def parse_memory_proposals(output: ModelOutput) -> tuple[MemoryProposal, ...]:
    """Validate the exact memory JSON contract without repairing output."""
    try:
        payload: Any = json.loads(
            output.content,
            object_pairs_hook=_build_unique_object,
        )
    except json.JSONDecodeError as error:
        raise InvalidMemorySchemaError(
            "Memory extractor output must be valid JSON without Markdown fences"
        ) from error

    if type(payload) is not dict:
        raise InvalidMemorySchemaError("Memory extractor output must be a JSON object")
    if set(payload) != _ROOT_FIELDS:
        raise InvalidMemorySchemaError(
            "Memory extractor output must contain exactly candidates"
        )

    candidates = payload["candidates"]
    if type(candidates) is not list:
        raise InvalidMemorySchemaError("Memory candidates must be a JSON list")
    if len(candidates) > MAX_MEMORY_CANDIDATES:
        raise InvalidMemorySchemaError(
            "Memory extractor output must contain at most "
            f"{MAX_MEMORY_CANDIDATES} candidates"
        )

    proposals: list[MemoryProposal] = []
    for candidate in candidates:
        if type(candidate) is not dict:
            raise InvalidMemorySchemaError(
                "Each memory candidate must be a JSON object"
            )
        if set(candidate) != _CANDIDATE_FIELDS:
            raise InvalidMemorySchemaError(
                "Each memory candidate must contain exactly type, key, and content"
            )

        raw_type = candidate["type"]
        raw_key = candidate["key"]
        raw_content = candidate["content"]
        if type(raw_type) is not str:
            raise InvalidMemorySchemaError("Memory candidate type must be a string")
        if type(raw_key) is not str:
            raise InvalidMemorySchemaError("Memory candidate key must be a string")
        if type(raw_content) is not str:
            raise InvalidMemorySchemaError("Memory candidate content must be a string")

        try:
            memory_type = MemoryType(raw_type)
            proposal = MemoryProposal(memory_type, raw_key, raw_content)
        except ValueError as error:
            raise InvalidMemorySchemaError(
                "Memory candidate has an invalid type, key, or content"
            ) from error
        proposals.append(proposal)

    return tuple(proposals)

"""Strict parser for provider-neutral agent route output."""

from __future__ import annotations

import json
from typing import Any

from personal_knowledge_agent.domain.errors import InvalidRouteSchemaError
from personal_knowledge_agent.domain.models import (
    DirectIntent,
    ModelOutput,
    RouteAction,
    RouteDecision,
    RouteSource,
)

_EXPECTED_FIELDS = {"action", "direct_intent"}


def _build_unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise InvalidRouteSchemaError(
                "Router output must not contain a duplicate field"
            )
        payload[key] = value
    return payload


def parse_route_decision(output: ModelOutput) -> RouteDecision:
    """Validate the exact route JSON contract without repairing model output."""
    try:
        payload: Any = json.loads(
            output.content,
            object_pairs_hook=_build_unique_object,
        )
    except json.JSONDecodeError as error:
        raise InvalidRouteSchemaError(
            "Router output must be valid JSON without Markdown fences"
        ) from error

    if type(payload) is not dict:
        raise InvalidRouteSchemaError("Router output must be a JSON object")
    if set(payload) != _EXPECTED_FIELDS:
        raise InvalidRouteSchemaError(
            "Router output must contain exactly action and direct_intent"
        )

    raw_action = payload["action"]
    raw_intent = payload["direct_intent"]
    if type(raw_action) is not str:
        raise InvalidRouteSchemaError("Router action must be a string")
    try:
        action = RouteAction(raw_action)
    except ValueError as error:
        raise InvalidRouteSchemaError(
            "Router output has an unknown route action"
        ) from error

    if raw_intent is not None and type(raw_intent) is not str:
        raise InvalidRouteSchemaError("Router direct_intent must be a string or null")

    intent: DirectIntent | None = None
    if isinstance(raw_intent, str):
        try:
            intent = DirectIntent(raw_intent)
        except ValueError as error:
            raise InvalidRouteSchemaError(
                "Router output has an unknown direct intent"
            ) from error

    if action is RouteAction.DIRECT and intent is None:
        raise InvalidRouteSchemaError("A direct route requires a direct intent")
    if action is RouteAction.RETRIEVE and intent is not None:
        raise InvalidRouteSchemaError("A retrieve route must not include an intent")

    return RouteDecision(action, intent, RouteSource.LLM)

"""Tests for strict structured route parsing."""

import pytest

from personal_knowledge_agent.domain.errors import InvalidRouteSchemaError
from personal_knowledge_agent.domain.models import (
    DirectIntent,
    ModelOutput,
    RouteAction,
    RouteSource,
)
from personal_knowledge_agent.infrastructure.structured_route_parser import (
    parse_route_decision,
)


def output(content: str) -> ModelOutput:
    return ModelOutput("router-model", content, "stop")


def test_parses_retrieve_decision() -> None:
    decision = parse_route_decision(
        output('{"action":"retrieve","direct_intent":null}')
    )

    assert decision.action is RouteAction.RETRIEVE
    assert decision.direct_intent is None
    assert decision.source is RouteSource.LLM
    assert decision.fallback_reason is None


@pytest.mark.parametrize(
    ("raw_intent", "expected"),
    [
        ("greeting", DirectIntent.GREETING),
        ("capabilities", DirectIntent.CAPABILITIES),
        ("thanks", DirectIntent.THANKS),
    ],
)
def test_parses_each_direct_intent(
    raw_intent: str,
    expected: DirectIntent,
) -> None:
    decision = parse_route_decision(
        output(f'{{"action":"direct","direct_intent":"{raw_intent}"}}')
    )

    assert decision.action is RouteAction.DIRECT
    assert decision.direct_intent is expected
    assert decision.source is RouteSource.LLM


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not JSON", "valid JSON"),
        (
            '```json\n{"action":"retrieve","direct_intent":null}\n```',
            "valid JSON",
        ),
        ('["retrieve", null]', "JSON object"),
        ('{"action":"retrieve"}', "exactly"),
        (
            '{"action":"retrieve","direct_intent":null,"reason":"safe"}',
            "exactly",
        ),
        ('{"action":1,"direct_intent":null}', "action must be a string"),
        ('{"action":"search","direct_intent":null}', "unknown route action"),
        ('{"action":"","direct_intent":null}', "unknown route action"),
        (
            '{"action":"direct","direct_intent":1}',
            "direct_intent must be a string or null",
        ),
        (
            '{"action":"direct","direct_intent":"smalltalk"}',
            "unknown direct intent",
        ),
        (
            '{"action":"direct","direct_intent":""}',
            "unknown direct intent",
        ),
        (
            '{"action":"direct","direct_intent":null}',
            "direct route requires",
        ),
        (
            '{"action":"retrieve","direct_intent":"greeting"}',
            "retrieve route",
        ),
    ],
)
def test_rejects_invalid_route_schema(content: str, message: str) -> None:
    with pytest.raises(InvalidRouteSchemaError, match=message):
        parse_route_decision(output(content))


def test_route_error_does_not_echo_provider_output() -> None:
    secret = "provider-body-secret-123"

    with pytest.raises(InvalidRouteSchemaError) as captured:
        parse_route_decision(output(f"not JSON {secret}"))

    assert secret not in str(captured.value)


@pytest.mark.parametrize(
    "content",
    [
        ('{"action":"retrieve","action":"direct","direct_intent":"greeting"}'),
        ('{"action":"direct","direct_intent":"greeting","direct_intent":"thanks"}'),
    ],
)
def test_rejects_duplicate_json_fields(content: str) -> None:
    with pytest.raises(InvalidRouteSchemaError, match="duplicate field"):
        parse_route_decision(output(content))

"""Strict JSON parsers for isolated LLM Judge responses."""

from __future__ import annotations

import json

from personal_knowledge_agent.domain.errors import InvalidJudgeSchemaError
from personal_knowledge_agent.domain.evaluation import (
    JudgeMetric,
    JudgeMetricScore,
    ReferenceQualityScores,
)
from personal_knowledge_agent.domain.models import ModelOutput


def parse_faithfulness_judge_output(output: ModelOutput) -> JudgeMetricScore:
    """Parse exactly one faithfulness metric object."""
    payload = _decode_object(output.content)
    _require_exact_keys(payload, {JudgeMetric.FAITHFULNESS.value})
    return _parse_metric(
        payload[JudgeMetric.FAITHFULNESS.value], JudgeMetric.FAITHFULNESS
    )


def parse_reference_quality_judge_output(
    output: ModelOutput,
) -> ReferenceQualityScores:
    """Parse exactly relevance and completeness metric objects."""
    payload = _decode_object(output.content)
    _require_exact_keys(
        payload,
        {JudgeMetric.RELEVANCE.value, JudgeMetric.COMPLETENESS.value},
    )
    return ReferenceQualityScores(
        _parse_metric(payload[JudgeMetric.RELEVANCE.value], JudgeMetric.RELEVANCE),
        _parse_metric(
            payload[JudgeMetric.COMPLETENESS.value], JudgeMetric.COMPLETENESS
        ),
    )


def _decode_object(content: str) -> dict[str, object]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise InvalidJudgeSchemaError("Judge output must be valid JSON") from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise InvalidJudgeSchemaError("Judge output must be a JSON object")
    return payload


def _require_exact_keys(payload: dict[str, object], expected: set[str]) -> None:
    if set(payload) != expected:
        raise InvalidJudgeSchemaError("Judge output contains unexpected metric fields")


def _parse_metric(value: object, metric: JudgeMetric) -> JudgeMetricScore:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise InvalidJudgeSchemaError(f"{metric.value} must be an object")
    if set(value) != {"score", "reason"}:
        raise InvalidJudgeSchemaError(
            f"{metric.value} must contain exactly score and reason"
        )
    score = value["score"]
    reason = value["reason"]
    if isinstance(score, bool) or not isinstance(score, int):
        raise InvalidJudgeSchemaError(f"{metric.value} score must be an integer")
    if not isinstance(reason, str):
        raise InvalidJudgeSchemaError(f"{metric.value} reason must be a string")
    try:
        return JudgeMetricScore(metric, score, reason)
    except ValueError as error:
        raise InvalidJudgeSchemaError(str(error)) from error

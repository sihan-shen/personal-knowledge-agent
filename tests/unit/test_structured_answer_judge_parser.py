"""Tests for strict LLM Judge structured output parsing."""

import pytest

from personal_knowledge_agent.domain.errors import InvalidJudgeSchemaError
from personal_knowledge_agent.domain.evaluation import JudgeMetric
from personal_knowledge_agent.domain.models import ModelOutput
from personal_knowledge_agent.infrastructure.structured_answer_judge_parser import (
    parse_faithfulness_judge_output,
    parse_reference_quality_judge_output,
)


def output(content: str) -> ModelOutput:
    return ModelOutput("fake-judge", content, "stop")


def test_parses_valid_faithfulness_score() -> None:
    result = parse_faithfulness_judge_output(
        output('{"faithfulness":{"score":2,"reason":"Supported."}}')
    )

    assert result.metric is JudgeMetric.FAITHFULNESS
    assert result.score == 2
    assert result.reason == "Supported."


@pytest.mark.parametrize(
    "content",
    (
        "[]",
        "{}",
        '{"faithfulness":{"score":true,"reason":"Supported."}}',
        '{"faithfulness":{"score":3,"reason":"Supported."}}',
        '{"faithfulness":{"score":2}}',
        '{"faithfulness":{"score":2,"reason":"Supported."},"extra":1}',
    ),
)
def test_faithfulness_parser_rejects_invalid_contract(content: str) -> None:
    with pytest.raises(InvalidJudgeSchemaError):
        parse_faithfulness_judge_output(output(content))


def test_parses_valid_reference_quality_scores() -> None:
    result = parse_reference_quality_judge_output(
        output(
            '{"relevance":{"score":2,"reason":"Direct."},'
            '"completeness":{"score":1,"reason":"Partial."}}'
        )
    )

    assert result.relevance.metric is JudgeMetric.RELEVANCE
    assert result.completeness.metric is JudgeMetric.COMPLETENESS


def test_reference_quality_parser_rejects_missing_metric() -> None:
    with pytest.raises(InvalidJudgeSchemaError):
        parse_reference_quality_judge_output(
            output('{"relevance":{"score":2,"reason":"Direct."}}')
        )

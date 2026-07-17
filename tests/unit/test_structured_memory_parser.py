"""Tests for strict structured memory proposal parsing."""

import pytest

from personal_knowledge_agent.domain.errors import InvalidMemorySchemaError
from personal_knowledge_agent.domain.memory import MemoryProposal, MemoryType
from personal_knowledge_agent.domain.models import ModelOutput
from personal_knowledge_agent.infrastructure.structured_memory_parser import (
    parse_memory_proposals,
)


def output(content: str) -> ModelOutput:
    return ModelOutput("memory-extractor", content, "stop")


def test_parses_empty_candidate_list() -> None:
    assert parse_memory_proposals(output('{"candidates":[]}')) == ()


def test_parses_up_to_three_candidates_and_constructs_domain_values() -> None:
    result = parse_memory_proposals(
        output(
            '{"candidates":['
            '{"type":"preference","key":"preference.editor","content":"Uses Neovim"},'
            '{"type":"project","key":"project.rag-agent",'
            '"content":"Build a RAG agent"},'
            '{"type":"learning_progress","key":"learning.python",'
            '"content":"Studying typing"}'
            "]}"
        )
    )

    assert result == (
        MemoryProposal(MemoryType.PREFERENCE, "preference.editor", "Uses Neovim"),
        MemoryProposal(MemoryType.PROJECT, "project.rag-agent", "Build a RAG agent"),
        MemoryProposal(
            MemoryType.LEARNING_PROGRESS,
            "learning.python",
            "Studying typing",
        ),
    )


def test_parses_goal_and_uses_domain_content_normalization() -> None:
    result = parse_memory_proposals(
        output(
            '{"candidates":[{"type":"goal","key":"goal.ai",'
            '"content":"  Build   AI systems  "}]}'
        )
    )

    assert result == (MemoryProposal(MemoryType.GOAL, "goal.ai", "Build AI systems"),)


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        '```json\n{"candidates":[]}\n```',
        "[]",
        '"candidates"',
        "null",
        "true",
        "1",
        "{}",
        '{"candidate":[]}',
        '{"candidates":[],"reason":"none"}',
        '{"candidates":null}',
        '{"candidates":{}}',
        '{"candidates":"none"}',
        '{"candidates":[{}]}',
        '{"candidates":[{"type":"goal","key":"goal.ai"}]}',
        '{"candidates":[{"type":"goal","content":"Build AI"}]}',
        '{"candidates":[{"key":"goal.ai","content":"Build AI"}]}',
        (
            '{"candidates":[{"type":"goal","key":"goal.ai",'
            '"content":"Build AI","confidence":1}]}'
        ),
        '{"candidates":[{"type":1,"key":"goal.ai","content":"Build AI"}]}',
        '{"candidates":[{"type":true,"key":"goal.ai","content":"Build AI"}]}',
        '{"candidates":[{"type":"goal","key":1,"content":"Build AI"}]}',
        '{"candidates":[{"type":"goal","key":true,"content":"Build AI"}]}',
        '{"candidates":[{"type":"goal","key":"goal.ai","content":1}]}',
        '{"candidates":[{"type":"goal","key":"goal.ai","content":true}]}',
        '{"candidates":[{"type":"fact","key":"goal.ai","content":"Build AI"}]}',
        '{"candidates":[{"type":"","key":"goal.ai","content":"Build AI"}]}',
        '{"candidates":[{"type":"goal","key":"Goal.ai","content":"Build AI"}]}',
        '{"candidates":[{"type":"goal","key":"1goal.ai","content":"Build AI"}]}',
        '{"candidates":[{"type":"goal","key":"goal/ai","content":"Build AI"}]}',
        (
            '{"candidates":[{"type":"goal","key":"a'
            + "x" * 64
            + '","content":"Build AI"}]}'
        ),
        '{"candidates":[{"type":"goal","key":"goal.ai","content":""}]}',
        '{"candidates":[{"type":"goal","key":"goal.ai","content":"  "}]}',
        (
            '{"candidates":[{"type":"goal","key":"goal.ai","content":"'
            + "x" * 501
            + '"}]}'
        ),
        (
            '{"candidates":['
            '{"type":"goal","key":"goal.one","content":"One"},'
            '{"type":"goal","key":"goal.two","content":"Two"},'
            '{"type":"goal","key":"goal.three","content":"Three"},'
            '{"type":"goal","key":"goal.four","content":"Four"}'
            "]}"
        ),
    ],
)
def test_rejects_invalid_memory_schema(content: str) -> None:
    with pytest.raises(InvalidMemorySchemaError):
        parse_memory_proposals(output(content))


@pytest.mark.parametrize(
    "content",
    [
        '{"candidates":[],"candidates":[]}',
        (
            '{"candidates":[{"type":"goal","type":"project",'
            '"key":"goal.ai","content":"Build AI"}]}'
        ),
        (
            '{"candidates":[{"type":"goal","key":"goal.ai",'
            '"content":{"value":"Build AI","value":"Leak"}}]}'
        ),
    ],
)
def test_rejects_duplicate_json_keys_at_every_object_level(content: str) -> None:
    with pytest.raises(InvalidMemorySchemaError, match="duplicate"):
        parse_memory_proposals(output(content))


def test_schema_error_does_not_echo_provider_output() -> None:
    secret = "provider-body-secret-8472"

    with pytest.raises(InvalidMemorySchemaError) as captured:
        parse_memory_proposals(output(f"not JSON {secret}"))

    assert secret not in str(captured.value)

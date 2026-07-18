"""Tests for conservative local memory relevance screening."""

from datetime import UTC, datetime

import pytest

from personal_knowledge_agent.domain.errors import InvalidQueryError
from personal_knowledge_agent.domain.memory import (
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
    memory_content_hash,
)
from personal_knowledge_agent.infrastructure.memory_relevance import (
    MIN_MEMORY_COSINE_SIMILARITY,
    ConservativeMemoryRelevancePolicy,
)

_NOW = datetime(2026, 7, 18, tzinfo=UTC)


def make_result(
    memory_key: str,
    content: str,
    similarity: float,
    *,
    memory_type: MemoryType = MemoryType.PREFERENCE,
) -> MemorySearchResult:
    record = MemoryRecord(
        "mem_00000000000000000000000000000001",
        memory_type,
        memory_key,
        content,
        memory_content_hash(content),
        MemoryStatus.ACTIVE,
        None,
        None,
        _NOW,
        _NOW,
    )
    return MemorySearchResult(record, 0.02, similarity)


@pytest.mark.parametrize(
    ("query", "memory_key", "content", "similarity", "expected"),
    [
        ("What is my learning goal?", "learning_goal", "Master agents", 0.70, True),
        ("How should you answer me?", "answer-style", "Be concise", 0.70, True),
        ("what is my", "unrelated", "Prefer dark mode", 0.8399, False),
        ("完全无关的问题", "answer_style", "简洁回答", 0.84, True),
        ("完全无关的问题", "answer_style", "简洁回答", 0.8399, False),
        ("rag agent", "learning.rag-agent", "Making progress", 0.10, True),
    ],
)
def test_conservative_relevance_gate(
    query: str,
    memory_key: str,
    content: str,
    similarity: float,
    expected: bool,
) -> None:
    result = make_result(memory_key, content, similarity)

    assert ConservativeMemoryRelevancePolicy().allows(query, result) is expected


def test_relevance_threshold_is_fixed_and_inclusive() -> None:
    assert MIN_MEMORY_COSINE_SIMILARITY == 0.84
    assert ConservativeMemoryRelevancePolicy().allows(
        "unrelated query",
        make_result("other", "different content", 0.84),
    )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_relevance_policy_rejects_blank_query(query: str) -> None:
    with pytest.raises(InvalidQueryError, match="query"):
        ConservativeMemoryRelevancePolicy().allows(
            query,
            make_result("answer_style", "Be concise", 0.9),
        )

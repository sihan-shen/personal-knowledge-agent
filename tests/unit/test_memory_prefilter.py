"""Tests for the local first-person memory extraction prefilter."""

import pytest

from personal_knowledge_agent.infrastructure.memory_prefilter import (
    should_extract_memory,
)


@pytest.mark.parametrize(
    "message",
    [
        "My goal is to become an AI Engineer.",
        "I prefer concise explanations.",
        "I am working on a Personal Knowledge Agent.",
        "I have completed the RAG retrieval module.",
        "我的目标是完成 AI Engineer 项目。",
        "我更喜欢先解释原理。",
        "我正在开发个人知识 Agent。",
        "我已经学完向量检索。",
    ],
)
def test_prefilter_accepts_durable_first_person_signals(message: str) -> None:
    assert should_extract_memory(message) is True


@pytest.mark.parametrize(
    "message",
    ["What is RAG?", "Explain my notes", "It is raining", "我应该学什么？", ""],
)
def test_prefilter_rejects_ordinary_or_nonstatement_text(message: str) -> None:
    assert should_extract_memory(message) is False


@pytest.mark.parametrize(
    "message",
    ["What is my goal?", "What is my goal", "我的目标是什么？", "我的目标是什么"],
)
def test_prefilter_rejects_questions_containing_durable_phrases(
    message: str,
) -> None:
    assert should_extract_memory(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "ＭＹ   ＧＯＡＬ is to learn RAG.",
        "I\n\tprefer concise answers.",
        "我正在\n做混合检索。",
    ],
)
def test_prefilter_normalizes_nfkc_and_whitespace(message: str) -> None:
    assert should_extract_memory(message) is True


@pytest.mark.parametrize("message", ["I", "my", "我", "我的"])
def test_prefilter_does_not_match_bare_pronouns(message: str) -> None:
    assert should_extract_memory(message) is False

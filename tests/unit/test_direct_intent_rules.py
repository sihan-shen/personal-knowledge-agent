"""Tests for deterministic safe direct-intent matching."""

import pytest

from personal_knowledge_agent.domain.models import DirectIntent
from personal_knowledge_agent.infrastructure.direct_intent_rules import (
    match_direct_intent,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("hi", DirectIntent.GREETING),
        ("  Hello!  ", DirectIntent.GREETING),
        ("HEY", DirectIntent.GREETING),
        ("你好。", DirectIntent.GREETING),
        ("您好！", DirectIntent.GREETING),
        ("thanks", DirectIntent.THANKS),
        ("Thank   you!", DirectIntent.THANKS),
        ("谢谢～", DirectIntent.THANKS),
        ("多谢", DirectIntent.THANKS),
        ("what can you do?", DirectIntent.CAPABILITIES),
        ("What   can   you   do", DirectIntent.CAPABILITIES),
        ("你能做什么？", DirectIntent.CAPABILITIES),
        ("你可以做什么", DirectIntent.CAPABILITIES),
    ],
)
def test_matches_safe_whole_message_intents(
    question: str,
    expected: DirectIntent,
) -> None:
    assert match_direct_intent(question) is expected


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        "你好，请解释 RAG",
        "hello what is RAG?",
        "thanks, now search my notes",
        "what can you do with RAG?",
        "What is RAG?",
        "请查找我的笔记",
        "hello there",
    ],
)
def test_does_not_match_factual_mixed_or_unknown_requests(question: str) -> None:
    assert match_direct_intent(question) is None

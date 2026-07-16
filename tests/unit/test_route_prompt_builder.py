"""Tests for conservative route-classification prompts."""

import pytest

from personal_knowledge_agent.domain.errors import InvalidQueryError
from personal_knowledge_agent.domain.models import MessageRole
from personal_knowledge_agent.infrastructure.route_prompt_builder import (
    build_route_messages,
)


def test_route_prompt_has_two_messages_and_exact_json_contract() -> None:
    system, user = build_route_messages("How does RAG work?")

    assert system.role is MessageRole.SYSTEM
    assert user.role is MessageRole.USER
    assert "action" in system.content
    assert "direct_intent" in system.content
    assert "exactly" in system.content.lower()
    assert "direct" in system.content
    assert "retrieve" in system.content
    assert "greeting" in system.content
    assert "capabilities" in system.content
    assert "thanks" in system.content
    assert "null" in system.content
    assert "without Markdown" in system.content
    assert user.content == "QUESTION:\nHow does RAG work?"


def test_route_prompt_is_conservative_for_factual_mixed_and_uncertain_requests() -> (
    None
):
    system, _ = build_route_messages("question")

    assert "factual" in system.content
    assert "learning" in system.content
    assert "mixed" in system.content
    assert "ambiguous" in system.content
    assert "uncertain" in system.content
    assert "must use retrieve" in system.content


def test_route_prompt_sends_only_the_question_to_the_user_message() -> None:
    question = "Find sentinel-question in my notes"
    system, user = build_route_messages(question)

    assert question in user.content
    assert "sentinel-source.md" not in system.content + user.content
    assert "sentinel-evidence" not in system.content + user.content
    assert "PKA_LLM_API_KEY" not in system.content + user.content


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_route_prompt_rejects_blank_question(question: str) -> None:
    with pytest.raises(InvalidQueryError, match="question"):
        build_route_messages(question)

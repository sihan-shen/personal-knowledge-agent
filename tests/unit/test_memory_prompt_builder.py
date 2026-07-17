"""Tests for privacy-minimized structured memory extraction prompts."""

import pytest

from personal_knowledge_agent.domain.errors import InvalidMemoryError
from personal_knowledge_agent.domain.memory import MAX_MEMORY_EXTRACTION_INPUT_LENGTH
from personal_knowledge_agent.domain.models import MessageRole
from personal_knowledge_agent.infrastructure.memory_prompt_builder import (
    build_memory_extraction_messages,
)


def test_memory_prompt_has_two_messages_and_exact_contract() -> None:
    messages = build_memory_extraction_messages("My goal is to build a RAG agent")

    assert len(messages) == 2
    system, user = messages
    assert system.role is MessageRole.SYSTEM
    assert user.role is MessageRole.USER
    assert (
        '{"candidates":[{"type":"goal","key":"goal.ai_engineer_internship",'
        '"content":"The user aims to complete an AI Engineer internship project."}]}'
        in system.content
    )
    assert "preference" in system.content
    assert "goal" in system.content
    assert "project" in system.content
    assert "learning_progress" in system.content
    assert r"[a-z][a-z0-9_.-]{0,63}" in system.content
    assert "at most 3" in system.content
    assert "empty list is valid" in system.content
    assert "preserve the user's language" in system.content.lower()
    assert "durable" in system.content.lower()
    assert "temporary" in system.content.lower()
    assert "Output JSON without Markdown fences" in system.content
    assert "extra fields" in system.content
    assert "commentary" in system.content
    assert user.content == "CURRENT USER MESSAGE:\nMy goal is to build a RAG agent"


def test_memory_prompt_forbids_sensitive_and_secret_categories() -> None:
    system, _ = build_memory_extraction_messages("Remember my preferences")

    policy = system.content.lower()
    for excluded in (
        "password",
        "api key",
        "access token",
        "private key",
        "identity document",
        "government identifier",
        "medical diagnoses",
        "treatment",
        "health record",
        "bank account",
        "payment card",
        "income",
        "debt",
        "financial account",
        "legal case",
        "privileged legal detail",
    ):
        assert excluded in policy
    assert "do not extract" in policy


def test_memory_prompt_sends_only_current_message_without_internal_context() -> None:
    message = "Sentinel current preference"

    system, user = build_memory_extraction_messages(message)

    assert user.content == f"CURRENT USER MESSAGE:\n{message}"
    combined = system.content + user.content
    for forbidden in (
        "sentinel-old-memory",
        "sentinel-evidence",
        "sentinel-source.md",
        "source_hash",
        "content_hash",
        "embedding",
        "PKA_LLM_API_KEY",
    ):
        assert forbidden not in combined


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_memory_prompt_rejects_blank_message(message: str) -> None:
    with pytest.raises(InvalidMemoryError, match="message"):
        build_memory_extraction_messages(message)


def test_memory_prompt_rejects_message_over_limit() -> None:
    message = "x" * (MAX_MEMORY_EXTRACTION_INPUT_LENGTH + 1)

    with pytest.raises(InvalidMemoryError, match="4,000"):
        build_memory_extraction_messages(message)

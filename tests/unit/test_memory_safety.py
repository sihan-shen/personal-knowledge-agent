"""Tests for the deterministic local memory safety policy."""

import pytest

from personal_knowledge_agent.infrastructure.memory_safety import (
    LocalMemorySafetyPolicy,
)


@pytest.mark.parametrize(
    "value",
    [
        "password: correct-horse-battery-staple",
        "My token is sk-abcdefghijklmnop",
        "-----BEGIN RSA PRIVATE KEY-----",
        "My passport renewal is due.",
        "我的身份证需要更新。",
        "My diagnosis changed.",
        "这是我的病历。",
        "My bank account is changing.",
        "I need to pay this credit card.",
        "我正在偿还债务。",
        "The lawsuit is still active.",
        "这个法律案件还在处理。",
    ],
)
def test_safety_policy_rejects_prohibited_raw_messages(value: str) -> None:
    assert LocalMemorySafetyPolicy().allows(value) is False


@pytest.mark.parametrize(
    ("key", "content"),
    [
        ("preference.password", "password=hunter2"),
        ("project.api_token", "sk-abcdefghijklmnop"),
        ("project.private_key", "-----BEGIN PRIVATE KEY-----"),
        ("identity.passport", "Renew it next year"),
        ("identity.身份证", "需要更新"),
        ("health.diagnosis", "Follow up next month"),
        ("health.病历", "下周复查"),
        ("finance.bank_account", "Update it soon"),
        ("finance.credit_card", "Pay it monthly"),
        ("finance.债务", "每月偿还"),
        ("legal.lawsuit", "Review the timeline"),
        ("legal.法律案件", "等待处理"),
    ],
)
def test_safety_policy_rejects_prohibited_keys_or_content(
    key: str,
    content: str,
) -> None:
    assert LocalMemorySafetyPolicy().allows(key, content) is False


@pytest.mark.parametrize(
    ("raw_message", "key", "content"),
    [
        ("password", "project.topic", "opaque content"),
        ("ordinary update", "preference.password", "opaque content"),
        ("ordinary update", "project.topic", "password"),
        ("passwd", "project.topic", "opaque content"),
        ("ordinary update", "credentials.passwd", "opaque content"),
        ("ordinary update", "project.topic", "passwd"),
        ("api_key", "project.topic", "opaque content"),
        ("ordinary update", "credentials.api_key", "opaque content"),
        ("ordinary update", "project.topic", "api_key"),
        ("access_token", "project.topic", "opaque content"),
        ("ordinary update", "credentials.access_token", "opaque content"),
        ("ordinary update", "project.topic", "access_token"),
        ("private_key", "project.topic", "opaque content"),
        ("ordinary update", "project.private_key", "opaque content"),
        ("ordinary update", "project.topic", "private_key"),
    ],
)
def test_safety_policy_rejects_sensitive_terms_in_every_input_position(
    raw_message: str,
    key: str,
    content: str,
) -> None:
    assert LocalMemorySafetyPolicy().allows(raw_message, key, content) is False


@pytest.mark.parametrize(
    "message",
    [
        "My goal is to become an AI Engineer.",
        "I prefer concise explanations.",
        "I am working on a Personal Knowledge Agent.",
        "I have completed the RAG retrieval module.",
    ],
)
def test_safety_policy_allows_normal_learning_examples(message: str) -> None:
    assert LocalMemorySafetyPolicy().allows(message) is True


def test_safety_policy_normalizes_nfkc_and_whitespace() -> None:
    assert LocalMemorySafetyPolicy().allows("ＰＡＳＳＷＯＲＤ\n: secret") is False

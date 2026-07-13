"""Tests for deterministic mixed-language tokenization."""

import pytest

from personal_knowledge_agent.infrastructure.mixed_tokenizer import (
    MixedLanguageTokenizer,
)


@pytest.fixture
def tokenizer() -> MixedLanguageTokenizer:
    return MixedLanguageTokenizer()


def test_tokenizes_lowercase_english_words(
    tokenizer: MixedLanguageTokenizer,
) -> None:
    assert tokenizer.tokenize("Retrieval with RAG") == [
        "retrieval",
        "with",
        "rag",
    ]


def test_preserves_apostrophe_hyphen_and_versions(
    tokenizer: MixedLanguageTokenizer,
) -> None:
    assert tokenizer.tokenize("Agent's retrieval-augmented Python 3.12") == [
        "agent's",
        "retrieval-augmented",
        "python",
        "3.12",
    ]


def test_normalizes_full_width_characters(
    tokenizer: MixedLanguageTokenizer,
) -> None:
    assert tokenizer.tokenize("ＲＡＧ ３.１２") == ["rag", "3.12"]


def test_creates_overlapping_cjk_bigrams(
    tokenizer: MixedLanguageTokenizer,
) -> None:
    assert tokenizer.tokenize("检索增强生成") == [
        "检索",
        "索增",
        "增强",
        "强生",
        "生成",
    ]


def test_keeps_single_cjk_character(tokenizer: MixedLanguageTokenizer) -> None:
    assert tokenizer.tokenize("问") == ["问"]


def test_tokenizes_mixed_text(tokenizer: MixedLanguageTokenizer) -> None:
    assert tokenizer.tokenize("RAG 检索 Python 3.12") == [
        "rag",
        "检索",
        "python",
        "3.12",
    ]


def test_omits_punctuation_and_whitespace(
    tokenizer: MixedLanguageTokenizer,
) -> None:
    assert tokenizer.tokenize("  ，。！？()  ") == []

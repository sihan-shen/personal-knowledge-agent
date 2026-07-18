"""Conservative local relevance policy for provider-bound memory context."""

from __future__ import annotations

import re

from personal_knowledge_agent.domain.errors import InvalidQueryError
from personal_knowledge_agent.domain.memory import MemorySearchResult
from personal_knowledge_agent.infrastructure.mixed_tokenizer import (
    MixedLanguageTokenizer,
)

MIN_MEMORY_COSINE_SIMILARITY = 0.84

_KEY_SEPARATOR_PATTERN = re.compile(r"[._-]+")
_LOW_INFORMATION_TOKENS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "be",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "what",
        "which",
        "who",
        "how",
        "do",
        "does",
        "did",
        "i",
        "my",
        "me",
        "you",
        "your",
        "什么",
        "怎么",
        "如何",
        "我的",
        "你的",
        "是否",
        "可以",
        "需要",
    }
)


class ConservativeMemoryRelevancePolicy:
    """Allow strong semantic matches or meaningful bilingual token overlap."""

    def __init__(self, tokenizer: MixedLanguageTokenizer | None = None) -> None:
        self._tokenizer = tokenizer or MixedLanguageTokenizer()

    def allows(self, query: str, result: MemorySearchResult) -> bool:
        """Return whether one ranked memory is relevant enough to disclose."""
        if not query.strip():
            raise InvalidQueryError("Memory relevance query must not be blank")

        normalized_key = _KEY_SEPARATOR_PATTERN.sub(" ", result.memory.memory_key)
        descriptor = " ".join(
            (
                result.memory.memory_type.value,
                normalized_key,
                result.memory.content,
            )
        )
        query_tokens = self._meaningful_tokens(query)
        memory_tokens = self._meaningful_tokens(descriptor)
        return bool(query_tokens & memory_tokens) or (
            result.semantic_similarity >= MIN_MEMORY_COSINE_SIMILARITY
        )

    def _meaningful_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in self._tokenizer.tokenize(text)
            if token not in _LOW_INFORMATION_TOKENS
        }

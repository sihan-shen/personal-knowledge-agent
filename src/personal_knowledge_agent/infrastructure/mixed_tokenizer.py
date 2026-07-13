"""Deterministic lexical tokenization for English and Chinese text."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:['-][a-z]+)*|\d+(?:\.\d+)*|[\u3400-\u4dbf\u4e00-\u9fff]+"
)


def _is_cjk_sequence(value: str) -> bool:
    return all(
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        for character in value
    )


class MixedLanguageTokenizer:
    """Create English-word, numeric, and Chinese-bigram tokens."""

    def tokenize(self, text: str) -> list[str]:
        """Normalize text and return tokens in source order."""
        normalized = unicodedata.normalize("NFKC", text).lower()
        tokens: list[str] = []

        for match in _TOKEN_PATTERN.finditer(normalized):
            value = match.group()
            if not _is_cjk_sequence(value):
                tokens.append(value)
            elif len(value) == 1:
                tokens.append(value)
            else:
                tokens.extend(
                    value[index : index + 2] for index in range(len(value) - 1)
                )

        return tokens

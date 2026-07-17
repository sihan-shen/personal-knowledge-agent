"""Local durable-information prefilter for memory extraction."""

from __future__ import annotations

import re
import unicodedata

_DURABLE_SIGNAL_PATTERNS = (
    re.compile(r"\bmy\s+goal\b", re.IGNORECASE),
    re.compile(r"\bi\s+prefer\b", re.IGNORECASE),
    re.compile(r"\bi\s+am\s+working\s+on\b", re.IGNORECASE),
    re.compile(r"\bi\s+have\s+completed\b", re.IGNORECASE),
    re.compile(r"我的\s*目标", re.IGNORECASE),
    re.compile(r"我\s*更喜欢", re.IGNORECASE),
    re.compile(r"我\s*正在\s*(?:做|开发)", re.IGNORECASE),
    re.compile(r"我\s*已经\s*学完", re.IGNORECASE),
)

_INTERROGATIVE_PATTERNS = (
    re.compile(
        r"^(?:what|why|how|when|where|who|which|do|does|did|is|are|was|were|"
        r"can|could|would|should|will|has|have)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:什么|为什么|怎么|如何|是否|是不是|吗|呢|哪(?:个|些|里|儿)?|谁|多少)"
    ),
)


def should_extract_memory(message: str) -> bool:
    """Return whether a message contains an approved durable first-person signal."""
    normalized = " ".join(unicodedata.normalize("NFKC", message).split())
    if "?" in normalized or any(
        pattern.search(normalized) for pattern in _INTERROGATIVE_PATTERNS
    ):
        return False
    return any(pattern.search(normalized) for pattern in _DURABLE_SIGNAL_PATTERNS)

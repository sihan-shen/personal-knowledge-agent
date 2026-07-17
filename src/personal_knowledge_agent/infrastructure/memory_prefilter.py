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


def should_extract_memory(message: str) -> bool:
    """Return whether a message contains an approved durable first-person signal."""
    normalized = " ".join(unicodedata.normalize("NFKC", message).split())
    return any(pattern.search(normalized) for pattern in _DURABLE_SIGNAL_PATTERNS)

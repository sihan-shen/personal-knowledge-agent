"""Deterministic whole-message rules for safe direct responses."""

from __future__ import annotations

import re

from personal_knowledge_agent.domain.models import DirectIntent

_TRAILING_PUNCTUATION = " .!?。！？,，;；~～"
_INTENTS_BY_MESSAGE = {
    "hi": DirectIntent.GREETING,
    "hello": DirectIntent.GREETING,
    "hey": DirectIntent.GREETING,
    "你好": DirectIntent.GREETING,
    "您好": DirectIntent.GREETING,
    "嗨": DirectIntent.GREETING,
    "thanks": DirectIntent.THANKS,
    "thank you": DirectIntent.THANKS,
    "thx": DirectIntent.THANKS,
    "谢谢": DirectIntent.THANKS,
    "多谢": DirectIntent.THANKS,
    "感谢": DirectIntent.THANKS,
    "what can you do": DirectIntent.CAPABILITIES,
    "你能做什么": DirectIntent.CAPABILITIES,
    "你可以做什么": DirectIntent.CAPABILITIES,
    "你会做什么": DirectIntent.CAPABILITIES,
}


def match_direct_intent(question: str) -> DirectIntent | None:
    """Return a safe direct intent only when the complete message matches."""
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    normalized = normalized.rstrip(_TRAILING_PUNCTUATION)
    return _INTENTS_BY_MESSAGE.get(normalized)

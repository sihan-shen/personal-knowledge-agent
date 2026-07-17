"""Conservative deterministic safety rules for long-term memory."""

from __future__ import annotations

import re
import unicodedata

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bsk-[a-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:password|passwd|api[_ -]?key|access[_ -]?token)\s*[:=]",
        re.IGNORECASE,
    ),
)

_TERM_SEPARATOR_PATTERN = re.compile(r"[._-]+")

_PROHIBITED_TERMS = (
    "password",
    "api key",
    "private key",
    "passport",
    "social security",
    "government id",
    "身份证",
    "护照",
    "diagnosis",
    "medical record",
    "treatment",
    "诊断",
    "病历",
    "治疗",
    "bank account",
    "credit card",
    "income",
    "debt",
    "银行卡",
    "信用卡",
    "收入",
    "债务",
    "lawsuit",
    "privileged legal",
    "诉讼",
    "法律案件",
)


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


class LocalMemorySafetyPolicy:
    """Reject likely prohibited data; conservative false positives are intentional."""

    def allows(self, *values: str) -> bool:
        """Return true only when every supplied value passes the local rules."""
        for value in values:
            normalized = _normalize(value)
            if any(pattern.search(normalized) for pattern in _SECRET_PATTERNS):
                return False
            term_value = _TERM_SEPARATOR_PATTERN.sub(" ", normalized)
            if any(term in term_value for term in _PROHIBITED_TERMS):
                return False
        return True

"""Validated, immutable values for reproducible RAG evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Final

from personal_knowledge_agent.domain.models import AnswerStatus, SearchMode

_LANGUAGES: Final[frozenset[EvaluationLanguage]]


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_bounded_metric(value: float, field_name: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0.0 and 1.0")


class EvaluationLanguage(StrEnum):
    """Languages represented by the bilingual evaluation suite."""

    EN = "en"
    ZH = "zh"


_LANGUAGES = frozenset(EvaluationLanguage)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One question, answer aliases, and its expected logical document."""

    case_id: str
    language: EvaluationLanguage
    question: str
    reference_answers: tuple[str, ...]
    gold_document_id: str

    def __post_init__(self) -> None:
        _require_text(self.case_id, "case_id")
        _require_text(self.question, "question")
        _require_text(self.gold_document_id, "gold_document_id")
        if not self.reference_answers:
            raise ValueError("reference_answers must not be empty")
        for answer in self.reference_answers:
            _require_text(answer, "reference_answers")


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    """One stable logical document identity in an evaluation suite."""

    document_id: str
    resource_name: str

    def __post_init__(self) -> None:
        _require_text(self.document_id, "document_id")
        _require_text(self.resource_name, "resource_name")


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    """Ordered documents and cases that define one reproducible benchmark."""

    suite_id: str
    version: str
    documents: tuple[EvaluationDocument, ...]
    cases: tuple[EvaluationCase, ...]
    attribution: str

    def __post_init__(self) -> None:
        _require_text(self.suite_id, "suite_id")
        _require_text(self.version, "version")
        _require_text(self.attribution, "attribution")
        if not self.documents:
            raise ValueError("documents must not be empty")
        if not self.cases:
            raise ValueError("cases must not be empty")

        document_ids = tuple(document.document_id for document in self.documents)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document IDs must be unique")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case IDs must be unique")
        known_document_ids = set(document_ids)
        if any(case.gold_document_id not in known_document_ids for case in self.cases):
            raise ValueError("every case gold_document_id must be a suite document")


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    """Four normalized scores for one evaluated answer."""

    answer_exact_match: float
    answer_token_f1: float
    citation_precision: float
    citation_recall: float

    def __post_init__(self) -> None:
        _require_bounded_metric(self.answer_exact_match, "answer_exact_match metric")
        _require_bounded_metric(self.answer_token_f1, "answer_token_f1 metric")
        _require_bounded_metric(self.citation_precision, "citation_precision metric")
        _require_bounded_metric(self.citation_recall, "citation_recall metric")


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """The answer outcome and metrics retained for one evaluation case."""

    case: EvaluationCase
    answer_status: AnswerStatus
    model_id: str | None
    cited_document_ids: tuple[str, ...]
    metrics: CaseMetrics

    def __post_init__(self) -> None:
        if self.model_id is not None:
            _require_text(self.model_id, "model_id")
        unique_document_ids = tuple(dict.fromkeys(self.cited_document_ids))
        for document_id in unique_document_ids:
            _require_text(document_id, "cited_document_ids")
        object.__setattr__(self, "cited_document_ids", unique_document_ids)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Macro-average metrics over a positive number of cases."""

    case_count: int
    answer_exact_match: float
    answer_token_f1: float
    citation_precision: float
    citation_recall: float

    def __post_init__(self) -> None:
        if self.case_count < 1:
            raise ValueError("case_count must be positive")
        _require_bounded_metric(self.answer_exact_match, "answer_exact_match metric")
        _require_bounded_metric(self.answer_token_f1, "answer_token_f1 metric")
        _require_bounded_metric(self.citation_precision, "citation_precision metric")
        _require_bounded_metric(self.citation_recall, "citation_recall metric")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """One fully scored evaluation run without provider or filesystem details."""

    suite_id: str
    suite_version: str
    mode: SearchMode
    limit: int
    cases: tuple[CaseEvaluation, ...]
    overall_summary: MetricSummary
    language_summaries: Mapping[EvaluationLanguage, MetricSummary]

    def __post_init__(self) -> None:
        _require_text(self.suite_id, "suite_id")
        _require_text(self.suite_version, "suite_version")
        if not 1 <= self.limit <= 10:
            raise ValueError("limit must be between 1 and 10")
        if set(self.language_summaries) != _LANGUAGES:
            raise ValueError("language summaries must contain exactly EN and ZH")
        object.__setattr__(
            self,
            "language_summaries",
            MappingProxyType(dict(self.language_summaries)),
        )

"""Validated, immutable values for reproducible RAG evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Final

from personal_knowledge_agent.domain.models import AnswerStatus, Evidence, SearchMode


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


_LANGUAGES: Final[frozenset[EvaluationLanguage]] = frozenset(EvaluationLanguage)


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


class JudgeMetric(StrEnum):
    """The three quality dimensions scored by an LLM Judge."""

    RELEVANCE = "relevance"
    FAITHFULNESS = "faithfulness"
    COMPLETENESS = "completeness"


@dataclass(frozen=True, slots=True)
class JudgeMetricScore:
    """One rubric score and short provider-visible explanation."""

    metric: JudgeMetric
    score: int
    reason: str

    def __post_init__(self) -> None:
        if type(self.score) is not int or self.score not in (0, 1, 2):
            raise ValueError("judge score must be 0, 1, or 2")
        _require_text(self.reason, "judge reason")
        if len(self.reason) > 200:
            raise ValueError("judge reason must contain at most 200 code points")


@dataclass(frozen=True, slots=True)
class FaithfulnessJudgeRequest:
    """The evidence-only input for a faithfulness assessment."""

    question: str
    answer: str
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        _require_text(self.question, "question")
        _require_text(self.answer, "answer")
        if not self.evidence:
            raise ValueError("evidence must not be empty")


@dataclass(frozen=True, slots=True)
class ReferenceQualityJudgeRequest:
    """The reference-only input for relevance and completeness assessment."""

    question: str
    answer: str
    reference_answers: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.question, "question")
        _require_text(self.answer, "answer")
        if not self.reference_answers:
            raise ValueError("reference_answers must not be empty")
        for reference_answer in self.reference_answers:
            _require_text(reference_answer, "reference_answers")


@dataclass(frozen=True, slots=True)
class ReferenceQualityScores:
    """The two scores that may read the reference-answer set."""

    relevance: JudgeMetricScore
    completeness: JudgeMetricScore

    def __post_init__(self) -> None:
        if self.relevance.metric is not JudgeMetric.RELEVANCE:
            raise ValueError("relevance score must use the relevance metric")
        if self.completeness.metric is not JudgeMetric.COMPLETENESS:
            raise ValueError("completeness score must use the completeness metric")


@dataclass(frozen=True, slots=True)
class JudgeEvaluation:
    """All three isolated quality scores for one answered case."""

    relevance: JudgeMetricScore
    faithfulness: JudgeMetricScore
    completeness: JudgeMetricScore

    def __post_init__(self) -> None:
        if self.relevance.metric is not JudgeMetric.RELEVANCE:
            raise ValueError("relevance score must use the relevance metric")
        if self.faithfulness.metric is not JudgeMetric.FAITHFULNESS:
            raise ValueError("faithfulness score must use the faithfulness metric")
        if self.completeness.metric is not JudgeMetric.COMPLETENESS:
            raise ValueError("completeness score must use the completeness metric")


@dataclass(frozen=True, slots=True)
class JudgeConfiguration:
    """Non-secret configuration that makes a Judge run inspectable."""

    model_id: str
    faithfulness_prompt_version: str
    reference_quality_prompt_version: str

    def __post_init__(self) -> None:
        _require_text(self.model_id, "model_id")
        _require_text(self.faithfulness_prompt_version, "faithfulness_prompt_version")
        _require_text(
            self.reference_quality_prompt_version,
            "reference_quality_prompt_version",
        )


@dataclass(frozen=True, slots=True)
class JudgeSummary:
    """Macro-average normalized Judge scores, or no scoreable cases."""

    case_count: int
    relevance: float | None
    faithfulness: float | None
    completeness: float | None

    def __post_init__(self) -> None:
        if self.case_count < 0:
            raise ValueError("judge case_count must not be negative")
        values = (self.relevance, self.faithfulness, self.completeness)
        if self.case_count == 0:
            if any(value is not None for value in values):
                raise ValueError("empty judge summary must not contain averages")
            return
        if any(value is None for value in values):
            raise ValueError("non-empty judge summary must contain all averages")
        for value, name in zip(
            values,
            ("relevance", "faithfulness", "completeness"),
            strict=True,
        ):
            assert value is not None
            _require_bounded_metric(value, f"{name} judge metric")


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """The answer outcome and metrics retained for one evaluation case."""

    case: EvaluationCase
    answer_status: AnswerStatus
    model_id: str | None
    cited_document_ids: tuple[str, ...]
    metrics: CaseMetrics
    judge: JudgeEvaluation | None = None

    def __post_init__(self) -> None:
        if self.model_id is not None:
            _require_text(self.model_id, "model_id")
        unique_document_ids = tuple(dict.fromkeys(self.cited_document_ids))
        for document_id in unique_document_ids:
            _require_text(document_id, "cited_document_ids")
        object.__setattr__(self, "cited_document_ids", unique_document_ids)
        if self.judge is not None and self.answer_status is not AnswerStatus.ANSWERED:
            raise ValueError("judge evaluation is valid only for answered cases")


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
    judge_configuration: JudgeConfiguration | None = None
    judge_overall_summary: JudgeSummary | None = None
    judge_language_summaries: Mapping[EvaluationLanguage, JudgeSummary] | None = None

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
        judge_values = (
            self.judge_configuration,
            self.judge_overall_summary,
            self.judge_language_summaries,
        )
        if any(value is not None for value in judge_values) and any(
            value is None for value in judge_values
        ):
            raise ValueError("judge report fields must be all present or all absent")
        if self.judge_configuration is None:
            if any(case.judge is not None for case in self.cases):
                raise ValueError("judge case results require judge report fields")
            return
        assert self.judge_language_summaries is not None
        if set(self.judge_language_summaries) != _LANGUAGES:
            raise ValueError("judge language summaries must contain exactly EN and ZH")
        if any(
            case.answer_status is AnswerStatus.ANSWERED and case.judge is None
            for case in self.cases
        ):
            raise ValueError("answered cases require judge results in judge mode")
        object.__setattr__(
            self,
            "judge_language_summaries",
            MappingProxyType(dict(self.judge_language_summaries)),
        )

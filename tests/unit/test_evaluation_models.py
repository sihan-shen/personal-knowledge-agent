"""Tests for immutable evaluation domain values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import inf, nan
from pathlib import Path

import pytest

from personal_knowledge_agent.application.evaluate import EvaluationCorpus
from personal_knowledge_agent.domain.evaluation import (
    CaseEvaluation,
    CaseMetrics,
    EvaluationCase,
    EvaluationDocument,
    EvaluationLanguage,
    EvaluationReport,
    EvaluationSuite,
    MetricSummary,
)
from personal_knowledge_agent.domain.models import AnswerStatus, SearchMode


def case() -> EvaluationCase:
    return EvaluationCase(
        "case:en",
        EvaluationLanguage.EN,
        "What is the capital of France?",
        ("Paris",),
        "doc:en",
    )


def suite() -> EvaluationSuite:
    return EvaluationSuite(
        "xquad-mini",
        "v1",
        (EvaluationDocument("doc:en", "france.md"),),
        (case(),),
        "Public-domain fixture.",
    )


def summary() -> MetricSummary:
    return MetricSummary(1, 1.0, 1.0, 1.0, 1.0)


def test_case_is_immutable() -> None:
    item = case()

    with pytest.raises(FrozenInstanceError):
        item.question = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["case_id", "question", "gold_document_id"])
def test_case_rejects_blank_required_text(field_name: str) -> None:
    values: dict[str, object] = {
        "case_id": "case:en",
        "language": EvaluationLanguage.EN,
        "question": "question",
        "reference_answers": ("answer",),
        "gold_document_id": "doc:en",
    }
    values[field_name] = "  "

    with pytest.raises(ValueError, match=field_name):
        EvaluationCase(**values)  # type: ignore[arg-type]


def test_case_rejects_empty_reference_answers() -> None:
    with pytest.raises(ValueError, match="reference_answers"):
        EvaluationCase("case:en", EvaluationLanguage.EN, "question", (), "doc:en")


def test_case_rejects_blank_reference_answer() -> None:
    with pytest.raises(ValueError, match="reference_answers"):
        EvaluationCase("case:en", EvaluationLanguage.EN, "question", ("  ",), "doc:en")


@pytest.mark.parametrize("field_name", ["document_id", "resource_name"])
def test_document_rejects_blank_required_text(field_name: str) -> None:
    values: dict[str, str] = {"document_id": "doc:en", "resource_name": "en.md"}
    values[field_name] = "  "

    with pytest.raises(ValueError, match=field_name):
        EvaluationDocument(**values)


def test_suite_rejects_duplicate_document_ids() -> None:
    document = EvaluationDocument("doc:en", "en.md")

    with pytest.raises(ValueError, match="document IDs"):
        EvaluationSuite(
            "xquad-mini",
            "v1",
            (document, document),
            (case(),),
            "Public-domain fixture.",
        )


def test_suite_rejects_case_with_missing_gold_document() -> None:
    missing_gold_case = EvaluationCase(
        "case:en",
        EvaluationLanguage.EN,
        "question",
        ("answer",),
        "doc:missing",
    )

    with pytest.raises(ValueError, match="gold_document_id"):
        EvaluationSuite(
            "xquad-mini",
            "v1",
            (EvaluationDocument("doc:en", "en.md"),),
            (missing_gold_case,),
            "Public-domain fixture.",
        )


@pytest.mark.parametrize("metric", [nan, inf, -inf, -0.1, 1.1])
def test_case_metrics_rejects_non_finite_or_out_of_range_values(metric: float) -> None:
    with pytest.raises(ValueError, match="metric"):
        CaseMetrics(metric, 1.0, 1.0, 1.0)


def test_case_evaluation_deduplicates_cited_document_ids_in_first_seen_order() -> None:
    evaluation = CaseEvaluation(
        case(),
        AnswerStatus.ANSWERED,
        "fake-model",
        ("doc:en", "doc:other", "doc:en"),
        CaseMetrics(1.0, 1.0, 0.5, 1.0),
    )

    assert evaluation.cited_document_ids == ("doc:en", "doc:other")


def test_case_evaluation_rejects_blank_cited_document_id() -> None:
    with pytest.raises(ValueError, match="cited_document_ids"):
        CaseEvaluation(
            case(),
            AnswerStatus.ANSWERED,
            "fake-model",
            ("  ",),
            CaseMetrics(1.0, 1.0, 1.0, 1.0),
        )


def test_metric_summary_requires_positive_count() -> None:
    with pytest.raises(ValueError, match="count"):
        MetricSummary(0, 1.0, 1.0, 1.0, 1.0)


def test_report_requires_english_and_chinese_summaries() -> None:
    with pytest.raises(ValueError, match="language summaries"):
        EvaluationReport("xquad-mini", "v1", SearchMode.HYBRID, 5, (), summary(), {})


def test_report_prevents_mutation_of_language_summaries() -> None:
    report = EvaluationReport(
        "xquad-mini",
        "v1",
        SearchMode.HYBRID,
        5,
        (),
        summary(),
        {EvaluationLanguage.EN: summary(), EvaluationLanguage.ZH: summary()},
    )

    with pytest.raises(TypeError):
        report.language_summaries[EvaluationLanguage.EN] = summary()  # type: ignore[index]


@pytest.mark.parametrize("limit", [0, 11])
def test_report_rejects_limit_outside_one_through_ten(limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        EvaluationReport(
            "xquad-mini",
            "v1",
            SearchMode.HYBRID,
            limit,
            (),
            summary(),
            {EvaluationLanguage.EN: summary(), EvaluationLanguage.ZH: summary()},
        )


def test_corpus_rejects_missing_or_extra_document_paths(tmp_path: Path) -> None:
    document_path = tmp_path / "en.md"
    document_path.write_text("fixture", encoding="utf-8")

    with pytest.raises(ValueError, match="document paths"):
        EvaluationCorpus(suite(), {})
    with pytest.raises(ValueError, match="document paths"):
        EvaluationCorpus(suite(), {"doc:en": document_path, "extra": document_path})


def test_corpus_rejects_a_path_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing file"):
        EvaluationCorpus(suite(), {"doc:en": tmp_path / "missing.md"})

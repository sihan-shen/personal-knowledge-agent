"""Tests for pure deterministic RAG evaluation metrics."""

from __future__ import annotations

import pytest

from personal_knowledge_agent.domain.errors import InvalidEvaluationError
from personal_knowledge_agent.domain.evaluation import (
    CaseMetrics,
    EvaluationCase,
    EvaluationLanguage,
    MetricSummary,
)
from personal_knowledge_agent.domain.models import AnswerStatus, GroundedAnswer
from personal_knowledge_agent.infrastructure.evaluation_metrics import (
    score_case,
    summarize,
)


def case_with_answers(*answers: str) -> EvaluationCase:
    return EvaluationCase(
        "case:en",
        EvaluationLanguage.EN,
        "question",
        answers,
        "gold-doc",
    )


def answered(text: str) -> GroundedAnswer:
    return GroundedAnswer(AnswerStatus.ANSWERED, text, (), (), "fake-model")


def insufficient_evidence() -> GroundedAnswer:
    return GroundedAnswer(
        AnswerStatus.INSUFFICIENT_EVIDENCE,
        "Evidence is insufficient.",
        (),
        (),
        None,
    )


def test_score_case_removes_only_document_citations_before_exact_match() -> None:
    answer = answered("Ｐａｒｉｓ [S1]")
    case = case_with_answers("Paris", "City of Paris")

    assert score_case(case, answer, ("gold-doc",)) == CaseMetrics(1.0, 1.0, 1.0, 1.0)


def test_score_case_uses_chinese_bigram_overlap_and_best_alias() -> None:
    answer = answered("检索增强生成 [S1]")
    case = case_with_answers("错误答案", "检索增强生成")

    assert score_case(case, answer, ("gold-doc",)).answer_token_f1 == 1.0


def test_score_case_penalizes_wrong_and_missing_citations() -> None:
    case = case_with_answers("answer")

    assert (
        score_case(case, answered("answer [S1]"), ("other-doc",)).citation_precision
        == 0.0
    )
    assert score_case(case, answered("answer [S1]"), ()).citation_recall == 0.0


def test_score_case_deduplicates_repeated_cited_document_ids() -> None:
    metrics = score_case(
        case_with_answers("answer"),
        answered("answer [S1]"),
        ("gold-doc", "gold-doc", "other-doc"),
    )

    assert metrics.citation_precision == 0.5
    assert metrics.citation_recall == 1.0


def test_score_case_normalizes_case_and_punctuation_variation() -> None:
    metrics = score_case(
        case_with_answers("Paris, France!"),
        answered("PARIS france. [S12]"),
        ("gold-doc",),
    )

    assert metrics.answer_exact_match == 1.0
    assert metrics.answer_token_f1 == 1.0


def test_score_case_returns_zero_answer_scores_for_no_token_overlap() -> None:
    metrics = score_case(
        case_with_answers("Paris"),
        answered("Beijing [S1]"),
        ("gold-doc",),
    )

    assert metrics.answer_exact_match == 0.0
    assert metrics.answer_token_f1 == 0.0


def test_score_case_returns_all_zeros_for_insufficient_evidence() -> None:
    metrics = score_case(
        case_with_answers("answer"),
        insufficient_evidence(),
        ("gold-doc",),
    )

    assert metrics == CaseMetrics(0.0, 0.0, 0.0, 0.0)


def test_summarize_rejects_empty_metrics() -> None:
    with pytest.raises(InvalidEvaluationError, match="at least one"):
        summarize(())


def test_summarize_returns_macro_averages_for_two_cases() -> None:
    summary = summarize(
        (
            CaseMetrics(1.0, 0.5, 1.0, 0.0),
            CaseMetrics(0.0, 1.0, 0.0, 1.0),
        )
    )

    assert summary == MetricSummary(2, 0.5, 0.75, 0.5, 0.5)

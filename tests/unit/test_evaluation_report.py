"""Tests for stable, safe evaluation report persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import EvaluationStorageError
from personal_knowledge_agent.domain.evaluation import (
    CaseEvaluation,
    CaseMetrics,
    EvaluationCase,
    EvaluationLanguage,
    EvaluationReport,
    MetricSummary,
)
from personal_knowledge_agent.domain.models import AnswerStatus, SearchMode
from personal_knowledge_agent.infrastructure.evaluation_report import (
    report_as_json,
    write_report,
)


def summary(case_count: int = 2) -> MetricSummary:
    return MetricSummary(case_count, 1.0, 0.5, 0.75, 1.0)


def report() -> EvaluationReport:
    english_case = EvaluationCase(
        "case:en",
        EvaluationLanguage.EN,
        "What is the capital of France?",
        ("Paris",),
        "gold-doc",
    )
    chinese_case = EvaluationCase(
        "case:zh",
        EvaluationLanguage.ZH,
        "法国的首都是哪里？",
        ("巴黎",),
        "gold-doc-zh",
    )
    return EvaluationReport(
        "xquad-mini",
        "v1",
        SearchMode.HYBRID,
        5,
        (
            CaseEvaluation(
                english_case,
                AnswerStatus.ANSWERED,
                "fake-model",
                ("gold-doc",),
                CaseMetrics(1.0, 0.5, 0.75, 1.0),
            ),
            CaseEvaluation(
                chinese_case,
                AnswerStatus.INSUFFICIENT_EVIDENCE,
                None,
                (),
                CaseMetrics(0.0, 0.0, 0.0, 0.0),
            ),
        ),
        summary(),
        {EvaluationLanguage.EN: summary(1), EvaluationLanguage.ZH: summary(1)},
    )


def _assert_sorted_keys(value: object) -> None:
    if isinstance(value, Mapping):
        assert list(value) == sorted(value)
        for nested_value in value.values():
            _assert_sorted_keys(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _assert_sorted_keys(nested_value)


def test_report_as_json_is_byte_stable_and_uses_logical_ids_only() -> None:
    serialized = report_as_json(report())

    assert serialized == report_as_json(report())
    assert serialized.endswith("\n")
    assert "法国的首都是哪里？" not in serialized
    assert "/absolute/source.md" not in serialized
    payload = json.loads(serialized)
    assert payload["cases"][0]["cited_document_ids"] == ["gold-doc"]
    assert [case["case_id"] for case in payload["cases"]] == ["case:en", "case:zh"]
    _assert_sorted_keys(payload)


def test_write_report_writes_utf8_json_with_final_newline(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    write_report(report(), output)

    assert output.read_bytes() == report_as_json(report()).encode("utf-8")


def test_write_report_wraps_filesystem_errors() -> None:
    with pytest.raises(EvaluationStorageError, match="report"):
        write_report(report(), Path("/missing-parent/report.json"))


def test_write_report_removes_temporary_file_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"

    def raise_os_error(_: Path, __: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", raise_os_error)

    with pytest.raises(EvaluationStorageError, match="report"):
        write_report(report(), output)

    assert list(tmp_path.iterdir()) == []

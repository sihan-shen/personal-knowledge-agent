"""Stable JSON serialization and atomic persistence for evaluation reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from personal_knowledge_agent.domain.errors import EvaluationStorageError
from personal_knowledge_agent.domain.evaluation import (
    CaseEvaluation,
    EvaluationReport,
    JudgeConfiguration,
    JudgeEvaluation,
    JudgeMetricScore,
    JudgeSummary,
    MetricSummary,
)


def _summary_as_dict(summary: MetricSummary) -> dict[str, int | float]:
    return {
        "case_count": summary.case_count,
        "answer_exact_match": summary.answer_exact_match,
        "answer_token_f1": summary.answer_token_f1,
        "citation_precision": summary.citation_precision,
        "citation_recall": summary.citation_recall,
    }


def _case_as_dict(case_evaluation: CaseEvaluation) -> dict[str, object]:
    metrics = case_evaluation.metrics
    payload: dict[str, object] = {
        "case_id": case_evaluation.case.case_id,
        "language": case_evaluation.case.language.value,
        "status": case_evaluation.answer_status.value,
        "model_id": case_evaluation.model_id,
        "cited_document_ids": list(case_evaluation.cited_document_ids),
        "metrics": {
            "answer_exact_match": metrics.answer_exact_match,
            "answer_token_f1": metrics.answer_token_f1,
            "citation_precision": metrics.citation_precision,
            "citation_recall": metrics.citation_recall,
        },
    }
    if case_evaluation.judge is not None:
        payload["judge"] = _judge_evaluation_as_dict(case_evaluation.judge)
    return payload


def _judge_score_as_dict(score: JudgeMetricScore) -> dict[str, int | str]:
    return {"score": score.score, "reason": score.reason}


def _judge_evaluation_as_dict(evaluation: JudgeEvaluation) -> dict[str, object]:
    return {
        "relevance": _judge_score_as_dict(evaluation.relevance),
        "faithfulness": _judge_score_as_dict(evaluation.faithfulness),
        "completeness": _judge_score_as_dict(evaluation.completeness),
    }


def _judge_configuration_as_dict(
    configuration: JudgeConfiguration,
) -> dict[str, str]:
    return {
        "model_id": configuration.model_id,
        "faithfulness_prompt_version": configuration.faithfulness_prompt_version,
        "reference_quality_prompt_version": (
            configuration.reference_quality_prompt_version
        ),
    }


def _judge_summary_as_dict(summary: JudgeSummary) -> dict[str, int | float | None]:
    return {
        "case_count": summary.case_count,
        "relevance": summary.relevance,
        "faithfulness": summary.faithfulness,
        "completeness": summary.completeness,
    }


def report_as_json(report: EvaluationReport) -> str:
    """Serialize a report using stable, privacy-safe primitive values."""
    payload: dict[str, object] = {
        "schema_version": 1,
        "suite": {"id": report.suite_id, "version": report.suite_version},
        "configuration": {"mode": report.mode.value, "limit": report.limit},
        "cases": [_case_as_dict(case) for case in report.cases],
        "summary": {
            "overall": _summary_as_dict(report.overall_summary),
            "by_language": {
                language.value: _summary_as_dict(summary)
                for language, summary in report.language_summaries.items()
            },
        },
    }
    if report.judge_configuration is not None:
        assert report.judge_overall_summary is not None
        assert report.judge_language_summaries is not None
        payload["schema_version"] = 2
        payload["judge_configuration"] = _judge_configuration_as_dict(
            report.judge_configuration
        )
        payload["judge_summary"] = {
            **_judge_summary_as_dict(report.judge_overall_summary),
            "by_language": {
                language.value: _judge_summary_as_dict(summary)
                for language, summary in report.judge_language_summaries.items()
            },
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def write_report(report: EvaluationReport, output: Path) -> None:
    """Atomically write a stable evaluation report to ``output``."""
    temporary_path: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(report_as_json(report))
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(output)
    except OSError as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise EvaluationStorageError("Could not write evaluation report") from error

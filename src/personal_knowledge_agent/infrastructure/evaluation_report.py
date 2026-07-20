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
    return {
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


def report_as_json(report: EvaluationReport) -> str:
    """Serialize a report using stable, privacy-safe primitive values."""
    payload = {
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

"""Pure deterministic answer and citation metrics for RAG evaluation."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from math import fsum

from personal_knowledge_agent.domain.errors import InvalidEvaluationError
from personal_knowledge_agent.domain.evaluation import (
    CaseMetrics,
    EvaluationCase,
    JudgeEvaluation,
    JudgeSummary,
    MetricSummary,
)
from personal_knowledge_agent.domain.models import AnswerStatus, GroundedAnswer
from personal_knowledge_agent.infrastructure.mixed_tokenizer import (
    MixedLanguageTokenizer,
)

_DOCUMENT_CITATION = re.compile(r"\[S[1-9]\d*\]")
_TOKENIZER = MixedLanguageTokenizer()


def _normalize_answer(value: str) -> str:
    without_citations = _DOCUMENT_CITATION.sub("", value)
    normalized = unicodedata.normalize("NFKC", without_citations).casefold()
    without_punctuation = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


def _token_f1(answer: str, reference: str) -> float:
    answer_tokens = Counter(_TOKENIZER.tokenize(answer))
    reference_tokens = Counter(_TOKENIZER.tokenize(reference))
    token_count = sum(answer_tokens.values()) + sum(reference_tokens.values())
    if token_count == 0:
        return 0.0
    overlap = sum((answer_tokens & reference_tokens).values())
    return 2.0 * overlap / token_count


def score_case(
    case: EvaluationCase,
    answer: GroundedAnswer,
    cited_document_ids: tuple[str, ...],
) -> CaseMetrics:
    """Score one answer against its aliases and expected document citation."""
    if answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        return CaseMetrics(0.0, 0.0, 0.0, 0.0)

    normalized_answer = _normalize_answer(answer.answer)
    normalized_references = tuple(
        _normalize_answer(reference) for reference in case.reference_answers
    )
    exact_match = float(normalized_answer in normalized_references)
    token_f1 = max(
        _token_f1(normalized_answer, reference) for reference in normalized_references
    )

    unique_document_ids = tuple(dict.fromkeys(cited_document_ids))
    matching_citations = sum(
        document_id == case.gold_document_id for document_id in unique_document_ids
    )
    citation_precision = (
        matching_citations / len(unique_document_ids) if unique_document_ids else 0.0
    )
    citation_recall = float(matching_citations > 0)

    return CaseMetrics(exact_match, token_f1, citation_precision, citation_recall)


def summarize(metrics: tuple[CaseMetrics, ...]) -> MetricSummary:
    """Return deterministic macro averages over evaluated cases."""
    if not metrics:
        raise InvalidEvaluationError("metrics must contain at least one case")

    count = len(metrics)
    return MetricSummary(
        count,
        fsum(item.answer_exact_match for item in metrics) / count,
        fsum(item.answer_token_f1 for item in metrics) / count,
        fsum(item.citation_precision for item in metrics) / count,
        fsum(item.citation_recall for item in metrics) / count,
    )


def summarize_judge(evaluations: tuple[JudgeEvaluation, ...]) -> JudgeSummary:
    """Return normalized macro averages, including the no-answered-case outcome."""
    if not evaluations:
        return JudgeSummary(0, None, None, None)

    count = len(evaluations)
    return JudgeSummary(
        count,
        fsum(item.relevance.score / 2.0 for item in evaluations) / count,
        fsum(item.faithfulness.score / 2.0 for item in evaluations) / count,
        fsum(item.completeness.score / 2.0 for item in evaluations) / count,
    )

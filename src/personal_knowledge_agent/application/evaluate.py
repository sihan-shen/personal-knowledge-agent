"""Application boundaries for evaluation corpora."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from personal_knowledge_agent.application.ingest import IngestService
from personal_knowledge_agent.domain.errors import InvalidEvaluationError
from personal_knowledge_agent.domain.evaluation import (
    CaseEvaluation,
    EvaluationCase,
    EvaluationLanguage,
    EvaluationReport,
    EvaluationSuite,
)
from personal_knowledge_agent.domain.models import GroundedAnswer, SearchMode
from personal_knowledge_agent.domain.protocols import GroundedQuestionAnswerer
from personal_knowledge_agent.infrastructure.evaluation_metrics import (
    score_case,
    summarize,
)


@dataclass(frozen=True, slots=True)
class EvaluationCorpus:
    """A suite paired with its exact, existing local document files."""

    suite: EvaluationSuite
    document_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        expected_document_ids = {
            document.document_id for document in self.suite.documents
        }
        if set(self.document_paths) != expected_document_ids:
            raise ValueError("document paths must match suite document IDs exactly")
        paths = dict(self.document_paths)
        if any(not path.is_file() for path in paths.values()):
            raise ValueError("document paths must refer to an existing file")
        object.__setattr__(self, "document_paths", MappingProxyType(paths))


class EvaluationCorpusLoader(Protocol):
    """Loads an evaluation suite together with local fixture paths."""

    def load(self) -> EvaluationCorpus:
        """Return the complete evaluation corpus."""


class EvaluationService:
    """Run one isolated, reproducible evaluation corpus through the RAG stack."""

    def __init__(
        self,
        corpus_loader: EvaluationCorpusLoader,
        ingester: IngestService,
        answerer: GroundedQuestionAnswerer,
    ) -> None:
        self._corpus_loader = corpus_loader
        self._ingester = ingester
        self._answerer = answerer

    def run(
        self,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 5,
    ) -> EvaluationReport:
        """Ingest the declared corpus, answer every case, and score the outcomes."""
        if not 1 <= limit <= 10:
            raise InvalidEvaluationError(
                "evaluation evidence limit must be between 1 and 10"
            )

        corpus = self._corpus_loader.load()
        source_document_ids = {
            path.resolve(): document_id
            for document_id, path in corpus.document_paths.items()
        }
        for document in corpus.suite.documents:
            self._ingester.ingest(corpus.document_paths[document.document_id])

        evaluations = tuple(
            self._evaluate_case(case, source_document_ids, limit, mode)
            for case in corpus.suite.cases
        )
        language_summaries = {
            language: summarize(
                tuple(
                    evaluation.metrics
                    for evaluation in evaluations
                    if evaluation.case.language is language
                )
            )
            for language in EvaluationLanguage
        }
        return EvaluationReport(
            corpus.suite.suite_id,
            corpus.suite.version,
            mode,
            limit,
            evaluations,
            summarize(tuple(evaluation.metrics for evaluation in evaluations)),
            language_summaries,
        )

    def _evaluate_case(
        self,
        case: EvaluationCase,
        source_document_ids: Mapping[Path, str],
        limit: int,
        mode: SearchMode,
    ) -> CaseEvaluation:
        answer = self._answerer.answer(case.question, limit=limit, mode=mode)
        cited_document_ids = self._cited_document_ids(answer, source_document_ids)
        return CaseEvaluation(
            case,
            answer.status,
            answer.model_id,
            cited_document_ids,
            score_case(case, answer, cited_document_ids),
        )

    @staticmethod
    def _cited_document_ids(
        answer: GroundedAnswer,
        source_document_ids: Mapping[Path, str],
    ) -> tuple[str, ...]:
        evidence_by_citation = {
            evidence.citation_id: evidence for evidence in answer.evidence
        }
        cited_document_ids: list[str] = []
        for citation_id in answer.citation_ids:
            evidence = evidence_by_citation.get(citation_id)
            if evidence is None:
                continue
            try:
                document_id = source_document_ids[Path(evidence.source).resolve()]
            except KeyError as error:
                raise InvalidEvaluationError(
                    "cited evidence source is not a suite document"
                ) from error
            if document_id not in cited_document_ids:
                cited_document_ids.append(document_id)
        return tuple(cited_document_ids)

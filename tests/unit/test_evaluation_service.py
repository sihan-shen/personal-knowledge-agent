"""Tests for isolated evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from personal_knowledge_agent.application.evaluate import (
    EvaluationCorpus,
    EvaluationCorpusLoader,
    EvaluationService,
)
from personal_knowledge_agent.domain.errors import InvalidEvaluationError
from personal_knowledge_agent.domain.evaluation import (
    EvaluationCase,
    EvaluationDocument,
    EvaluationLanguage,
    EvaluationSuite,
    FaithfulnessJudgeRequest,
    JudgeConfiguration,
    JudgeMetric,
    JudgeMetricScore,
    ReferenceQualityJudgeRequest,
    ReferenceQualityScores,
)
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    Evidence,
    GroundedAnswer,
    SearchMode,
)


@dataclass
class FakeCorpusLoader(EvaluationCorpusLoader):
    corpus: EvaluationCorpus
    calls: int = 0

    def load(self) -> EvaluationCorpus:
        self.calls += 1
        return self.corpus

    def allows(self, *values: str) -> bool:
        return True


@dataclass
class FakeIngester:
    paths: list[Path] = field(default_factory=list)

    def ingest(self, path: Path) -> None:
        self.paths.append(path)


@dataclass
class FakeAnswerer:
    answers: list[GroundedAnswer]
    calls: list[tuple[str, int, SearchMode]] = field(default_factory=list)

    def answer(self, question: str, *, limit: int, mode: SearchMode) -> GroundedAnswer:
        self.calls.append((question, limit, mode))
        return self.answers.pop(0)


@dataclass
class FakeFaithfulnessJudge:
    calls: list[str]
    requests: list[FaithfulnessJudgeRequest] = field(default_factory=list)

    def judge_faithfulness(self, request: FaithfulnessJudgeRequest) -> JudgeMetricScore:
        self.calls.append(f"faithfulness:{request.question}")
        self.requests.append(request)
        return JudgeMetricScore(JudgeMetric.FAITHFULNESS, 2, "Supported.")


@dataclass
class FakeReferenceQualityJudge:
    calls: list[str]
    requests: list[ReferenceQualityJudgeRequest] = field(default_factory=list)

    def judge_reference_quality(
        self, request: ReferenceQualityJudgeRequest
    ) -> ReferenceQualityScores:
        self.calls.append(f"quality:{request.question}")
        self.requests.append(request)
        return ReferenceQualityScores(
            JudgeMetricScore(JudgeMetric.RELEVANCE, 2, "Direct."),
            JudgeMetricScore(JudgeMetric.COMPLETENESS, 1, "Partial."),
        )


def judge_configuration() -> JudgeConfiguration:
    return JudgeConfiguration("judge-model", "faithfulness-judge-v1", "quality-v1")


def make_suite() -> EvaluationSuite:
    return EvaluationSuite(
        "suite",
        "v1",
        (
            EvaluationDocument("doc:en", "english.md"),
            EvaluationDocument("doc:zh", "chinese.md"),
        ),
        (
            EvaluationCase(
                "case:en",
                EvaluationLanguage.EN,
                "English question",
                ("Paris",),
                "doc:en",
            ),
            EvaluationCase(
                "case:zh", EvaluationLanguage.ZH, "中文问题", ("北京",), "doc:zh"
            ),
        ),
        "attribution",
    )


def answered(
    text: str, *evidence: Evidence, citations: tuple[str, ...]
) -> GroundedAnswer:
    return GroundedAnswer(
        AnswerStatus.ANSWERED, text, citations, evidence, "fake-model"
    )


def evidence(citation_id: str, source: Path) -> Evidence:
    return Evidence(citation_id, f"chunk-{citation_id}", str(source), (), "text", 0.5)


@pytest.fixture
def corpus(tmp_path: Path) -> EvaluationCorpus:
    english_path = tmp_path / "english.md"
    chinese_path = tmp_path / "chinese.md"
    english_path.write_text("English", encoding="utf-8")
    chinese_path.write_text("中文", encoding="utf-8")
    return EvaluationCorpus(
        make_suite(), {"doc:en": english_path, "doc:zh": chinese_path}
    )


def test_run_ingests_documents_then_answers_cases_in_manifest_order(
    corpus: EvaluationCorpus,
) -> None:
    english_path = corpus.document_paths["doc:en"]
    chinese_path = corpus.document_paths["doc:zh"]
    ingester = FakeIngester()
    answerer = FakeAnswerer(
        [
            answered("Paris", evidence("S1", english_path), citations=("S1",)),
            answered("北京", evidence("S1", chinese_path), citations=("S1",)),
        ]
    )
    service = EvaluationService(FakeCorpusLoader(corpus), ingester, answerer)

    report = service.run(mode=SearchMode.SEMANTIC, limit=3)

    assert ingester.paths == [english_path, chinese_path]
    assert answerer.calls == [
        ("English question", 3, SearchMode.SEMANTIC),
        ("中文问题", 3, SearchMode.SEMANTIC),
    ]
    assert [item.case.case_id for item in report.cases] == ["case:en", "case:zh"]


def test_run_maps_only_cited_evidence_sources_to_logical_ids_in_citation_order(
    corpus: EvaluationCorpus,
) -> None:
    english_path = corpus.document_paths["doc:en"]
    chinese_path = corpus.document_paths["doc:zh"]
    answerer = FakeAnswerer(
        [
            answered(
                "Paris",
                evidence("S1", chinese_path),
                evidence("S2", english_path),
                evidence("S3", english_path),
                citations=("S2", "S1", "S3"),
            ),
            answered("北京", citations=()),
        ]
    )
    service = EvaluationService(FakeCorpusLoader(corpus), FakeIngester(), answerer)

    report = service.run()

    assert report.cases[0].cited_document_ids == ("doc:en", "doc:zh")
    assert report.cases[0].metrics.citation_precision == 0.5
    assert report.cases[0].metrics.citation_recall == 1.0


def test_run_handles_answers_without_citations(corpus: EvaluationCorpus) -> None:
    answerer = FakeAnswerer(
        [answered("Paris", citations=()), answered("北京", citations=())]
    )
    service = EvaluationService(FakeCorpusLoader(corpus), FakeIngester(), answerer)

    report = service.run()

    assert report.cases[0].cited_document_ids == ()
    assert report.cases[0].metrics.citation_precision == 0.0
    assert report.cases[0].metrics.citation_recall == 0.0


def test_run_calls_isolated_judges_in_case_order_for_answered_cases(
    corpus: EvaluationCorpus,
) -> None:
    english_path = corpus.document_paths["doc:en"]
    chinese_path = corpus.document_paths["doc:zh"]
    calls: list[str] = []
    faithfulness = FakeFaithfulnessJudge(calls)
    quality = FakeReferenceQualityJudge(calls)
    service = EvaluationService(
        FakeCorpusLoader(corpus),
        FakeIngester(),
        FakeAnswerer(
            [
                answered("Paris", evidence("S1", english_path), citations=("S1",)),
                answered("北京", evidence("S1", chinese_path), citations=("S1",)),
            ]
        ),
        faithfulness,
        quality,
        judge_configuration(),
    )

    report = service.run()

    assert calls == [
        "faithfulness:English question",
        "quality:English question",
        "faithfulness:中文问题",
        "quality:中文问题",
    ]
    assert faithfulness.requests[0].evidence[0].source == str(english_path)
    assert quality.requests[0].reference_answers == ("Paris",)
    assert report.cases[0].judge is not None
    assert report.judge_overall_summary is not None
    assert report.judge_overall_summary.faithfulness == 1.0
    assert report.judge_overall_summary.completeness == 0.5


def test_run_skips_judges_for_insufficient_evidence(corpus: EvaluationCorpus) -> None:
    insufficient = GroundedAnswer(
        AnswerStatus.INSUFFICIENT_EVIDENCE, "Insufficient evidence.", (), (), None
    )
    calls: list[str] = []
    service = EvaluationService(
        FakeCorpusLoader(corpus),
        FakeIngester(),
        FakeAnswerer([insufficient, insufficient]),
        FakeFaithfulnessJudge(calls),
        FakeReferenceQualityJudge(calls),
        judge_configuration(),
    )

    report = service.run()

    assert calls == []
    assert report.judge_overall_summary is not None
    assert report.judge_overall_summary.case_count == 0
    assert report.judge_overall_summary.faithfulness is None


def test_run_scores_insufficient_evidence_as_all_zeros(
    corpus: EvaluationCorpus,
) -> None:
    insufficient = GroundedAnswer(
        AnswerStatus.INSUFFICIENT_EVIDENCE, "Insufficient evidence.", (), (), None
    )
    answerer = FakeAnswerer([insufficient, insufficient])
    service = EvaluationService(FakeCorpusLoader(corpus), FakeIngester(), answerer)

    report = service.run()

    assert report.cases[0].metrics.answer_exact_match == 0.0
    assert report.cases[0].metrics.answer_token_f1 == 0.0
    assert report.cases[0].metrics.citation_precision == 0.0
    assert report.cases[0].metrics.citation_recall == 0.0
    assert report.overall_summary.answer_exact_match == 0.0


def test_run_rejects_cited_evidence_outside_the_suite(
    corpus: EvaluationCorpus, tmp_path: Path
) -> None:
    outsider = tmp_path / "outsider.md"
    outsider.write_text("outside", encoding="utf-8")
    answerer = FakeAnswerer(
        [
            answered("Paris", evidence("S1", outsider), citations=("S1",)),
            answered("北京", citations=()),
        ]
    )
    service = EvaluationService(FakeCorpusLoader(corpus), FakeIngester(), answerer)

    with pytest.raises(InvalidEvaluationError, match="suite document"):
        service.run()


def test_run_propagates_answerer_failure_without_returning_partial_report(
    corpus: EvaluationCorpus,
) -> None:
    class FailingAnswerer:
        def __init__(self) -> None:
            self.calls = 0

        def answer(
            self, question: str, *, limit: int, mode: SearchMode
        ) -> GroundedAnswer:
            self.calls += 1
            raise RuntimeError("provider unavailable")

    answerer = FailingAnswerer()
    service = EvaluationService(FakeCorpusLoader(corpus), FakeIngester(), answerer)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        service.run()

    assert answerer.calls == 1


def test_run_rejects_invalid_limit_before_calling_dependencies(
    corpus: EvaluationCorpus,
) -> None:
    loader = FakeCorpusLoader(corpus)
    ingester = FakeIngester()
    answerer = FakeAnswerer([])
    service = EvaluationService(loader, ingester, answerer)

    with pytest.raises(InvalidEvaluationError, match="limit"):
        service.run(limit=0)

    assert loader.calls == 0
    assert ingester.paths == []
    assert answerer.calls == []

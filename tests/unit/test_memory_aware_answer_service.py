"""Tests for document and memory-aware answer coordination."""

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from personal_knowledge_agent.application.memory_aware_answer import (
    MemoryAwareAnswerService,
)
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.errors import (
    IndexStorageError,
    InvalidQueryError,
    KnowledgeBaseEmptyError,
)
from personal_knowledge_agent.domain.memory import (
    MemoryEvidence,
    MemoryRecallReport,
    MemoryRecallWarning,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    memory_content_hash,
)
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    Chunk,
    Document,
    ModelOutput,
    SearchMode,
    SearchResult,
)
from tests.helpers import FakeLanguageModel

_NOW = datetime(2026, 7, 18, tzinfo=UTC)


class FakeIndex:
    def __init__(self, result: list[SearchResult] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, int, SearchMode]] = []

    def replace_document(self, document: Document, chunks: Sequence[Chunk]) -> bool:
        del document, chunks
        return False

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        mode: SearchMode = SearchMode.LEXICAL,
    ) -> list[SearchResult]:
        self.calls.append((query, limit, mode))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeRecaller:
    def __init__(self, report: MemoryRecallReport) -> None:
        self.report = report
        self.calls: list[tuple[str, int]] = []

    def recall(self, query: str, *, limit: int = 3) -> MemoryRecallReport:
        self.calls.append((query, limit))
        return self.report


def memory_report(*, warning: bool = False) -> MemoryRecallReport:
    content = "Prefer concise answers"
    record = MemoryRecord(
        "mem_00000000000000000000000000000001",
        MemoryType.PREFERENCE,
        "answer_style",
        content,
        memory_content_hash(content),
        MemoryStatus.ACTIVE,
        None,
        None,
        _NOW,
        _NOW,
    )
    warnings = (MemoryRecallWarning.UNSAFE_MEMORY,) if warning else ()
    return MemoryRecallReport((MemoryEvidence("M1", record, 0.02),), warnings)


def document_results() -> list[SearchResult]:
    chunk = Chunk("chunk", "document", 0, "RAG uses retrieval.", ("RAG",))
    return [SearchResult(chunk, "/notes/rag.md", 0.9)]


@pytest.mark.parametrize(
    ("index_result", "report", "model_json", "expected_citations"),
    [
        (
            document_results(),
            MemoryRecallReport((), ()),
            '{"status":"answered","answer":"RAG retrieves. [S1]",'
            '"citation_ids":["S1"]}',
            ("S1",),
        ),
        (
            KnowledgeBaseEmptyError("empty"),
            memory_report(),
            '{"status":"answered","answer":"Be concise. [M1]","citation_ids":["M1"]}',
            ("M1",),
        ),
        (
            document_results(),
            memory_report(),
            '{"status":"answered","answer":"RAG [S1], concise [M1].",'
            '"citation_ids":["S1","M1"]}',
            ("S1", "M1"),
        ),
    ],
)
def test_answer_supports_document_memory_and_mixed_context(
    index_result: list[SearchResult] | Exception,
    report: MemoryRecallReport,
    model_json: str,
    expected_citations: tuple[str, ...],
) -> None:
    index = FakeIndex(index_result)
    recaller = FakeRecaller(report)
    model = FakeLanguageModel(ModelOutput("answer-model", model_json, "stop"))

    result = MemoryAwareAnswerService(SearchService(index), recaller, model).answer(
        "question", limit=4, mode=SearchMode.SEMANTIC
    )

    assert result.answer.citation_ids == expected_citations
    assert result.recall is report
    assert index.calls == [("question", 4, SearchMode.SEMANTIC)]
    assert recaller.calls == [("question", 3)]
    assert len(model.calls) == 1


def test_empty_context_returns_local_insufficient_without_model_call() -> None:
    index = FakeIndex([])
    recaller = FakeRecaller(MemoryRecallReport((), ()))
    model = FakeLanguageModel(AssertionError("model must not be called"))

    result = MemoryAwareAnswerService(SearchService(index), recaller, model).answer(
        "question"
    )

    assert result.answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.answer.model_id is None
    assert model.calls == []


def test_recall_warning_is_preserved_with_valid_answer() -> None:
    report = memory_report(warning=True)
    model = FakeLanguageModel(
        ModelOutput(
            "model",
            '{"status":"answered","answer":"Be concise. [M1]","citation_ids":["M1"]}',
            "stop",
        )
    )
    result = MemoryAwareAnswerService(
        SearchService(FakeIndex([])), FakeRecaller(report), model
    ).answer("question")
    assert result.recall.warnings == (MemoryRecallWarning.UNSAFE_MEMORY,)


def test_non_empty_document_failure_is_not_hidden() -> None:
    service = MemoryAwareAnswerService(
        SearchService(FakeIndex(IndexStorageError("broken"))),
        FakeRecaller(MemoryRecallReport((), ())),
        FakeLanguageModel(AssertionError("unused")),
    )
    with pytest.raises(IndexStorageError, match="broken"):
        service.answer("question")


@pytest.mark.parametrize(("question", "limit"), [("", 5), ("question", 0)])
def test_invalid_input_is_rejected_before_dependencies(
    question: str, limit: int
) -> None:
    index = FakeIndex([])
    recaller = FakeRecaller(MemoryRecallReport((), ()))
    service = MemoryAwareAnswerService(
        SearchService(index), recaller, FakeLanguageModel(AssertionError("unused"))
    )
    with pytest.raises(InvalidQueryError):
        service.answer(question, limit=limit)
    assert index.calls == []
    assert recaller.calls == []

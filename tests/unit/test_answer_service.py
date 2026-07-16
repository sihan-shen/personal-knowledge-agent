"""Tests for grounded-answer application orchestration."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from personal_knowledge_agent.application.answer import AnswerService
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.errors import (
    InvalidAnswerSchemaError,
    InvalidCitationError,
    InvalidQueryError,
    LanguageModelError,
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


class FakeIndex:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, SearchMode]] = []

    def replace_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> bool:
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
        return self.results


def result() -> SearchResult:
    chunk = Chunk("chunk-id", "document", 0, "RAG uses evidence.", ("RAG",))
    return SearchResult(chunk, "/notes/rag.md", 0.9)


def answered_output() -> ModelOutput:
    return ModelOutput(
        "fake-provider-model",
        '{"status":"answered","answer":"RAG uses evidence. [S1]",'
        '"citation_ids":["S1"]}',
        "stop",
    )


def make_service(
    results: list[SearchResult],
    model_result: ModelOutput | Exception,
) -> tuple[AnswerService, FakeIndex, FakeLanguageModel]:
    index = FakeIndex(results)
    model = FakeLanguageModel(model_result)
    service = AnswerService(SearchService(index), model)
    return service, index, model


def test_answer_forwards_default_and_explicit_search_options() -> None:
    service, index, _ = make_service([result()], answered_output())

    service.answer("How does RAG work?")
    service.answer("How?", limit=3, mode=SearchMode.SEMANTIC)

    assert index.calls == [
        ("How does RAG work?", 5, SearchMode.HYBRID),
        ("How?", 3, SearchMode.SEMANTIC),
    ]


@pytest.mark.parametrize(
    ("question", "limit", "message"),
    [
        ("", 5, "question"),
        ("   ", 5, "question"),
        ("question", 0, "between 1 and 10"),
        ("question", 11, "between 1 and 10"),
    ],
)
def test_answer_rejects_invalid_input_before_dependencies(
    question: str,
    limit: int,
    message: str,
) -> None:
    service, index, model = make_service([result()], answered_output())

    with pytest.raises(InvalidQueryError, match=message):
        service.answer(question, limit=limit)

    assert index.calls == []
    assert model.calls == []


def test_empty_retrieval_returns_local_refusal_without_model_call() -> None:
    service, _, model = make_service([], answered_output())

    answer = service.answer("unknown question")

    assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.answer == "The knowledge base does not contain enough evidence."
    assert answer.citation_ids == ()
    assert answer.evidence == ()
    assert answer.model_id is None
    assert model.calls == []


def test_non_empty_retrieval_builds_prompt_and_returns_validated_answer() -> None:
    search_result = result()
    service, _, model = make_service([search_result], answered_output())

    answer = service.answer("How does RAG work?")

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.citation_ids == ("S1",)
    assert answer.evidence[0].chunk_id == search_result.chunk.chunk_id
    assert answer.model_id == "fake-provider-model"
    assert len(model.calls) == 1
    system, user = model.calls[0]
    assert "untrusted data" in system.content
    assert "RAG uses evidence." in user.content


@pytest.mark.parametrize(
    ("model_result", "error_type", "message"),
    [
        (LanguageModelError("provider failed"), LanguageModelError, "provider failed"),
        (
            ModelOutput("model", "not-json", "stop"),
            InvalidAnswerSchemaError,
            "valid JSON",
        ),
        (
            ModelOutput(
                "model",
                '{"status":"answered","answer":"wrong [S99]","citation_ids":["S99"]}',
                "stop",
            ),
            InvalidCitationError,
            "unknown citation",
        ),
    ],
)
def test_answer_does_not_hide_model_or_validation_errors(
    model_result: ModelOutput | Exception,
    error_type: type[Exception],
    message: str,
) -> None:
    service, _, model = make_service([result()], model_result)

    with pytest.raises(error_type, match=message):
        service.answer("question")
    assert len(model.calls) == 1

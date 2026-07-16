"""Tests for conservative agent application orchestration."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from personal_knowledge_agent.application.agent import AgentService
from personal_knowledge_agent.application.answer import AnswerService
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.errors import InvalidQueryError, LanguageModelError
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    Chunk,
    DirectIntent,
    Document,
    ModelOutput,
    RouteAction,
    RouteDecision,
    RouterFallbackReason,
    RouteSource,
    SearchMode,
    SearchResult,
)
from tests.helpers import FakeLanguageModel


class FakeRouter:
    def __init__(self, decision: RouteDecision) -> None:
        self._decision = decision
        self.calls: list[str] = []

    def route(self, question: str) -> RouteDecision:
        self.calls.append(question)
        return self._decision


class FakeIndex:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
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
        return self._results


def make_answer_service(
    model_result: ModelOutput | Exception | None = None,
) -> tuple[AnswerService, FakeIndex, FakeLanguageModel]:
    chunk = Chunk("chunk", "document", 0, "RAG uses evidence.", ("RAG",))
    index = FakeIndex([SearchResult(chunk, "/notes/rag.md", 0.9)])
    default_output = ModelOutput(
        "answer-model",
        '{"status":"answered","answer":"RAG uses evidence. [S1]",'
        '"citation_ids":["S1"]}',
        "stop",
    )
    model = FakeLanguageModel(model_result or default_output)
    return AnswerService(SearchService(index), model), index, model


@pytest.mark.parametrize(
    ("question", "intent", "expected"),
    [
        (
            "hello",
            DirectIntent.GREETING,
            "Hi. I can search your local knowledge base and answer with citations "
            "when needed.",
        ),
        (
            "你好",
            DirectIntent.GREETING,
            "你好。我可以在需要时检索你的本地知识库，并用引用回答问题。",
        ),
        (
            "what can you do",
            DirectIntent.CAPABILITIES,
            "I can search your local knowledge base, answer from retrieved evidence, "
            "and show cited sources.",
        ),
        (
            "你能做什么",
            DirectIntent.CAPABILITIES,
            "我可以检索你的本地知识库、基于证据回答问题，并显示引用来源。",
        ),
        ("thanks", DirectIntent.THANKS, "You're welcome."),
        ("谢谢", DirectIntent.THANKS, "不客气。"),
    ],
)
def test_direct_route_returns_local_language_aware_template_without_answer_call(
    question: str,
    intent: DirectIntent,
    expected: str,
) -> None:
    route = RouteDecision(RouteAction.DIRECT, intent, RouteSource.RULE)
    router = FakeRouter(route)
    answer_service, index, model = make_answer_service()

    response = AgentService(router, answer_service).respond(question)

    assert response.route == route
    assert response.direct_answer == expected
    assert response.grounded_answer is None
    assert router.calls == [question]
    assert index.calls == []
    assert model.calls == []


@pytest.mark.parametrize(
    "route",
    [
        RouteDecision(RouteAction.RETRIEVE, None, RouteSource.LLM),
        RouteDecision(
            RouteAction.RETRIEVE,
            None,
            RouteSource.FALLBACK,
            RouterFallbackReason.INVALID_OUTPUT,
        ),
    ],
)
def test_retrieve_and_fallback_routes_delegate_to_grounded_answer(
    route: RouteDecision,
) -> None:
    router = FakeRouter(route)
    answer_service, index, model = make_answer_service()

    response = AgentService(router, answer_service).respond(
        "How does RAG work?",
        limit=3,
        mode=SearchMode.SEMANTIC,
    )

    assert response.route == route
    assert response.direct_answer is None
    assert response.grounded_answer is not None
    assert response.grounded_answer.status is AnswerStatus.ANSWERED
    assert index.calls == [("How does RAG work?", 3, SearchMode.SEMANTIC)]
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    ("question", "limit", "message"),
    [
        ("", 5, "question"),
        ("   ", 5, "question"),
        ("question", 0, "between 1 and 10"),
        ("question", 11, "between 1 and 10"),
    ],
)
def test_invalid_input_is_rejected_before_router_or_answer_dependencies(
    question: str,
    limit: int,
    message: str,
) -> None:
    route = RouteDecision(RouteAction.RETRIEVE, None, RouteSource.LLM)
    router = FakeRouter(route)
    answer_service, index, model = make_answer_service()

    with pytest.raises(InvalidQueryError, match=message):
        AgentService(router, answer_service).respond(question, limit=limit)

    assert router.calls == []
    assert index.calls == []
    assert model.calls == []


def test_route_observer_runs_before_grounded_answer_failure() -> None:
    route = RouteDecision(RouteAction.RETRIEVE, None, RouteSource.LLM)
    router = FakeRouter(route)
    answer_service, _, _ = make_answer_service(LanguageModelError("answer failed"))
    observed: list[RouteDecision] = []

    with pytest.raises(LanguageModelError, match="answer failed"):
        AgentService(router, answer_service).respond(
            "How does RAG work?",
            on_route=observed.append,
        )

    assert observed == [route]

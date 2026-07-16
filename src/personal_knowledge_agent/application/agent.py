"""Application service for conservative agent routing and response production."""

from __future__ import annotations

from personal_knowledge_agent.application.answer import AnswerService
from personal_knowledge_agent.domain.errors import InvalidQueryError
from personal_knowledge_agent.domain.models import (
    AgentResponse,
    DirectIntent,
    RouteAction,
    SearchMode,
)
from personal_knowledge_agent.domain.protocols import QuestionRouter

_DIRECT_ANSWERS = {
    (
        DirectIntent.GREETING,
        False,
    ): (
        "Hi. I can search your local knowledge base and answer with citations "
        "when needed."
    ),
    (
        DirectIntent.GREETING,
        True,
    ): "你好。我可以在需要时检索你的本地知识库，并用引用回答问题。",
    (
        DirectIntent.CAPABILITIES,
        False,
    ): (
        "I can search your local knowledge base, answer from retrieved evidence, "
        "and show cited sources."
    ),
    (
        DirectIntent.CAPABILITIES,
        True,
    ): "我可以检索你的本地知识库、基于证据回答问题，并显示引用来源。",
    (DirectIntent.THANKS, False): "You're welcome.",
    (DirectIntent.THANKS, True): "不客气。",
}


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def render_direct_answer(intent: DirectIntent, question: str) -> str:
    """Render one deterministic response in the question's language family."""
    return _DIRECT_ANSWERS[(intent, _contains_chinese(question))]


class AgentService:
    """Coordinate route selection with direct or grounded answer production."""

    def __init__(
        self,
        router: QuestionRouter,
        answer_service: AnswerService,
    ) -> None:
        self._router = router
        self._answer_service = answer_service

    def respond(
        self,
        question: str,
        *,
        limit: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> AgentResponse:
        """Choose a route and return exactly one corresponding answer."""
        if not question.strip():
            raise InvalidQueryError("Agent question must not be blank")
        if not 1 <= limit <= 10:
            raise InvalidQueryError("Agent evidence limit must be between 1 and 10")

        route = self._router.route(question)
        if route.action is RouteAction.DIRECT:
            assert route.direct_intent is not None
            return AgentResponse(
                route,
                render_direct_answer(route.direct_intent, question),
                None,
            )

        grounded_answer = self._answer_service.answer(
            question,
            limit=limit,
            mode=mode,
        )
        return AgentResponse(route, None, grounded_answer)

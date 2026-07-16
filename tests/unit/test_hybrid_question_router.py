"""Tests for the rule-first, fail-closed hybrid router."""

import pytest

from personal_knowledge_agent.domain.errors import InvalidQueryError, LanguageModelError
from personal_knowledge_agent.domain.models import (
    DirectIntent,
    ModelOutput,
    RouteAction,
    RouterFallbackReason,
    RouteSource,
)
from personal_knowledge_agent.infrastructure.hybrid_question_router import (
    HybridQuestionRouter,
)
from tests.helpers import FakeLanguageModel


def test_rule_direct_short_circuits_model_call() -> None:
    model = FakeLanguageModel(RuntimeError("must not be called"))

    decision = HybridQuestionRouter(model).route("Hello!")

    assert decision.action is RouteAction.DIRECT
    assert decision.direct_intent is DirectIntent.GREETING
    assert decision.source is RouteSource.RULE
    assert model.calls == []


@pytest.mark.parametrize(
    ("content", "action", "intent"),
    [
        (
            '{"action":"retrieve","direct_intent":null}',
            RouteAction.RETRIEVE,
            None,
        ),
        (
            '{"action":"direct","direct_intent":"capabilities"}',
            RouteAction.DIRECT,
            DirectIntent.CAPABILITIES,
        ),
    ],
)
def test_unmatched_question_uses_valid_llm_decision(
    content: str,
    action: RouteAction,
    intent: DirectIntent | None,
) -> None:
    model = FakeLanguageModel(ModelOutput("router", content, "stop"))

    decision = HybridQuestionRouter(model).route("Tell me how you can help")

    assert decision.action is action
    assert decision.direct_intent is intent
    assert decision.source is RouteSource.LLM
    assert decision.fallback_reason is None
    assert len(model.calls) == 1
    assert model.calls[0][1].content == "QUESTION:\nTell me how you can help"


def test_provider_error_falls_back_to_retrieval() -> None:
    model = FakeLanguageModel(LanguageModelError("secret provider details"))

    decision = HybridQuestionRouter(model).route("How does RAG work?")

    assert decision.action is RouteAction.RETRIEVE
    assert decision.direct_intent is None
    assert decision.source is RouteSource.FALLBACK
    assert decision.fallback_reason is RouterFallbackReason.PROVIDER_ERROR


def test_invalid_output_falls_back_to_retrieval() -> None:
    model = FakeLanguageModel(ModelOutput("router", "not JSON", "stop"))

    decision = HybridQuestionRouter(model).route("How does RAG work?")

    assert decision.action is RouteAction.RETRIEVE
    assert decision.direct_intent is None
    assert decision.source is RouteSource.FALLBACK
    assert decision.fallback_reason is RouterFallbackReason.INVALID_OUTPUT


def test_blank_question_is_rejected_before_model_call() -> None:
    model = FakeLanguageModel(RuntimeError("must not be called"))

    with pytest.raises(InvalidQueryError, match="question"):
        HybridQuestionRouter(model).route("   ")

    assert model.calls == []


def test_unexpected_model_error_is_not_hidden_as_fallback() -> None:
    model = FakeLanguageModel(RuntimeError("programming failure"))

    with pytest.raises(RuntimeError, match="programming failure"):
        HybridQuestionRouter(model).route("How does RAG work?")

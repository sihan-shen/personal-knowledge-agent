"""Rule-first route selection with conservative model fallbacks."""

from __future__ import annotations

from personal_knowledge_agent.domain.errors import (
    InvalidRouteSchemaError,
    LanguageModelError,
)
from personal_knowledge_agent.domain.models import (
    RouteAction,
    RouteDecision,
    RouterFallbackReason,
    RouteSource,
)
from personal_knowledge_agent.domain.protocols import LanguageModel
from personal_knowledge_agent.infrastructure.direct_intent_rules import (
    match_direct_intent,
)
from personal_knowledge_agent.infrastructure.route_prompt_builder import (
    build_route_messages,
)
from personal_knowledge_agent.infrastructure.structured_route_parser import (
    parse_route_decision,
)


class HybridQuestionRouter:
    """Use exact safe rules before asking a model for a strict route decision."""

    def __init__(self, language_model: LanguageModel) -> None:
        self._language_model = language_model

    def route(self, question: str) -> RouteDecision:
        """Return a direct decision or fail closed to retrieval."""
        intent = match_direct_intent(question)
        if intent is not None:
            return RouteDecision(RouteAction.DIRECT, intent, RouteSource.RULE)

        messages = build_route_messages(question)
        try:
            output = self._language_model.generate(messages)
            return parse_route_decision(output)
        except LanguageModelError:
            return RouteDecision(
                RouteAction.RETRIEVE,
                None,
                RouteSource.FALLBACK,
                RouterFallbackReason.PROVIDER_ERROR,
            )
        except InvalidRouteSchemaError:
            return RouteDecision(
                RouteAction.RETRIEVE,
                None,
                RouteSource.FALLBACK,
                RouterFallbackReason.INVALID_OUTPUT,
            )

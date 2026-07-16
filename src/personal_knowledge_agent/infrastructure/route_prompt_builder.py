"""Provider-neutral messages for conservative route classification."""

from __future__ import annotations

from personal_knowledge_agent.domain.errors import InvalidQueryError
from personal_knowledge_agent.domain.models import ChatMessage, MessageRole

_ROUTE_POLICY = """You are a conservative router for a personal knowledge agent.
Return exactly one JSON object with exactly these fields:
{"action":"direct|retrieve","direct_intent":"greeting|capabilities|thanks|null"}

Use direct only when the complete request is a greeting, a question about the
assistant's capabilities, or thanks. For every factual, learning,
personal-knowledge, mixed, ambiguous, unsupported, or uncertain request, you must
use retrieve. You must use retrieve in all of those cases. A greeting combined with
another request is mixed and must retrieve.

For retrieve, direct_intent must be null. For direct, direct_intent must be one of
greeting, capabilities, or thanks. Output JSON without Markdown fences, commentary,
or additional fields.

Examples:
User: hello
Assistant: {"action":"direct","direct_intent":"greeting"}
User: hello, explain RAG
Assistant: {"action":"retrieve","direct_intent":null}
"""


def build_route_messages(question: str) -> tuple[ChatMessage, ChatMessage]:
    """Build the policy and question messages for one route decision."""
    if not question.strip():
        raise InvalidQueryError("Router question must not be blank")
    return (
        ChatMessage(MessageRole.SYSTEM, _ROUTE_POLICY),
        ChatMessage(MessageRole.USER, f"QUESTION:\n{question}"),
    )

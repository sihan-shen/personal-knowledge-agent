"""Privacy-minimized messages for structured memory extraction."""

from __future__ import annotations

from personal_knowledge_agent.domain.errors import InvalidMemoryError
from personal_knowledge_agent.domain.memory import MAX_MEMORY_EXTRACTION_INPUT_LENGTH
from personal_knowledge_agent.domain.models import ChatMessage, MessageRole

_MEMORY_EXTRACTION_POLICY = (
    "You extract durable user memories from one current user message.\n"
    "Return exactly one JSON object in this exact shape:\n"
    '{"candidates":[{"type":"goal","key":"goal.ai_engineer_internship",'
    '"content":"The user aims to complete an AI Engineer internship project."}]}\n\n'
    "Each candidate must contain exactly type, key, and content. The only allowed "
    "types\nare preference, goal, project, and learning_progress. A key must match\n"
    "[a-z][a-z0-9_.-]{0,63}. Return at most 3 candidates. An empty list is valid.\n\n"
    "Extract only information likely to remain useful across future conversations.\n"
    "Durable preferences, goals, projects, and learning progress qualify. Temporary\n"
    "requests, passing moods, one-time logistics, and facts that are only relevant "
    "to\nthe current exchange do not qualify. Preserve the user's language when "
    "practical.\n\n"
    "Do not extract passwords, API keys, access tokens, or private keys.\n"
    "Do not extract identity documents, government identifiers, medical diagnoses, "
    "treatment, health records, bank accounts, payment cards, income, debt, other "
    "financial account records, legal cases, or privileged legal details.\n"
    "When any candidate would "
    "include such sensitive information, omit that candidate.\n\n"
    "Output JSON without Markdown fences, extra fields, commentary, or other text.\n"
)


def build_memory_extraction_messages(
    message: str,
) -> tuple[ChatMessage, ChatMessage]:
    """Build the policy and current-message payload for memory extraction."""
    if not message.strip():
        raise InvalidMemoryError("Memory extraction message must not be blank")
    if len(message) > MAX_MEMORY_EXTRACTION_INPUT_LENGTH:
        raise InvalidMemoryError(
            "Memory extraction message must not exceed 4,000 code points"
        )
    return (
        ChatMessage(MessageRole.SYSTEM, _MEMORY_EXTRACTION_POLICY),
        ChatMessage(MessageRole.USER, f"CURRENT USER MESSAGE:\n{message}"),
    )

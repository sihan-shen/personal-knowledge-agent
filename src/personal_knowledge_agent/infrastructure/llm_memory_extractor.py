"""Language-model adapter for strict structured memory extraction."""

from __future__ import annotations

from personal_knowledge_agent.domain.memory import MemoryProposal
from personal_knowledge_agent.domain.protocols import LanguageModel
from personal_knowledge_agent.infrastructure.memory_prompt_builder import (
    build_memory_extraction_messages,
)
from personal_knowledge_agent.infrastructure.structured_memory_parser import (
    parse_memory_proposals,
)


class LLMMemoryExtractor:
    """Extract validated memory proposals from one current user message."""

    def __init__(self, language_model: LanguageModel) -> None:
        self._language_model = language_model

    def extract(self, message: str) -> tuple[MemoryProposal, ...]:
        """Generate and parse proposals while preserving boundary errors."""
        messages = build_memory_extraction_messages(message)
        output = self._language_model.generate(messages)
        return parse_memory_proposals(output)

"""Application service for document and memory-aware answers."""

from __future__ import annotations

from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.errors import (
    InvalidQueryError,
    KnowledgeBaseEmptyError,
)
from personal_knowledge_agent.domain.memory import MAX_RECALLED_MEMORIES
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    GroundedAnswer,
    MemoryAwareAnswerResult,
    SearchMode,
)
from personal_knowledge_agent.domain.protocols import LanguageModel, MemoryRecaller
from personal_knowledge_agent.infrastructure.evidence_builder import build_evidence
from personal_knowledge_agent.infrastructure.memory_aware_prompt_builder import (
    build_memory_aware_messages,
)
from personal_knowledge_agent.infrastructure.structured_contextual_answer_parser import (  # noqa: E501
    parse_contextual_answer,
)

_INSUFFICIENT_EVIDENCE = "The knowledge base does not contain enough evidence."


class MemoryAwareAnswerService:
    """Coordinate document retrieval, memory recall, prompting, and validation."""

    def __init__(
        self,
        search_service: SearchService,
        memory_recaller: MemoryRecaller,
        language_model: LanguageModel,
    ) -> None:
        self._search_service = search_service
        self._memory_recaller = memory_recaller
        self._language_model = language_model

    def answer(
        self,
        question: str,
        *,
        limit: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> MemoryAwareAnswerResult:
        """Answer from documents, confirmed memories, or both."""
        if not question.strip():
            raise InvalidQueryError("Answer question must not be blank")
        if not 1 <= limit <= 10:
            raise InvalidQueryError("Answer evidence limit must be between 1 and 10")

        try:
            results = self._search_service.search(question, limit=limit, mode=mode)
        except KnowledgeBaseEmptyError:
            results = []
        recall = self._memory_recaller.recall(
            question,
            limit=MAX_RECALLED_MEMORIES,
        )
        evidence = build_evidence(results)
        if not evidence and not recall.memories:
            return MemoryAwareAnswerResult(
                GroundedAnswer(
                    AnswerStatus.INSUFFICIENT_EVIDENCE,
                    _INSUFFICIENT_EVIDENCE,
                    (),
                    (),
                    None,
                ),
                recall,
            )
        messages = build_memory_aware_messages(question, evidence, recall.memories)
        output = self._language_model.generate(messages)
        answer = parse_contextual_answer(output, evidence, recall.memories)
        return MemoryAwareAnswerResult(answer, recall)

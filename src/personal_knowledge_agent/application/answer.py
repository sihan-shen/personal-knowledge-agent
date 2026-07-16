"""Application service for evidence-grounded answer generation."""

from __future__ import annotations

from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.errors import InvalidQueryError
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    GroundedAnswer,
    SearchMode,
)
from personal_knowledge_agent.domain.protocols import LanguageModel
from personal_knowledge_agent.infrastructure.evidence_builder import build_evidence
from personal_knowledge_agent.infrastructure.grounded_prompt_builder import (
    build_grounded_messages,
)
from personal_knowledge_agent.infrastructure.structured_answer_parser import (
    parse_grounded_answer,
)

_INSUFFICIENT_EVIDENCE = "The knowledge base does not contain enough evidence."


class AnswerService:
    """Coordinate retrieval, grounded prompting, generation, and validation."""

    def __init__(
        self,
        search_service: SearchService,
        language_model: LanguageModel,
    ) -> None:
        self._search_service = search_service
        self._language_model = language_model

    def answer(
        self,
        question: str,
        *,
        limit: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> GroundedAnswer:
        """Answer from retrieved evidence or return a local evidence refusal."""
        if not question.strip():
            raise InvalidQueryError("Answer question must not be blank")
        if not 1 <= limit <= 10:
            raise InvalidQueryError("Answer evidence limit must be between 1 and 10")

        results = self._search_service.search(question, limit=limit, mode=mode)
        if not results:
            return GroundedAnswer(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer=_INSUFFICIENT_EVIDENCE,
                citation_ids=(),
                evidence=(),
                model_id=None,
            )

        evidence = build_evidence(results)
        messages = build_grounded_messages(question, evidence)
        output = self._language_model.generate(messages)
        return parse_grounded_answer(output, evidence)

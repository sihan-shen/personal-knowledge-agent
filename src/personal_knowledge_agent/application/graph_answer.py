"""Application service for explicit graph-grounded answering."""

from __future__ import annotations

from collections.abc import Callable

from personal_knowledge_agent.domain.errors import InvalidGraphError
from personal_knowledge_agent.domain.graph_answer import GraphGroundedAnswer
from personal_knowledge_agent.domain.models import AnswerStatus
from personal_knowledge_agent.domain.protocols import KnowledgeGraphStore, LanguageModel
from personal_knowledge_agent.infrastructure.graph_answer_prompt_builder import (
    build_graph_answer_messages,
)
from personal_knowledge_agent.infrastructure.graph_evidence_builder import (
    build_graph_evidence,
)
from personal_knowledge_agent.infrastructure.structured_graph_answer_parser import (
    parse_graph_grounded_answer,
)

_LOCAL_REFUSAL = "The knowledge graph does not contain enough relationship evidence."


class GraphAnswerService:
    def __init__(
        self,
        store: KnowledgeGraphStore,
        language_model_factory: Callable[[], LanguageModel],
    ) -> None:
        self._store = store
        self._language_model_factory = language_model_factory

    def answer(self, question: str, *, limit: int = 3) -> GraphGroundedAnswer:
        if not question.strip():
            raise InvalidGraphError("Graph answer question must not be blank")
        if not 1 <= limit <= 5:
            raise InvalidGraphError("Graph answer seed limit must be between 1 and 5")

        seeds = self._store.search_entities(question, limit=limit)
        neighborhoods = tuple(
            self._store.get_neighborhood(seed.entity_id) for seed in seeds
        )
        evidence = build_graph_evidence(neighborhoods)
        if not evidence:
            return GraphGroundedAnswer(
                AnswerStatus.INSUFFICIENT_EVIDENCE,
                _LOCAL_REFUSAL,
                (),
                (),
                None,
            )

        messages = build_graph_answer_messages(question, evidence)
        output = self._language_model_factory().generate(messages)
        return parse_graph_grounded_answer(output, evidence)

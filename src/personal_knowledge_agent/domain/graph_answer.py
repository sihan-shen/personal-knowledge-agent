"""Immutable values for provenance-grounded graph answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from personal_knowledge_agent.domain.graph import (
    EntityType,
    GraphProvenance,
    RelationType,
)
from personal_knowledge_agent.domain.models import AnswerStatus

_GRAPH_CITATION_ID = re.compile(r"G(?:[1-9]|1[0-2])$")


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


@dataclass(frozen=True, slots=True)
class GraphEvidence:
    """One relation paired with exactly one local provenance record."""

    citation_id: str
    source_entity_type: EntityType
    source_entity_name: str
    relation_type: RelationType
    target_entity_type: EntityType
    target_entity_name: str
    provenance: GraphProvenance

    def __post_init__(self) -> None:
        if not _GRAPH_CITATION_ID.fullmatch(self.citation_id):
            raise ValueError("citation_id must be between G1 and G12")
        _require_text(self.source_entity_name, "source_entity_name")
        _require_text(self.target_entity_name, "target_entity_name")


@dataclass(frozen=True, slots=True)
class GraphGroundedAnswer:
    """A validated graph answer containing only its cited evidence."""

    status: AnswerStatus
    answer: str
    citation_ids: tuple[str, ...]
    evidence: tuple[GraphEvidence, ...]
    model_id: str | None

    def __post_init__(self) -> None:
        _require_text(self.answer, "answer")
        if self.model_id is not None:
            _require_text(self.model_id, "model_id")
        evidence_ids = tuple(item.citation_id for item in self.evidence)
        if len(set(self.citation_ids)) != len(self.citation_ids):
            raise ValueError("citation_ids must be unique")
        if self.status is AnswerStatus.ANSWERED:
            if not self.citation_ids:
                raise ValueError("answered graph result requires citations")
            if evidence_ids != self.citation_ids:
                raise ValueError("evidence must match citation_ids in order")
            if self.model_id is None:
                raise ValueError("answered graph result requires model_id")
        elif self.citation_ids or self.evidence:
            raise ValueError("insufficient graph result must not contain evidence")

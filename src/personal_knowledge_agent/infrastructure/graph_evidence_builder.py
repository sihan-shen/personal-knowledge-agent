"""Build deterministic provenance-level evidence from graph neighborhoods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from personal_knowledge_agent.domain.graph import (
    EntityNeighborhood,
    EntityType,
    GraphNeighbor,
    GraphProvenance,
    RelationType,
    make_relation_id,
)
from personal_knowledge_agent.domain.graph_answer import GraphEvidence

MAX_GRAPH_ANSWER_EVIDENCE = 12


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One traversed relation paired with one exact provenance record."""

    seed_rank: int
    direction_rank: int
    source_entity_id: str
    source_entity_type: EntityType
    source_entity_name: str
    target_entity_id: str
    target_entity_type: EntityType
    target_entity_name: str
    neighbor_display_name: str
    neighbor_entity_id: str
    relation_type: RelationType
    provenance: GraphProvenance


def build_graph_evidence(
    neighborhoods: Sequence[EntityNeighborhood],
) -> tuple[GraphEvidence, ...]:
    """Expand ranked one-hop neighborhoods into bounded cited graph evidence."""
    candidates = [
        candidate
        for seed_rank, neighborhood in enumerate(neighborhoods)
        for candidate in _neighborhood_candidates(seed_rank, neighborhood)
    ]
    candidates.sort(key=_candidate_sort_key)

    evidence: list[GraphEvidence] = []
    seen: set[tuple[str, str, str, str]] = set()
    for candidate in candidates:
        relation_id = make_relation_id(
            candidate.relation_type,
            candidate.source_entity_id,
            candidate.target_entity_id,
        )
        provenance = candidate.provenance
        dedupe_key = (
            relation_id,
            provenance.source,
            provenance.chunk_id,
            provenance.evidence_quote,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        evidence.append(
            GraphEvidence(
                citation_id=f"G{len(evidence) + 1}",
                source_entity_type=candidate.source_entity_type,
                source_entity_name=candidate.source_entity_name,
                relation_type=candidate.relation_type,
                target_entity_type=candidate.target_entity_type,
                target_entity_name=candidate.target_entity_name,
                provenance=provenance,
            )
        )
        if len(evidence) == MAX_GRAPH_ANSWER_EVIDENCE:
            break
    return tuple(evidence)


def _neighborhood_candidates(
    seed_rank: int,
    neighborhood: EntityNeighborhood,
) -> tuple[_Candidate, ...]:
    candidates: list[_Candidate] = []
    for neighbor in neighborhood.outgoing:
        candidates.extend(
            _neighbor_candidates(
                seed_rank,
                0,
                neighborhood.entity.entity_id,
                neighborhood.entity.entity_type,
                neighborhood.entity.display_name,
                neighbor,
                outgoing=True,
            )
        )
    for neighbor in neighborhood.incoming:
        candidates.extend(
            _neighbor_candidates(
                seed_rank,
                1,
                neighborhood.entity.entity_id,
                neighborhood.entity.entity_type,
                neighborhood.entity.display_name,
                neighbor,
                outgoing=False,
            )
        )
    return tuple(candidates)


def _neighbor_candidates(
    seed_rank: int,
    direction_rank: int,
    selected_entity_id: str,
    selected_entity_type: EntityType,
    selected_entity_name: str,
    neighbor: GraphNeighbor,
    *,
    outgoing: bool,
) -> tuple[_Candidate, ...]:
    if outgoing:
        source_entity_id = selected_entity_id
        source_entity_type = selected_entity_type
        source_entity_name = selected_entity_name
        target_entity_id = neighbor.entity_id
        target_entity_type = neighbor.entity_type
        target_entity_name = neighbor.display_name
    else:
        source_entity_id = neighbor.entity_id
        source_entity_type = neighbor.entity_type
        source_entity_name = neighbor.display_name
        target_entity_id = selected_entity_id
        target_entity_type = selected_entity_type
        target_entity_name = selected_entity_name
    return tuple(
        _Candidate(
            seed_rank=seed_rank,
            direction_rank=direction_rank,
            source_entity_id=source_entity_id,
            source_entity_type=source_entity_type,
            source_entity_name=source_entity_name,
            target_entity_id=target_entity_id,
            target_entity_type=target_entity_type,
            target_entity_name=target_entity_name,
            neighbor_display_name=neighbor.display_name,
            neighbor_entity_id=neighbor.entity_id,
            relation_type=neighbor.relation_type,
            provenance=provenance,
        )
        for provenance in neighbor.provenance
    )


def _candidate_sort_key(candidate: _Candidate) -> tuple[object, ...]:
    return (
        candidate.seed_rank,
        candidate.direction_rank,
        candidate.relation_type.value,
        candidate.neighbor_display_name,
        candidate.neighbor_entity_id,
        candidate.provenance.source,
        candidate.provenance.chunk_position,
        candidate.provenance.chunk_id,
        candidate.provenance.evidence_quote,
    )

"""Domain values and identity helpers for the knowledge graph.

This module defines the controlled ontology, immutable extraction/build/query
values, graph limits, and deterministic identity helpers. It is free of
infrastructure dependencies.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from personal_knowledge_agent.domain.models import Chunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_GRAPH_ENTITIES_PER_CHUNK = 15
MAX_GRAPH_RELATIONS_PER_CHUNK = 20
MAX_GRAPH_ENTITY_NAME_LENGTH = 120
MAX_GRAPH_EVIDENCE_QUOTE_LENGTH = 300
MAX_GRAPH_SEARCH_LIMIT = 20


# ---------------------------------------------------------------------------
# Controlled ontology
# ---------------------------------------------------------------------------


class EntityType(StrEnum):
    """Controlled types for knowledge-graph entities."""

    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    METHOD = "method"
    PROJECT = "project"
    RESOURCE = "resource"


class RelationType(StrEnum):
    """Controlled types for knowledge-graph relations."""

    IS_A = "is_a"
    PART_OF = "part_of"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    RELATED_TO = "related_to"


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name for identity comparison.

    Applies NFKC normalization, full casefolding, and whitespace collapse,
    then strips outer whitespace.
    """
    nfkc = unicodedata.normalize("NFKC", name)
    casefolded = nfkc.casefold()
    collapsed = _collapse_whitespace(casefolded)
    return collapsed.strip()


def _collapse_whitespace(text: str) -> str:
    """Collapse any contiguous whitespace into a single space."""
    return re.sub(r"\s+", " ", text)


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _make_id(prefix: str, *parts: str) -> str:
    """Create a deterministic hex identifier from null-separated parts."""
    raw = "\0".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def make_entity_id(entity_type: EntityType, name: str) -> str:
    """Create a deterministic entity ID from type and normalized name."""
    normalized = normalize_entity_name(name)
    return _make_id("ent", entity_type.value, normalized)


def make_relation_id(
    relation_type: RelationType,
    source_entity_id: str,
    target_entity_id: str,
) -> str:
    """Create a deterministic relation ID.

    For symmetric RELATION_TO, canonicalize by sorting endpoint IDs.
    For all other directed types, use the supplied order.
    """
    if source_entity_id == target_entity_id:
        raise ValueError("self-loop relations are not allowed")
    if relation_type is RelationType.RELATED_TO:
        eid_a, eid_b = sorted((source_entity_id, target_entity_id))
    else:
        eid_a, eid_b = source_entity_id, target_entity_id
    return _make_id("rel", relation_type.value, eid_a, eid_b)


# ---------------------------------------------------------------------------
# Immutable domain records
# ---------------------------------------------------------------------------


class GraphEntity:
    """One typed entity in the knowledge graph."""

    __slots__ = ("entity_id", "entity_type", "display_name")

    def __init__(
        self,
        entity_id: str,
        entity_type: EntityType,
        display_name: str,
    ) -> None:
        _require_text(display_name, "name")
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.display_name = display_name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphEntity):
            return NotImplemented
        return (
            self.entity_id == other.entity_id
            and self.entity_type is other.entity_type
            and self.display_name == other.display_name
        )

    def __hash__(self) -> int:
        return hash((self.entity_id, self.entity_type, self.display_name))

    def __repr__(self) -> str:
        return (
            f"GraphEntity(entity_id={self.entity_id!r}, "
            f"entity_type={self.entity_type.value!r}, "
            f"display_name={self.display_name!r})"
        )


class GraphRelation:
    """One typed relation between two entities."""

    __slots__ = ("relation_id", "relation_type", "source_entity_id", "target_entity_id")

    def __init__(
        self,
        relation_id: str,
        relation_type: RelationType,
        source_entity_id: str,
        target_entity_id: str,
    ) -> None:
        self.relation_id = relation_id
        self.relation_type = relation_type
        self.source_entity_id = source_entity_id
        self.target_entity_id = target_entity_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphRelation):
            return NotImplemented
        return (
            self.relation_id == other.relation_id
            and self.relation_type is other.relation_type
            and self.source_entity_id == other.source_entity_id
            and self.target_entity_id == other.target_entity_id
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.relation_id,
                self.relation_type,
                self.source_entity_id,
                self.target_entity_id,
            )
        )

    def __repr__(self) -> str:
        return (
            f"GraphRelation(relation_id={self.relation_id!r}, "
            f"relation_type={self.relation_type.value!r}, "
            f"source_entity_id={self.source_entity_id!r}, "
            f"target_entity_id={self.target_entity_id!r})"
        )


class GraphRelationEvidence:
    """One relation paired with its exact supporting quote."""

    __slots__ = ("relation_id", "evidence_quote")

    def __init__(self, relation_id: str, evidence_quote: str) -> None:
        _require_text(evidence_quote, "evidence_quote")
        self.relation_id = relation_id
        self.evidence_quote = evidence_quote

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphRelationEvidence):
            return NotImplemented
        return (
            self.relation_id == other.relation_id
            and self.evidence_quote == other.evidence_quote
        )

    def __hash__(self) -> int:
        return hash((self.relation_id, self.evidence_quote))

    def __repr__(self) -> str:
        return f"GraphRelationEvidence(relation_id={self.relation_id!r})"


class GraphChunkExtraction:
    """One chunk's full validated extraction result."""

    __slots__ = ("chunk", "entities", "relation_evidence")

    def __init__(
        self,
        chunk: Chunk,
        entities: Sequence[GraphEntity],
        relation_evidence: Sequence[GraphRelationEvidence],
    ) -> None:
        entities_tuple = tuple(entities)
        relation_evidence_tuple = tuple(relation_evidence)
        if len(entities_tuple) > MAX_GRAPH_ENTITIES_PER_CHUNK:
            raise ValueError(
                f"at most {MAX_GRAPH_ENTITIES_PER_CHUNK} entities per chunk, "
                f"got {len(entities_tuple)}"
            )
        if len(relation_evidence_tuple) > MAX_GRAPH_RELATIONS_PER_CHUNK:
            raise ValueError(
                f"at most {MAX_GRAPH_RELATIONS_PER_CHUNK} relations per chunk, "
                f"got {len(relation_evidence_tuple)}"
            )
        self.chunk = chunk
        self.entities = entities_tuple
        self.relation_evidence = relation_evidence_tuple

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphChunkExtraction):
            return NotImplemented
        return (
            self.chunk == other.chunk
            and self.entities == other.entities
            and self.relation_evidence == other.relation_evidence
        )

    def __hash__(self) -> int:
        return hash((self.chunk, self.entities, self.relation_evidence))

    def __repr__(self) -> str:
        return (
            f"GraphChunkExtraction(chunk_id={self.chunk.chunk_id!r}, "
            f"entities={len(self.entities)}, "
            f"relations={len(self.relation_evidence)})"
        )


class GraphSourceDocument:
    """One indexed source document loaded for graph building.

    Contains the document metadata and all its current chunks in order.
    """

    __slots__ = ("document_id", "source", "content_hash", "chunks")

    def __init__(
        self,
        document_id: str,
        source: Path,
        content_hash: str,
        chunks: Sequence[Chunk],
    ) -> None:
        _require_text(document_id, "document_id")
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise ValueError(
                f"content_hash must be a 64-character hex string, "
                f"got {len(content_hash)} chars"
            )
        chunks_tuple = tuple(chunks)
        if chunks_tuple:
            for i, chunk in enumerate(chunks_tuple):
                if chunk.document_id != document_id:
                    raise ValueError(
                        f"chunk at position {i} has document_id "
                        f"{chunk.document_id!r}, expected {document_id!r}"
                    )
            positions = [c.position for c in chunks_tuple]
            if not all(
                positions[i] < positions[i + 1] for i in range(len(positions) - 1)
            ):
                raise ValueError("chunk positions must be in strictly ascending order")
        self.document_id = document_id
        self.source = source
        self.content_hash = content_hash
        self.chunks = chunks_tuple

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphSourceDocument):
            return NotImplemented
        return (
            self.document_id == other.document_id
            and self.source == other.source
            and self.content_hash == other.content_hash
            and self.chunks == other.chunks
        )

    def __hash__(self) -> int:
        return hash((self.document_id, self.source, self.content_hash))

    def __repr__(self) -> str:
        return (
            f"GraphSourceDocument(document_id={self.document_id!r}, "
            f"source={str(self.source)!r}, "
            f"chunks={len(self.chunks)})"
        )


class GraphBuildResult:
    """Summary counts from one document graph build."""

    __slots__ = ("source", "chunks", "entities", "relations", "evidence")

    def __init__(
        self,
        source: str,
        chunks: int,
        entities: int,
        relations: int,
        evidence: int,
    ) -> None:
        if chunks < 0:
            raise ValueError(f"chunks must not be negative, got {chunks}")
        if entities < 0:
            raise ValueError(f"entities must not be negative, got {entities}")
        if relations < 0:
            raise ValueError(f"relations must not be negative, got {relations}")
        if evidence < 0:
            raise ValueError(f"evidence must not be negative, got {evidence}")
        self.source = source
        self.chunks = chunks
        self.entities = entities
        self.relations = relations
        self.evidence = evidence

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphBuildResult):
            return NotImplemented
        return (
            self.source == other.source
            and self.chunks == other.chunks
            and self.entities == other.entities
            and self.relations == other.relations
            and self.evidence == other.evidence
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.source,
                self.chunks,
                self.entities,
                self.relations,
                self.evidence,
            )
        )

    def __repr__(self) -> str:
        return (
            f"GraphBuildResult(source={self.source!r}, "
            f"chunks={self.chunks}, entities={self.entities}, "
            f"relations={self.relations}, evidence={self.evidence})"
        )


class GraphProvenance:
    """Complete provenance for one supported relation instance."""

    __slots__ = (
        "source",
        "chunk_id",
        "chunk_position",
        "heading_path",
        "evidence_quote",
    )

    def __init__(
        self,
        source: str,
        chunk_id: str,
        chunk_position: int,
        heading_path: tuple[str, ...],
        evidence_quote: str,
    ) -> None:
        _require_text(source, "source")
        _require_text(chunk_id, "chunk_id")
        _require_text(evidence_quote, "evidence_quote")
        self.source = source
        self.chunk_id = chunk_id
        self.chunk_position = chunk_position
        self.heading_path = heading_path
        self.evidence_quote = evidence_quote

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphProvenance):
            return NotImplemented
        return (
            self.source == other.source
            and self.chunk_id == other.chunk_id
            and self.chunk_position == other.chunk_position
            and self.heading_path == other.heading_path
            and self.evidence_quote == other.evidence_quote
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.source,
                self.chunk_id,
                self.chunk_position,
                self.heading_path,
                self.evidence_quote,
            )
        )

    def __repr__(self) -> str:
        return (
            f"GraphProvenance(source={self.source!r}, "
            f"chunk_id={self.chunk_id!r}, "
            f"position={self.chunk_position})"
        )


class GraphNeighbor:
    """One neighbor entity with its relation type and provenance."""

    __slots__ = (
        "entity_id",
        "display_name",
        "entity_type",
        "relation_type",
        "provenance",
    )

    def __init__(
        self,
        entity_id: str,
        display_name: str,
        entity_type: EntityType,
        relation_type: RelationType,
        provenance: tuple[GraphProvenance, ...],
    ) -> None:
        _require_text(entity_id, "entity_id")
        _require_text(display_name, "display_name")
        self.entity_id = entity_id
        self.display_name = display_name
        self.entity_type = entity_type
        self.relation_type = relation_type
        self.provenance = provenance

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNeighbor):
            return NotImplemented
        return (
            self.entity_id == other.entity_id
            and self.display_name == other.display_name
            and self.entity_type is other.entity_type
            and self.relation_type is other.relation_type
            and self.provenance == other.provenance
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.entity_id,
                self.display_name,
                self.entity_type,
                self.relation_type,
                self.provenance,
            )
        )

    def __repr__(self) -> str:
        return (
            f"GraphNeighbor(entity_id={self.entity_id!r}, "
            f"display_name={self.display_name!r}, "
            f"relation_type={self.relation_type.value!r})"
        )


class EntityNeighborhood:
    """A selected entity with its outgoing and incoming relations."""

    __slots__ = ("entity", "outgoing", "incoming")

    def __init__(
        self,
        entity: GraphEntity,
        outgoing: Sequence[GraphNeighbor],
        incoming: Sequence[GraphNeighbor],
    ) -> None:
        self.entity = entity
        self.outgoing = tuple(outgoing)
        self.incoming = tuple(incoming)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EntityNeighborhood):
            return NotImplemented
        return (
            self.entity == other.entity
            and self.outgoing == other.outgoing
            and self.incoming == other.incoming
        )

    def __hash__(self) -> int:
        return hash((self.entity, self.outgoing, self.incoming))

    def __repr__(self) -> str:
        return (
            f"EntityNeighborhood(entity_id={self.entity.entity_id!r}, "
            f"outgoing={len(self.outgoing)}, "
            f"incoming={len(self.incoming)})"
        )


class EntitySearchResult:
    """One entity matched by a local search query."""

    __slots__ = ("entity_id", "entity_type", "display_name", "score")

    def __init__(
        self,
        entity_id: str,
        entity_type: EntityType,
        display_name: str,
        score: float,
    ) -> None:
        _require_text(entity_id, "entity_id")
        _require_text(display_name, "display_name")
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.display_name = display_name
        self.score = score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EntitySearchResult):
            return NotImplemented
        return (
            self.entity_id == other.entity_id
            and self.entity_type is other.entity_type
            and self.display_name == other.display_name
            and self.score == other.score
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.entity_id,
                self.entity_type,
                self.display_name,
                self.score,
            )
        )

    def __repr__(self) -> str:
        return (
            f"EntitySearchResult(entity_id={self.entity_id!r}, "
            f"display_name={self.display_name!r}, "
            f"score={self.score:.4f})"
        )

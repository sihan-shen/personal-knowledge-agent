"""Strict JSON/reference/evidence validation for per-chunk graph extraction."""

from __future__ import annotations

import json

from personal_knowledge_agent.domain.errors import InvalidGraphSchemaError
from personal_knowledge_agent.domain.graph import (
    MAX_GRAPH_ENTITIES_PER_CHUNK,
    MAX_GRAPH_ENTITY_NAME_LENGTH,
    MAX_GRAPH_EVIDENCE_QUOTE_LENGTH,
    MAX_GRAPH_RELATIONS_PER_CHUNK,
    EntityType,
    GraphChunkExtraction,
    GraphEntity,
    GraphRelation,
    GraphRelationEvidence,
    RelationType,
    make_entity_id,
    make_relation_id,
    normalize_entity_name,
)
from personal_knowledge_agent.domain.models import Chunk, ModelOutput


def parse_graph_extraction(
    output: ModelOutput,
    chunk: Chunk,
) -> GraphChunkExtraction:
    """Parse and validate one chunk's extracted entities and relations.

    Raises ``InvalidGraphSchemaError`` for any schema, reference, size,
    cardinality, or evidence violation.
    """
    try:
        root = json.loads(output.content)
    except (json.JSONDecodeError, ValueError) as error:
        raise InvalidGraphSchemaError(
            f"Invalid graph extraction output: invalid JSON ({error})"
        ) from error

    if not isinstance(root, dict):
        raise InvalidGraphSchemaError(
            "Invalid graph extraction output: expected JSON object"
        )

    if "entities" not in root:
        raise InvalidGraphSchemaError(
            "Invalid graph extraction output: missing 'entities' field"
        )
    if "relations" not in root:
        raise InvalidGraphSchemaError(
            "Invalid graph extraction output: missing 'relations' field"
        )

    entities_raw = root["entities"]
    relations_raw = root["relations"]

    if not isinstance(entities_raw, list):
        raise InvalidGraphSchemaError(
            "Invalid graph extraction output: 'entities' must be a list"
        )
    if not isinstance(relations_raw, list):
        raise InvalidGraphSchemaError(
            "Invalid graph extraction output: 'relations' must be a list"
        )

    # Reject extra top-level fields
    if len(root) != 2:
        extra = [k for k in root if k not in ("entities", "relations")]
        raise InvalidGraphSchemaError(
            f"Invalid graph extraction output: unexpected field(s): {', '.join(extra)}"
        )

    # Build entities
    entities: list[GraphEntity] = []
    seen_identities: set[tuple[str, str]] = set()  # (entity_type, normalized_name)

    for i, item in enumerate(entities_raw):
        if not isinstance(item, dict):
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: expected dict at entities index {i}"
            )

        if "local_id" not in item:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"entity at index {i} missing 'local_id'"
            )
        if "name" not in item:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: entity at index {i} missing 'name'"
            )
        if "type" not in item:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: entity at index {i} missing 'type'"
            )

        local_id = item["local_id"]
        name = item["name"]
        raw_type = item["type"]

        if not isinstance(local_id, str) or not local_id:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"entity at index {i} has non-string or blank local_id"
            )

        expected_id = f"E{i + 1}"
        if local_id != expected_id:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"expected local_id {expected_id!r}, got {local_id!r}"
            )

        if not isinstance(name, str):
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"entity {local_id} has non-string name"
            )
        if not name.strip():
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: entity {local_id} has blank name"
            )
        if len(name) > MAX_GRAPH_ENTITY_NAME_LENGTH:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"entity {local_id} name exceeds "
                f"{MAX_GRAPH_ENTITY_NAME_LENGTH} characters"
            )

        if not isinstance(raw_type, str):
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"entity {local_id} has non-string type"
            )
        try:
            entity_type = EntityType(raw_type)
        except ValueError as error:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"entity {local_id} has unknown type {raw_type!r}"
            ) from error

        normalized = normalize_entity_name(name)
        identity_key = (entity_type.value, normalized)
        if identity_key in seen_identities:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"duplicate entity identity ({entity_type.value}, {normalized})"
            )
        seen_identities.add(identity_key)

        entity_id = make_entity_id(entity_type, name)
        entity = GraphEntity(entity_id, entity_type, name)
        entities.append(entity)

    num_entities = len(entities)
    if num_entities > MAX_GRAPH_ENTITIES_PER_CHUNK:
        raise InvalidGraphSchemaError(
            f"Invalid graph extraction output: "
            f"at most {MAX_GRAPH_ENTITIES_PER_CHUNK} entities, got {num_entities}"
        )

    # Build local ID → entity ID map
    local_to_entity: dict[str, str] = {}
    for i, entity in enumerate(entities):
        local_to_entity[f"E{i + 1}"] = entity.entity_id

    # Build relations
    relations: list[GraphRelation] = []
    evidence_list: list[dict[str, str]] = []  # (relation_id, evidence_quote)

    for i, item in enumerate(relations_raw):
        if not isinstance(item, dict):
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: expected dict at relations index {i}"
            )

        if "source_id" not in item:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} missing 'source_id'"
            )
        if "type" not in item:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: relation at index {i} missing 'type'"
            )
        if "target_id" not in item:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} missing 'target_id'"
            )
        if "evidence_quote" not in item:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} missing 'evidence_quote'"
            )

        source_id = item["source_id"]
        raw_rel_type = item["type"]
        target_id = item["target_id"]
        evidence_quote = item["evidence_quote"]

        if not isinstance(source_id, str):
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} has non-string source_id"
            )
        if not isinstance(target_id, str):
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} has non-string target_id"
            )
        if not isinstance(raw_rel_type, str):
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} has non-string type"
            )
        if not isinstance(evidence_quote, str):
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} has non-string evidence_quote"
            )

        if source_id not in local_to_entity:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} references unknown "
                f"source {source_id!r}"
            )
        if target_id not in local_to_entity:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} references unknown "
                f"target {target_id!r}"
            )

        try:
            rel_type = RelationType(raw_rel_type)
        except ValueError as error:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} has unknown type {raw_rel_type!r}"
            ) from error

        source_entity_id = local_to_entity[source_id]
        target_entity_id = local_to_entity[target_id]

        # Check for self-loop (including casefold-normalized equivalence)
        if rel_type is RelationType.RELATED_TO:
            source_entity = next(e for e in entities if e.entity_id == source_entity_id)
            target_entity = next(e for e in entities if e.entity_id == target_entity_id)
            source_norm = normalize_entity_name(source_entity.display_name)
            target_norm = normalize_entity_name(target_entity.display_name)
            if source_norm == target_norm:
                raise InvalidGraphSchemaError(
                    "Invalid graph extraction output: "
                    "self-loop relation detected "
                    f"({source_norm})"
                )

        try:
            relation_id = make_relation_id(
                rel_type,
                source_entity_id,
                target_entity_id,
            )
        except ValueError as error:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: relation at index {i}: {error}"
            ) from error

        # Validate evidence quote
        if not evidence_quote.strip():
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} has blank evidence_quote"
            )
        if len(evidence_quote) > MAX_GRAPH_EVIDENCE_QUOTE_LENGTH:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} evidence_quote exceeds "
                f"{MAX_GRAPH_EVIDENCE_QUOTE_LENGTH} characters"
            )
        if evidence_quote not in chunk.text:
            raise InvalidGraphSchemaError(
                f"Invalid graph extraction output: "
                f"relation at index {i} evidence_quote is not in chunk text"
            )

        relation = GraphRelation(
            relation_id,
            rel_type,
            source_entity_id,
            target_entity_id,
        )
        relations.append(relation)
        evidence_list.append(
            {"relation_id": relation_id, "evidence_quote": evidence_quote}
        )

    num_relations = len(relations)
    if num_relations > MAX_GRAPH_RELATIONS_PER_CHUNK:
        raise InvalidGraphSchemaError(
            f"Invalid graph extraction output: "
            f"at most {MAX_GRAPH_RELATIONS_PER_CHUNK} relations, "
            f"got {num_relations}"
        )

    # Check for duplicate (relation_id, evidence_quote) pairs
    seen_evidence: set[tuple[str, str]] = set()
    for ev in evidence_list:
        key = (ev["relation_id"], ev["evidence_quote"])
        if key in seen_evidence:
            raise InvalidGraphSchemaError(
                "Invalid graph extraction output: "
                "duplicate (relation, evidence_quote) entry"
            )
        seen_evidence.add(key)

    # Build GraphRelationEvidence list
    relation_evidence = [
        GraphRelationEvidence(ev["relation_id"], ev["evidence_quote"])
        for ev in evidence_list
    ]

    return GraphChunkExtraction(chunk, entities, relation_evidence)

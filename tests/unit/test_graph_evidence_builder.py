"""Tests for deterministic provenance-level graph-answer evidence."""

from __future__ import annotations

from personal_knowledge_agent.domain.graph import (
    EntityNeighborhood,
    EntityType,
    GraphEntity,
    GraphNeighbor,
    GraphProvenance,
    RelationType,
    make_entity_id,
)
from personal_knowledge_agent.infrastructure.graph_evidence_builder import (
    MAX_GRAPH_ANSWER_EVIDENCE,
    build_graph_evidence,
)


def entity(entity_type: EntityType, name: str) -> GraphEntity:
    return GraphEntity(make_entity_id(entity_type, name), entity_type, name)


def provenance(
    quote: str,
    position: int,
    *,
    source: str = "/notes/rag.md",
    chunk_id: str | None = None,
) -> GraphProvenance:
    return GraphProvenance(
        source,
        chunk_id or f"chunk-{position}",
        position,
        ("RAG", "Retrieval"),
        quote,
    )


def neighbor(
    selected: GraphEntity,
    relation_type: RelationType,
    *items: GraphProvenance,
) -> GraphNeighbor:
    return GraphNeighbor(
        selected.entity_id,
        selected.display_name,
        selected.entity_type,
        relation_type,
        tuple(items),
    )


def two_endpoint_neighborhoods(
    outgoing_provenance: GraphProvenance,
    incoming_provenance: GraphProvenance,
) -> tuple[EntityNeighborhood, EntityNeighborhood]:
    rag = entity(EntityType.METHOD, "RAG")
    vectors = entity(EntityType.TECHNOLOGY, "Vector Retrieval")
    return (
        EntityNeighborhood(
            rag,
            (neighbor(vectors, RelationType.USES, outgoing_provenance),),
            (),
        ),
        EntityNeighborhood(
            vectors,
            (),
            (neighbor(rag, RelationType.USES, incoming_provenance),),
        ),
    )


def one_relation_neighborhood(
    *items: GraphProvenance,
) -> tuple[EntityNeighborhood, ...]:
    rag = entity(EntityType.METHOD, "RAG")
    vectors = entity(EntityType.TECHNOLOGY, "Vector Retrieval")
    return (
        EntityNeighborhood(
            rag,
            (neighbor(vectors, RelationType.USES, *items),),
            (),
        ),
    )


def test_restores_outgoing_and_incoming_direction() -> None:
    rag = entity(EntityType.METHOD, "RAG")
    vectors = entity(EntityType.TECHNOLOGY, "Vector Retrieval")
    outgoing = neighbor(vectors, RelationType.USES, provenance("out", 0))
    incoming = neighbor(rag, RelationType.USES, provenance("in", 1))

    evidence = build_graph_evidence(
        (
            EntityNeighborhood(rag, (outgoing,), ()),
            EntityNeighborhood(vectors, (), (incoming,)),
        )
    )

    assert len(evidence) == 2
    assert evidence[0].source_entity_name == "RAG"
    assert evidence[0].target_entity_name == "Vector Retrieval"
    assert evidence[1].source_entity_name == "RAG"
    assert evidence[1].target_entity_name == "Vector Retrieval"


def test_deduplicates_same_relation_and_provenance_reached_from_both_seeds() -> None:
    shared = provenance("shared", 0)

    evidence = build_graph_evidence(two_endpoint_neighborhoods(shared, shared))

    assert [item.citation_id for item in evidence] == ["G1"]


def test_keeps_distinct_provenance_for_one_relation() -> None:
    evidence = build_graph_evidence(
        one_relation_neighborhood(
            provenance("first", 0),
            provenance("second", 1),
        )
    )

    assert [item.provenance.evidence_quote for item in evidence] == [
        "first",
        "second",
    ]


def test_orders_outgoing_before_incoming_within_a_seed() -> None:
    rag = entity(EntityType.METHOD, "RAG")
    alpha = entity(EntityType.CONCEPT, "Alpha")
    zulu = entity(EntityType.CONCEPT, "Zulu")
    neighborhood = EntityNeighborhood(
        rag,
        (neighbor(zulu, RelationType.USES, provenance("outgoing", 0)),),
        (neighbor(alpha, RelationType.USES, provenance("incoming", 1)),),
    )

    evidence = build_graph_evidence((neighborhood,))

    assert [item.provenance.evidence_quote for item in evidence] == [
        "outgoing",
        "incoming",
    ]


def test_orders_neighbors_by_relation_display_name_and_entity_id() -> None:
    rag = entity(EntityType.METHOD, "RAG")
    same_concept = entity(EntityType.CONCEPT, "Same")
    same_technology = entity(EntityType.TECHNOLOGY, "Same")
    alpha = entity(EntityType.CONCEPT, "Alpha")
    neighborhood = EntityNeighborhood(
        rag,
        (
            neighbor(same_technology, RelationType.USES, provenance("technology", 0)),
            neighbor(alpha, RelationType.USES, provenance("alpha", 1)),
            neighbor(same_concept, RelationType.USES, provenance("concept", 2)),
            neighbor(alpha, RelationType.DEPENDS_ON, provenance("depends", 3)),
        ),
        (),
    )

    evidence = build_graph_evidence((neighborhood,))

    same_order = sorted(
        (
            (same_concept.entity_id, "concept"),
            (same_technology.entity_id, "technology"),
        )
    )
    assert [item.provenance.evidence_quote for item in evidence] == [
        "depends",
        "alpha",
        *(label for _, label in same_order),
    ]


def test_orders_non_ascii_neighbor_names_by_unicode_code_point() -> None:
    rag = entity(EntityType.METHOD, "RAG")
    neighborhood = EntityNeighborhood(
        rag,
        (
            neighbor(
                entity(EntityType.CONCEPT, "中文"),
                RelationType.USES,
                provenance("chinese", 0),
            ),
            neighbor(
                entity(EntityType.CONCEPT, "Éclair"),
                RelationType.USES,
                provenance("eclair", 1),
            ),
            neighbor(
                entity(EntityType.CONCEPT, "Ångström"),
                RelationType.USES,
                provenance("angstrom", 2),
            ),
        ),
        (),
    )

    evidence = build_graph_evidence((neighborhood,))

    assert [item.provenance.evidence_quote for item in evidence] == [
        "angstrom",
        "eclair",
        "chinese",
    ]


def test_orders_provenance_by_source_position_chunk_and_quote() -> None:
    evidence = build_graph_evidence(
        one_relation_neighborhood(
            provenance("zulu", 1, source="/notes/a.md", chunk_id="chunk-z"),
            provenance("alpha", 1, source="/notes/a.md", chunk_id="chunk-z"),
            provenance("later-chunk", 1, source="/notes/a.md", chunk_id="chunk-y"),
            provenance("position-one", 1, source="/notes/a.md", chunk_id="chunk-a"),
            provenance("position-zero", 0, source="/notes/a.md", chunk_id="chunk-z"),
            provenance("other-source", 0, source="/notes/b.md", chunk_id="chunk-a"),
        )
    )

    assert [item.provenance.evidence_quote for item in evidence] == [
        "position-zero",
        "position-one",
        "later-chunk",
        "alpha",
        "zulu",
        "other-source",
    ]


def test_best_seed_rank_wins_duplicate_traversal_before_later_seed_evidence() -> None:
    first_seed, second_seed = two_endpoint_neighborhoods(
        provenance("shared", 7, chunk_id="shared-chunk"),
        provenance("shared", 0, chunk_id="shared-chunk"),
    )
    other = entity(EntityType.CONCEPT, "Other")
    later_seed = EntityNeighborhood(
        other,
        (
            neighbor(
                entity(EntityType.CONCEPT, "Third"),
                RelationType.USES,
                provenance("later", 1),
            ),
        ),
        (),
    )

    evidence = build_graph_evidence((first_seed, second_seed, later_seed))

    assert [item.provenance.evidence_quote for item in evidence] == ["shared", "later"]
    assert evidence[0].provenance.chunk_position == 7


def test_is_stable_when_neighbor_and_provenance_containers_are_shuffled() -> None:
    rag = entity(EntityType.METHOD, "RAG")
    alpha = entity(EntityType.CONCEPT, "Alpha")
    beta = entity(EntityType.CONCEPT, "Beta")
    first = EntityNeighborhood(
        rag,
        (
            neighbor(beta, RelationType.USES, provenance("b", 1), provenance("a", 0)),
            neighbor(alpha, RelationType.DEPENDS_ON, provenance("c", 2)),
        ),
        (),
    )
    shuffled = EntityNeighborhood(
        rag,
        (
            neighbor(alpha, RelationType.DEPENDS_ON, provenance("c", 2)),
            neighbor(beta, RelationType.USES, provenance("a", 0), provenance("b", 1)),
        ),
        (),
    )

    assert build_graph_evidence((first,)) == build_graph_evidence((shuffled,))


def test_keeps_canonical_related_to_direction_from_neighborhood() -> None:
    alpha = entity(EntityType.CONCEPT, "Alpha")
    beta = entity(EntityType.CONCEPT, "Beta")
    source, target = sorted((alpha, beta), key=lambda item: item.entity_id)
    outgoing = EntityNeighborhood(
        source,
        (neighbor(target, RelationType.RELATED_TO, provenance("canonical", 0)),),
        (),
    )
    incoming = EntityNeighborhood(
        target,
        (),
        (neighbor(source, RelationType.RELATED_TO, provenance("canonical", 0)),),
    )

    evidence = build_graph_evidence((outgoing, incoming))

    assert len(evidence) == 1
    assert evidence[0].source_entity_name == source.display_name
    assert evidence[0].target_entity_name == target.display_name


def test_related_to_keeps_direction_with_higher_ranked_incoming_seed() -> None:
    alpha = entity(EntityType.CONCEPT, "Alpha")
    beta = entity(EntityType.CONCEPT, "Beta")
    source, target = sorted((alpha, beta), key=lambda item: item.entity_id)
    shared = provenance("canonical", 0)
    higher_ranked_incoming = EntityNeighborhood(
        target,
        (),
        (neighbor(source, RelationType.RELATED_TO, shared),),
    )
    lower_ranked_outgoing = EntityNeighborhood(
        source,
        (neighbor(target, RelationType.RELATED_TO, shared),),
        (),
    )

    evidence = build_graph_evidence((higher_ranked_incoming, lower_ranked_outgoing))

    assert len(evidence) == 1
    assert evidence[0].source_entity_name == source.display_name
    assert evidence[0].target_entity_name == target.display_name


def test_caps_evidence_and_assigns_continuous_labels() -> None:
    evidence = build_graph_evidence(
        one_relation_neighborhood(
            *(provenance(f"quote-{position}", position) for position in range(13))
        )
    )

    assert len(evidence) == MAX_GRAPH_ANSWER_EVIDENCE == 12
    assert [item.citation_id for item in evidence] == [
        f"G{position}" for position in range(1, 13)
    ]
    assert [item.provenance.evidence_quote for item in evidence] == [
        f"quote-{position}" for position in range(12)
    ]


def test_returns_empty_tuple_when_no_relations_exist() -> None:
    neighborhood = EntityNeighborhood(entity(EntityType.CONCEPT, "RAG"), (), ())

    assert build_graph_evidence((neighborhood,)) == ()

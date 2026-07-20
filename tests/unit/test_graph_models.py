"""Tests for knowledge graph domain contracts and identities."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from personal_knowledge_agent.domain.graph import (
    MAX_GRAPH_ENTITIES_PER_CHUNK,
    MAX_GRAPH_ENTITY_NAME_LENGTH,
    MAX_GRAPH_EVIDENCE_QUOTE_LENGTH,
    MAX_GRAPH_RELATIONS_PER_CHUNK,
    MAX_GRAPH_SEARCH_LIMIT,
    EntityNeighborhood,
    EntitySearchResult,
    EntityType,
    GraphBuildResult,
    GraphChunkExtraction,
    GraphEntity,
    GraphNeighbor,
    GraphProvenance,
    GraphRelation,
    GraphRelationEvidence,
    GraphSourceDocument,
    RelationType,
    make_entity_id,
    make_relation_id,
    normalize_entity_name,
)
from personal_knowledge_agent.domain.models import Chunk

# ---------------------------------------------------------------------------
# normalize_entity_name
# ---------------------------------------------------------------------------


class TestNormalizeEntityName:
    def test_whitespace_nfkc_casefold(self) -> None:
        assert normalize_entity_name("  \uff32\uff21\uff27  Model ") == "rag model"

    def test_already_clean(self) -> None:
        assert normalize_entity_name("rag") == "rag"

    def test_mixed_case(self) -> None:
        assert normalize_entity_name("RAG") == "rag"
        assert normalize_entity_name("Vector Retrieval") == "vector retrieval"

    def test_fullwidth_and_spaces(self) -> None:
        assert normalize_entity_name("\u3000\uff4d\u3000L\u3000") == "m l"

    def test_empty_string(self) -> None:
        assert normalize_entity_name("") == ""

    def test_only_whitespace(self) -> None:
        assert normalize_entity_name("  \t  ") == ""


# ---------------------------------------------------------------------------
# make_entity_id / make_relation_id
# ---------------------------------------------------------------------------


_ENT_ID_PATTERN = re.compile(r"^ent_[0-9a-f]{32}$")
_REL_ID_PATTERN = re.compile(r"^rel_[0-9a-f]{32}$")


class TestMakeEntityId:
    def test_deterministic_for_same_type(self) -> None:
        assert make_entity_id(EntityType.METHOD, "RAG") == make_entity_id(
            EntityType.METHOD, "rag"
        )

    def test_different_types_different_ids(self) -> None:
        assert make_entity_id(EntityType.METHOD, "RAG") != make_entity_id(
            EntityType.CONCEPT, "RAG"
        )

    def test_matches_ent_pattern(self) -> None:
        entity_id = make_entity_id(EntityType.CONCEPT, "RAG")
        assert _ENT_ID_PATTERN.match(entity_id), (
            f"ID {entity_id!r} does not match ent_[0-9a-f]{32}"
        )

    def test_different_names_different_ids(self) -> None:
        assert make_entity_id(EntityType.CONCEPT, "RAG") != make_entity_id(
            EntityType.CONCEPT, "Retrieval"
        )


class TestMakeRelationId:
    def test_related_to_canonical_reversal(self) -> None:
        rid_a = make_relation_id(
            RelationType.RELATED_TO,
            make_entity_id(EntityType.CONCEPT, "A"),
            make_entity_id(EntityType.CONCEPT, "B"),
        )
        rid_b = make_relation_id(
            RelationType.RELATED_TO,
            make_entity_id(EntityType.CONCEPT, "B"),
            make_entity_id(EntityType.CONCEPT, "A"),
        )
        assert rid_a == rid_b

    def test_directed_not_reversible(self) -> None:
        rid_a = make_relation_id(
            RelationType.USES,
            make_entity_id(EntityType.METHOD, "RAG"),
            make_entity_id(EntityType.TECHNOLOGY, "Vector Retrieval"),
        )
        rid_b = make_relation_id(
            RelationType.USES,
            make_entity_id(EntityType.TECHNOLOGY, "Vector Retrieval"),
            make_entity_id(EntityType.METHOD, "RAG"),
        )
        assert rid_a != rid_b

    def test_matches_rel_pattern(self) -> None:
        rel_id = make_relation_id(
            RelationType.IS_A,
            make_entity_id(EntityType.CONCEPT, "A"),
            make_entity_id(EntityType.CONCEPT, "B"),
        )
        assert _REL_ID_PATTERN.match(rel_id), (
            f"ID {rel_id!r} does not match rel_[0-9a-f]{32}"
        )

    def test_self_loop_raises_value_error(self) -> None:
        eid = make_entity_id(EntityType.CONCEPT, "Self")
        with pytest.raises(ValueError, match="self-loop"):
            make_relation_id(RelationType.RELATED_TO, eid, eid)


# ---------------------------------------------------------------------------
# EntityType / RelationType enums
# ---------------------------------------------------------------------------


class TestEntityTypeEnum:
    def test_all_values_match_spec(self) -> None:
        values = {member.value for member in EntityType}
        assert values == {"concept", "technology", "method", "project", "resource"}


class TestRelationTypeEnum:
    def test_all_values_match_spec(self) -> None:
        values = {member.value for member in RelationType}
        assert values == {
            "is_a",
            "part_of",
            "uses",
            "depends_on",
            "implements",
            "related_to",
        }


# ---------------------------------------------------------------------------
# GraphEntity
# ---------------------------------------------------------------------------


class TestGraphEntity:
    def test_display_name_preserves_case(self) -> None:
        entity = GraphEntity(
            make_entity_id(EntityType.METHOD, "RAG"),
            EntityType.METHOD,
            "RAG",
        )
        assert entity.display_name == "RAG"
        assert entity.display_name != "rag"

    def test_empty_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="name must not be blank"):
            GraphEntity(
                make_entity_id(EntityType.CONCEPT, "test"),
                EntityType.CONCEPT,
                "   ",
            )

    def test_normalizes_display_name_without_changing_case(self) -> None:
        entity = GraphEntity(
            make_entity_id(EntityType.METHOD, "RAG Model"),
            EntityType.METHOD,
            "  ＲＡＧ\tModel  ",
        )

        assert entity.display_name == "RAG Model"

    def test_rejects_an_id_for_a_different_identity(self) -> None:
        with pytest.raises(ValueError, match="entity_id"):
            GraphEntity(
                make_entity_id(EntityType.METHOD, "RAG"),
                EntityType.METHOD,
                "Vector Retrieval",
            )

    def test_is_immutable_after_construction(self) -> None:
        entity = GraphEntity(
            make_entity_id(EntityType.METHOD, "RAG"),
            EntityType.METHOD,
            "RAG",
        )

        with pytest.raises(AttributeError):
            entity.display_name = "Tampered"


# ---------------------------------------------------------------------------
# GraphRelation
# ---------------------------------------------------------------------------


class TestGraphRelation:
    def test_related_to_canonical_endpoints(self) -> None:
        eid_a = make_entity_id(EntityType.CONCEPT, "A")
        eid_b = make_entity_id(EntityType.CONCEPT, "B")
        rel = GraphRelation(
            make_relation_id(RelationType.RELATED_TO, eid_b, eid_a),
            RelationType.RELATED_TO,
            eid_b,
            eid_a,
        )
        assert rel.source_entity_id == min(eid_a, eid_b)
        assert rel.target_entity_id == max(eid_a, eid_b)

    def test_rejects_an_id_for_a_different_relation(self) -> None:
        source_id = make_entity_id(EntityType.METHOD, "RAG")
        target_id = make_entity_id(EntityType.TECHNOLOGY, "Vector Retrieval")
        with pytest.raises(ValueError, match="relation_id"):
            GraphRelation(
                make_relation_id(RelationType.DEPENDS_ON, source_id, target_id),
                RelationType.USES,
                source_id,
                target_id,
            )

    def test_store_quote(self) -> None:
        source_id = make_entity_id(EntityType.METHOD, "RAG")
        target_id = make_entity_id(EntityType.TECHNOLOGY, "Vector Retrieval")
        relation = GraphRelation(
            make_relation_id(RelationType.USES, source_id, target_id),
            RelationType.USES,
            source_id,
            target_id,
        )
        evidence = GraphRelationEvidence(
            relation,
            "RAG uses vector retrieval.",
        )
        assert evidence.relation is relation
        assert evidence.evidence_quote == "RAG uses vector retrieval."

    def test_blank_quote_raises(self) -> None:
        source_id = make_entity_id(EntityType.METHOD, "RAG")
        target_id = make_entity_id(EntityType.TECHNOLOGY, "Vector Retrieval")
        relation = GraphRelation(
            make_relation_id(RelationType.USES, source_id, target_id),
            RelationType.USES,
            source_id,
            target_id,
        )
        with pytest.raises(ValueError, match="evidence_quote must not be blank"):
            GraphRelationEvidence(
                relation,
                "  ",
            )


# ---------------------------------------------------------------------------
# GraphChunkExtraction
# ---------------------------------------------------------------------------


class TestGraphChunkExtraction:
    def test_holds_chunk_and_entities(self) -> None:
        chunk = Chunk("chunk1", "doc1", 0, "RAG uses vector retrieval.")
        entity = GraphEntity(
            make_entity_id(EntityType.METHOD, "RAG"),
            EntityType.METHOD,
            "RAG",
        )
        extraction = GraphChunkExtraction(chunk, (entity,), ())
        assert extraction.chunk == chunk
        assert extraction.entities == (entity,)
        assert extraction.relation_evidence == ()

    def test_too_many_entities_raises(self) -> None:
        chunk = Chunk(
            "c1",
            "d1",
            0,
            " ".join(f"evidence {index}" for index in range(21)),
        )
        entities = tuple(
            GraphEntity(
                make_entity_id(EntityType.CONCEPT, str(i)),
                EntityType.CONCEPT,
                str(i),
            )
            for i in range(MAX_GRAPH_ENTITIES_PER_CHUNK + 1)
        )
        with pytest.raises(ValueError, match=f"at most {MAX_GRAPH_ENTITIES_PER_CHUNK}"):
            GraphChunkExtraction(chunk, entities, ())

    def test_too_many_relations_raises(self) -> None:
        chunk = Chunk("c1", "d1", 0, "x")
        eid_a = make_entity_id(EntityType.CONCEPT, "A")
        eid_b = make_entity_id(EntityType.CONCEPT, "B")
        entities = (
            GraphEntity(eid_a, EntityType.CONCEPT, "A"),
            GraphEntity(eid_b, EntityType.CONCEPT, "B"),
        )
        relation = GraphRelation(
            make_relation_id(RelationType.RELATED_TO, eid_a, eid_b),
            RelationType.RELATED_TO,
            eid_a,
            eid_b,
        )
        relation_evidence = tuple(
            GraphRelationEvidence(relation, f"evidence {index}")
            for index in range(MAX_GRAPH_RELATIONS_PER_CHUNK + 1)
        )
        with pytest.raises(
            ValueError, match=f"at most {MAX_GRAPH_RELATIONS_PER_CHUNK}"
        ):
            GraphChunkExtraction(chunk, entities, relation_evidence)


# ---------------------------------------------------------------------------
# GraphSourceDocument
# ---------------------------------------------------------------------------


class TestGraphSourceDocument:
    def test_valid_document(self) -> None:
        chunk = Chunk("c1", "doc1", 0, "RAG content", ("Architecture",))
        doc = GraphSourceDocument(
            "doc1",
            Path("/notes/rag.md"),
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            (chunk,),
        )
        assert doc.document_id == "doc1"

    def test_blank_document_id_raises(self) -> None:
        chunk = Chunk("c1", "doc1", 0, "x")
        with pytest.raises(ValueError, match="document_id must not be blank"):
            GraphSourceDocument(
                "",
                Path("/notes/rag.md"),
                "a" * 64,
                (chunk,),
            )

    def test_non_64_hex_hash_raises(self) -> None:
        chunk = Chunk("c1", "doc1", 0, "x")
        with pytest.raises(ValueError, match="content_hash"):
            GraphSourceDocument(
                "doc1",
                Path("/notes/rag.md"),
                "not-a-real-hash",
                (chunk,),
            )

    def test_chunk_from_different_document_raises(self) -> None:
        chunk = Chunk("c1", "other_doc", 0, "x")
        with pytest.raises(ValueError, match="document"):
            GraphSourceDocument(
                "doc1",
                Path("/notes/rag.md"),
                "a" * 64,
                (chunk,),
            )

    def test_duplicate_chunk_positions_raises(self) -> None:
        with pytest.raises(ValueError, match="position"):
            GraphSourceDocument(
                "doc1",
                Path("/notes/rag.md"),
                "a" * 64,
                (
                    Chunk("c1", "doc1", 0, "first"),
                    Chunk("c2", "doc1", 0, "second"),
                ),
            )

    def test_nonscending_chunk_positions_raises(self) -> None:
        """Raises when positions are not in ascending order."""
        with pytest.raises(ValueError, match="position"):
            GraphSourceDocument(
                "doc1",
                Path("/notes/rag.md"),
                "a" * 64,
                (
                    Chunk("c2", "doc1", 1, "second"),
                    Chunk("c1", "doc1", 0, "first"),
                ),
            )


# ---------------------------------------------------------------------------
# GraphBuildResult
# ---------------------------------------------------------------------------


class TestGraphBuildResult:
    def test_counts(self) -> None:
        result = GraphBuildResult(
            source="test.md", chunks=3, entities=5, relations=7, evidence=9
        )
        assert result.source == "test.md"
        assert result.chunks == 3
        assert result.entities == 5
        assert result.relations == 7
        assert result.evidence == 9

    def test_negative_count_raises(self) -> None:
        with pytest.raises(ValueError, match="chunks"):
            GraphBuildResult(
                source="t.md", chunks=-1, entities=0, relations=0, evidence=0
            )


# ---------------------------------------------------------------------------
# GraphProvenance
# ---------------------------------------------------------------------------


class TestGraphProvenance:
    def test_fields(self) -> None:
        prov = GraphProvenance(
            source="/notes/rag.md",
            chunk_id="c1",
            chunk_position=0,
            heading_path=("Architecture",),
            evidence_quote="RAG uses vector retrieval.",
        )
        assert prov.source == "/notes/rag.md"
        assert prov.chunk_id == "c1"
        assert prov.chunk_position == 0
        assert prov.heading_path == ("Architecture",)
        assert prov.evidence_quote == "RAG uses vector retrieval."


# ---------------------------------------------------------------------------
# GraphNeighbor and EntityNeighborhood
# ---------------------------------------------------------------------------


class TestGraphNeighbor:
    def test_fields(self) -> None:
        neighbor = GraphNeighbor(
            entity_id="ent_abcd1234abcd1234abcd1234abcd1234",
            display_name="RAG",
            entity_type=EntityType.METHOD,
            relation_type=RelationType.USES,
            provenance=(),
        )
        assert neighbor.entity_type is EntityType.METHOD
        assert neighbor.relation_type is RelationType.USES


class TestEntityNeighborhood:
    def test_entity_fields(self) -> None:
        entity = GraphEntity(
            make_entity_id(EntityType.METHOD, "RAG"),
            EntityType.METHOD,
            "RAG",
        )
        neighborhood = EntityNeighborhood(entity=entity, outgoing=(), incoming=())
        assert neighborhood.entity == entity
        assert neighborhood.outgoing == ()
        assert neighborhood.incoming == ()


# ---------------------------------------------------------------------------
# EntitySearchResult
# ---------------------------------------------------------------------------


class TestEntitySearchResult:
    def test_fields(self) -> None:
        result = EntitySearchResult(
            entity_id="ent_abcd1234abcd1234abcd1234abcd1234",
            entity_type=EntityType.CONCEPT,
            display_name="RAG",
            score=0.95,
        )
        assert result.score == 0.95


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_max_entity_name_length(self) -> None:
        assert MAX_GRAPH_ENTITY_NAME_LENGTH == 120

    def test_max_evidence_quote_length(self) -> None:
        assert MAX_GRAPH_EVIDENCE_QUOTE_LENGTH == 300

    def test_max_search_limit(self) -> None:
        assert MAX_GRAPH_SEARCH_LIMIT == 20

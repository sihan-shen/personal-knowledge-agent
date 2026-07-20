"""Integration tests for the SQLite knowledge graph store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import (
    GraphEmptyError,
    GraphEntityNotFoundError,
    GraphStorageError,
)
from personal_knowledge_agent.domain.graph import (
    EntityType,
    GraphChunkExtraction,
    GraphEntity,
    GraphRelation,
    GraphRelationEvidence,
    GraphSourceDocument,
    RelationType,
    make_entity_id,
    make_relation_id,
)
from personal_knowledge_agent.domain.models import Chunk
from personal_knowledge_agent.infrastructure.sqlite_knowledge_graph_store import (
    SQLiteKnowledgeGraphStore,
)


def _make_chunk(position: int = 0, text: str = "RAG uses vector retrieval.") -> Chunk:
    cid = f"c{position}"
    return Chunk(cid, "doc1", position, text, ("Architecture",))


def _make_doc(source: str = "/tmp/rag.md") -> GraphSourceDocument:
    return GraphSourceDocument(
        "doc1",
        Path(source),
        "a" * 64,
        (_make_chunk(),),
    )


def _make_entity(
    name: str = "RAG", etype: EntityType = EntityType.METHOD
) -> GraphEntity:
    return GraphEntity(
        make_entity_id(etype, name),
        etype,
        name,
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteKnowledgeGraphStore:
    return SQLiteKnowledgeGraphStore(tmp_path / "graph.db")


class TestStoreSchema:
    def test_creates_tables(self, tmp_path: Path) -> None:
        db = tmp_path / "graph.db"
        store = SQLiteKnowledgeGraphStore(db)
        # Trigger initialization
        _replace_simple(store)
        import sqlite3

        with sqlite3.connect(db) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert {
            "schema_metadata",
            "graph_documents",
            "entities",
            "entity_mentions",
            "relations",
            "relation_evidence",
            "entity_postings",
        }.issubset(tables)


class TestReplaceDocument:
    def test_basic_replacement(self, store: SQLiteKnowledgeGraphStore) -> None:
        result = _replace_simple(store)
        assert result.entities == 2
        assert result.chunks == 1

    def test_search_after_replacement(self, store: SQLiteKnowledgeGraphStore) -> None:
        _replace_simple(store)
        results = store.search_entities("RAG", limit=5)
        assert len(results) >= 1
        assert results[0].display_name == "RAG"


class TestSearchEntities:
    def test_empty_graph_raises(self, store: SQLiteKnowledgeGraphStore) -> None:
        with pytest.raises(GraphEmptyError, match="empty"):
            store.search_entities("RAG")

    def test_returns_filtered_results(self, store: SQLiteKnowledgeGraphStore) -> None:
        _replace_simple(store)
        results = store.search_entities("Vector", limit=5)
        assert len(results) >= 1


class TestGetNeighborhood:
    def test_entity_header(self, store: SQLiteKnowledgeGraphStore) -> None:
        _replace_simple(store)
        result = store.get_neighborhood(make_entity_id(EntityType.METHOD, "RAG"))
        assert result.entity.display_name == "RAG"
        assert result.entity.entity_type is EntityType.METHOD

    def test_unknown_entity_raises(self, store: SQLiteKnowledgeGraphStore) -> None:
        _replace_simple(store)
        with pytest.raises(GraphEntityNotFoundError):
            store.get_neighborhood("ent_00000000000000000000000000000000")


def _replace_simple(store: SQLiteKnowledgeGraphStore) -> object:
    """Replace with a simple RAG + Vector Retrieval extraction."""
    rag_entity = _make_entity("RAG", EntityType.METHOD)
    vr_entity = _make_entity("Vector Retrieval", EntityType.TECHNOLOGY)
    chunk = _make_chunk()
    doc = _make_doc()

    rel = GraphRelation(
        make_relation_id(RelationType.USES, rag_entity.entity_id, vr_entity.entity_id),
        RelationType.USES,
        rag_entity.entity_id,
        vr_entity.entity_id,
    )
    evidence = GraphRelationEvidence(rel, "RAG uses vector retrieval.")

    extraction = GraphChunkExtraction(chunk, (rag_entity, vr_entity), (evidence,))
    return store.replace_document(doc, (extraction,))


def _document_with_relation(
    *,
    document_id: str,
    source: str,
    quote: str,
    rag_name: str = "RAG",
    vector_name: str = "Vector Retrieval",
) -> tuple[GraphSourceDocument, GraphChunkExtraction]:
    rag = _make_entity(rag_name, EntityType.METHOD)
    vector_retrieval = _make_entity(vector_name, EntityType.TECHNOLOGY)
    chunk = Chunk(
        f"{document_id}-c1",
        document_id,
        0,
        quote,
        ("Architecture",),
    )
    relation = GraphRelation(
        make_relation_id(
            RelationType.USES,
            rag.entity_id,
            vector_retrieval.entity_id,
        ),
        RelationType.USES,
        rag.entity_id,
        vector_retrieval.entity_id,
    )
    return (
        GraphSourceDocument(document_id, Path(source), "a" * 64, (chunk,)),
        GraphChunkExtraction(
            chunk,
            (rag, vector_retrieval),
            (GraphRelationEvidence(relation, quote),),
        ),
    )


def test_replacement_keeps_relation_direction_and_complete_provenance(
    store: SQLiteKnowledgeGraphStore,
) -> None:
    _replace_simple(store)

    neighborhood = store.get_neighborhood(make_entity_id(EntityType.METHOD, "RAG"))

    assert len(neighborhood.outgoing) == 1
    neighbor = neighborhood.outgoing[0]
    assert neighbor.relation_type is RelationType.USES
    assert neighbor.entity_type is EntityType.TECHNOLOGY
    assert neighbor.display_name == "Vector Retrieval"
    assert len(neighbor.provenance) == 1
    assert neighbor.provenance[0].source == "/tmp/rag.md"
    assert neighbor.provenance[0].heading_path == ("Architecture",)
    assert neighbor.provenance[0].evidence_quote == "RAG uses vector retrieval."


def test_neighborhood_keeps_all_evidence_in_provenance_order(
    store: SQLiteKnowledgeGraphStore,
) -> None:
    rag = _make_entity("RAG", EntityType.METHOD)
    vector_retrieval = _make_entity("Vector Retrieval", EntityType.TECHNOLOGY)
    relation = GraphRelation(
        make_relation_id(
            RelationType.USES,
            rag.entity_id,
            vector_retrieval.entity_id,
        ),
        RelationType.USES,
        rag.entity_id,
        vector_retrieval.entity_id,
    )
    first_chunk = Chunk(
        "c1",
        "doc1",
        0,
        "RAG uses vector retrieval for context.",
        ("Architecture",),
    )
    second_chunk = Chunk(
        "c2",
        "doc1",
        1,
        "RAG uses vector retrieval for ranking.",
        ("Retrieval",),
    )
    document = GraphSourceDocument(
        "doc1",
        Path("/tmp/rag.md"),
        "a" * 64,
        (first_chunk, second_chunk),
    )
    store.replace_document(
        document,
        (
            GraphChunkExtraction(
                first_chunk,
                (rag, vector_retrieval),
                (GraphRelationEvidence(relation, first_chunk.text),),
            ),
            GraphChunkExtraction(
                second_chunk,
                (rag, vector_retrieval),
                (GraphRelationEvidence(relation, second_chunk.text),),
            ),
        ),
    )

    neighborhood = store.get_neighborhood(rag.entity_id)
    assert [item.chunk_id for item in neighborhood.outgoing[0].provenance] == [
        "c1",
        "c2",
    ]
    assert [item.heading_path for item in neighborhood.outgoing[0].provenance] == [
        ("Architecture",),
        ("Retrieval",),
    ]


def test_replacement_preserves_shared_relation_evidence_from_another_source(
    store: SQLiteKnowledgeGraphStore,
) -> None:
    first_document, first_extraction = _document_with_relation(
        document_id="doc1",
        source="/tmp/first.md",
        quote="RAG uses vector retrieval.",
    )
    second_document, second_extraction = _document_with_relation(
        document_id="doc2",
        source="/tmp/second.md",
        quote="RAG uses vector retrieval.",
    )
    store.replace_document(first_document, (first_extraction,))
    store.replace_document(second_document, (second_extraction,))

    empty_first_document = GraphSourceDocument(
        "doc1", Path("/tmp/first.md"), "b" * 64, ()
    )
    store.replace_document(empty_first_document, ())

    neighborhood = store.get_neighborhood(make_entity_id(EntityType.METHOD, "RAG"))
    assert len(neighborhood.outgoing) == 1
    assert [item.source for item in neighborhood.outgoing[0].provenance] == [
        "/tmp/second.md"
    ]


def test_replacement_recomputes_display_name_from_active_mentions(
    store: SQLiteKnowledgeGraphStore,
) -> None:
    first_document, first_extraction = _document_with_relation(
        document_id="doc1",
        source="/tmp/first.md",
        quote="RAG uses Vector Retrieval.",
    )
    second_document, second_extraction = _document_with_relation(
        document_id="doc2",
        source="/tmp/second.md",
        quote="rag uses vector retrieval.",
        rag_name="rag",
        vector_name="vector retrieval",
    )
    store.replace_document(first_document, (first_extraction,))
    store.replace_document(second_document, (second_extraction,))
    assert (
        store.get_neighborhood(
            make_entity_id(EntityType.METHOD, "RAG")
        ).entity.display_name
        == "RAG"
    )

    store.replace_document(
        GraphSourceDocument("doc1", Path("/tmp/first.md"), "b" * 64, ()),
        (),
    )

    assert (
        store.get_neighborhood(
            make_entity_id(EntityType.METHOD, "RAG")
        ).entity.display_name
        == "rag"
    )


def test_replacement_rolls_back_when_a_database_write_fails(tmp_path: Path) -> None:
    database = tmp_path / "graph.db"
    store = SQLiteKnowledgeGraphStore(database)
    original_document, original_extraction = _document_with_relation(
        document_id="doc1",
        source="/tmp/rag.md",
        quote="RAG uses vector retrieval.",
    )
    store.replace_document(original_document, (original_extraction,))

    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TRIGGER fail_entity_mentions BEFORE INSERT ON entity_mentions "
            "BEGIN SELECT RAISE(ABORT, 'forced failure'); END"
        )

    replacement_document, replacement_extraction = _document_with_relation(
        document_id="doc1",
        source="/tmp/rag.md",
        quote="RAG uses vector retrieval in a replacement.",
    )
    with pytest.raises(GraphStorageError):
        store.replace_document(replacement_document, (replacement_extraction,))

    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER fail_entity_mentions")

    neighborhood = store.get_neighborhood(make_entity_id(EntityType.METHOD, "RAG"))
    assert neighborhood.outgoing[0].provenance[0].evidence_quote == (
        "RAG uses vector retrieval."
    )


def test_empty_graph_neighborhood_raises_empty_graph_error(
    store: SQLiteKnowledgeGraphStore,
) -> None:
    with pytest.raises(GraphEmptyError):
        store.get_neighborhood(make_entity_id(EntityType.METHOD, "RAG"))


def test_corrupt_stored_entity_data_maps_to_storage_error(tmp_path: Path) -> None:
    database = tmp_path / "graph.db"
    store = SQLiteKnowledgeGraphStore(database)
    _replace_simple(store)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE entities SET entity_type = 'invalid'")

    with pytest.raises(GraphStorageError):
        store.search_entities("RAG")


def test_corrupt_stored_entity_identity_maps_to_storage_error(tmp_path: Path) -> None:
    database = tmp_path / "graph.db"
    store = SQLiteKnowledgeGraphStore(database)
    _replace_simple(store)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE entities SET display_name = 'Tampered' WHERE display_name = 'RAG'"
        )

    with pytest.raises(GraphStorageError):
        store.search_entities("RAG")


def test_corrupt_stored_relation_identity_maps_to_storage_error(tmp_path: Path) -> None:
    database = tmp_path / "graph.db"
    store = SQLiteKnowledgeGraphStore(database)
    _replace_simple(store)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE relations SET relation_type = 'depends_on'")

    with pytest.raises(GraphStorageError):
        store.get_neighborhood(make_entity_id(EntityType.METHOD, "RAG"))

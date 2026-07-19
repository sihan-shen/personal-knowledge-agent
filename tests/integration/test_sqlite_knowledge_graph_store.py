"""Integration tests for the SQLite knowledge graph store."""

from __future__ import annotations

import json
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


def _make_entity(name: str = "RAG", etype: EntityType = EntityType.METHOD) -> GraphEntity:
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
        assert "graph_entities" in tables
        assert "graph_edges" in tables
        assert "graph_entity_postings" in tables


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
        result = store.get_neighborhood(
            make_entity_id(EntityType.METHOD, "RAG")
        )
        assert result.entity.display_name == "RAG"
        assert result.entity.entity_type is EntityType.METHOD

    def test_unknown_entity_raises(self, store: SQLiteKnowledgeGraphStore) -> None:
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
    evidence = GraphRelationEvidence(rel.relation_id, "RAG uses vector retrieval.")

    extraction = GraphChunkExtraction(chunk, (rag_entity, vr_entity), (evidence,))
    return store.replace_document(doc, (extraction,))

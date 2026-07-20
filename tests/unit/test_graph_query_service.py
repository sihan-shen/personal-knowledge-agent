"""Tests for the graph query validation service."""

from __future__ import annotations

import pytest

from personal_knowledge_agent.application.graph_query import GraphQueryService
from personal_knowledge_agent.domain.errors import InvalidGraphError
from personal_knowledge_agent.domain.graph import (
    EntityNeighborhood,
    EntitySearchResult,
    EntityType,
)


class _FakeStore:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []
        self.show_calls: list[str] = []

    def search_entities(
        self, query: str, *, limit: int = 10
    ) -> list[EntitySearchResult]:
        self.search_calls.append((query, limit))
        return [
            EntitySearchResult("ent_test", EntityType.CONCEPT, "Test", 1.0),
        ]

    def get_neighborhood(self, entity_id: str) -> EntityNeighborhood:
        self.show_calls.append(entity_id)
        raise NotImplementedError

    def replace_document(self, *args, **kwargs):
        raise NotImplementedError


class TestGraphQueryService:
    def test_valid_search(self) -> None:
        store = _FakeStore()
        service = GraphQueryService(store)
        results = service.search("RAG", limit=5)
        assert len(results) == 1
        assert results[0].display_name == "Test"

    def test_blank_query_raises(self) -> None:
        service = GraphQueryService(_FakeStore())
        with pytest.raises(InvalidGraphError, match="blank"):
            service.search("  ")

    def test_limit_zero_raises(self) -> None:
        service = GraphQueryService(_FakeStore())
        with pytest.raises(InvalidGraphError, match="limit"):
            service.search("RAG", limit=0)

    def test_limit_21_raises(self) -> None:
        service = GraphQueryService(_FakeStore())
        with pytest.raises(InvalidGraphError, match="limit"):
            service.search("RAG", limit=21)

    def test_invalid_entity_id_raises(self) -> None:
        service = GraphQueryService(_FakeStore())
        with pytest.raises(InvalidGraphError, match="entity"):
            service.show("invalid_id")

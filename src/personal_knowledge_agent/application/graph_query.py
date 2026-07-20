"""Local query validation and delegation for knowledge graph queries."""

from __future__ import annotations

import re

from personal_knowledge_agent.domain.errors import InvalidGraphError
from personal_knowledge_agent.domain.graph import (
    MAX_GRAPH_SEARCH_LIMIT,
    EntityNeighborhood,
    EntitySearchResult,
)
from personal_knowledge_agent.domain.protocols import KnowledgeGraphStore

_ENTITY_ID_PATTERN = re.compile(r"^ent_[0-9a-f]{32}$")


class GraphQueryService:
    """Validate queries and entity IDs, then delegate to the graph store.

    This service contains no tokenizer, SQL, or ranking implementation.
    """

    def __init__(self, store: KnowledgeGraphStore) -> None:
        self._store = store

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[EntitySearchResult]:
        """Search graph entities by BM25-ranked query."""
        if not query.strip():
            raise InvalidGraphError("Search query must not be blank")
        if not 1 <= limit <= MAX_GRAPH_SEARCH_LIMIT:
            raise InvalidGraphError(
                f"Search limit must be between 1 and {MAX_GRAPH_SEARCH_LIMIT}"
            )
        return self._store.search_entities(query, limit=limit)

    def show(self, entity_id: str) -> EntityNeighborhood:
        """Return one entity with its inbound and outbound relations."""
        if not entity_id or not _ENTITY_ID_PATTERN.match(entity_id):
            raise InvalidGraphError(
                f"Invalid entity ID: {entity_id}. "
                "Expected ent_ followed by 32 hex characters."
            )
        return self._store.get_neighborhood(entity_id)

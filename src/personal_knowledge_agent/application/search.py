"""Application service for knowledge search."""

from __future__ import annotations

from personal_knowledge_agent.domain.errors import InvalidQueryError
from personal_knowledge_agent.domain.models import SearchResult
from personal_knowledge_agent.domain.protocols import KnowledgeIndex


class SearchService:
    """Validate search options and delegate ranking to an index."""

    def __init__(self, index: KnowledgeIndex) -> None:
        self._index = index

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Return ranked chunks within the supported result limit."""
        if not 1 <= limit <= 20:
            raise InvalidQueryError("Search limit must be between 1 and 20")
        return self._index.search(query, limit=limit)

"""Convert ranked search results into request-local citation evidence."""

from __future__ import annotations

from collections.abc import Sequence

from personal_knowledge_agent.domain.models import Evidence, SearchResult


def build_evidence(results: Sequence[SearchResult]) -> tuple[Evidence, ...]:
    """Assign stable request-local labels while preserving retrieval order."""
    return tuple(
        Evidence(
            citation_id=f"S{rank}",
            chunk_id=result.chunk.chunk_id,
            source=result.source,
            heading_path=result.chunk.heading_path,
            text=result.chunk.text,
            retrieval_score=result.score,
        )
        for rank, result in enumerate(results, start=1)
    )

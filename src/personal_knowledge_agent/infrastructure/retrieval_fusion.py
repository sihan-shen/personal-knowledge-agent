"""Pure rank-fusion utilities for retrieval results."""

from __future__ import annotations

from collections.abc import Sequence

from personal_knowledge_agent.domain.models import SearchResult


def reciprocal_rank_fusion(
    *rankings: Sequence[SearchResult],
    limit: int,
    rank_constant: int = 60,
) -> list[SearchResult]:
    """Combine rankings using RRF and stable chunk-ID tie breaking."""
    scores: dict[str, float] = {}
    candidates: dict[str, SearchResult] = {}

    for ranking in rankings:
        seen: set[str] = set()
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            candidates.setdefault(chunk_id, result)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (
                rank_constant + rank
            )

    fused = [
        SearchResult(
            chunk=candidates[chunk_id].chunk,
            source=candidates[chunk_id].source,
            score=score,
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
    return fused[:limit]

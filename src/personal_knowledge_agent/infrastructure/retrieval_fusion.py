"""Pure rank-fusion utilities for retrieval results."""

from __future__ import annotations

from collections.abc import Sequence

from personal_knowledge_agent.domain.models import SearchResult


def reciprocal_rank_scores(
    *rankings: Sequence[str],
    limit: int,
    rank_constant: int = 60,
) -> list[tuple[str, float]]:
    """Combine stable ID rankings using RRF and ID tie breaking."""
    scores: dict[str, float] = {}

    for ranking in rankings:
        seen: set[str] = set()
        for rank, stable_id in enumerate(ranking, start=1):
            if stable_id in seen:
                continue
            seen.add(stable_id)
            scores[stable_id] = scores.get(stable_id, 0.0) + 1 / (rank_constant + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def reciprocal_rank_fusion(
    *rankings: Sequence[SearchResult],
    limit: int,
    rank_constant: int = 60,
) -> list[SearchResult]:
    """Combine rankings using RRF and stable chunk-ID tie breaking."""
    candidates: dict[str, SearchResult] = {}

    for ranking in rankings:
        for result in ranking:
            chunk_id = result.chunk.chunk_id
            candidates.setdefault(chunk_id, result)

    return [
        SearchResult(
            chunk=candidates[chunk_id].chunk,
            source=candidates[chunk_id].source,
            score=score,
        )
        for chunk_id, score in reciprocal_rank_scores(
            *([result.chunk.chunk_id for result in ranking] for ranking in rankings),
            limit=limit,
            rank_constant=rank_constant,
        )
    ]

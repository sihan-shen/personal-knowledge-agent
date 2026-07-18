"""Tests for deterministic reciprocal-rank fusion."""

import pytest

from personal_knowledge_agent.domain.models import Chunk, SearchResult
from personal_knowledge_agent.infrastructure.retrieval_fusion import (
    reciprocal_rank_fusion,
    reciprocal_rank_scores,
)


def result(chunk_id: str, score: float = 1.0) -> SearchResult:
    chunk = Chunk(chunk_id, "document", 0, chunk_id)
    return SearchResult(chunk, "/notes.md", score)


def test_rrf_combines_rankings_and_keeps_single_ranking_candidates() -> None:
    lexical = [result("a"), result("b")]
    semantic = [result("b"), result("c")]

    fused = reciprocal_rank_fusion(lexical, semantic, limit=3)

    assert [item.chunk.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)


def test_rrf_breaks_equal_scores_by_chunk_id_and_applies_limit() -> None:
    fused = reciprocal_rank_fusion(
        [result("z")],
        [result("a")],
        limit=1,
    )

    assert [item.chunk.chunk_id for item in fused] == ["a"]


def test_rrf_does_not_count_duplicate_chunk_twice_in_one_ranking() -> None:
    duplicate = result("a")

    fused = reciprocal_rank_fusion([duplicate, duplicate], limit=1)

    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_scores_stable_ids_for_non_document_consumers() -> None:
    fused = reciprocal_rank_scores(["mem_a", "mem_b"], ["mem_b", "mem_c"], limit=3)

    assert [memory_id for memory_id, _ in fused] == ["mem_b", "mem_a", "mem_c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_id_scores_ignore_duplicates_and_break_ties_by_id() -> None:
    assert reciprocal_rank_scores(["z", "z"], ["a"], limit=1)[0][0] == "a"

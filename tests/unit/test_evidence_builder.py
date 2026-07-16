"""Tests for converting ranked retrieval results into local evidence."""

from personal_knowledge_agent.domain.models import Chunk, SearchResult
from personal_knowledge_agent.infrastructure.evidence_builder import build_evidence


def make_result(position: int, score: float) -> SearchResult:
    chunk = Chunk(
        f"chunk-{position}",
        "document",
        position,
        f"evidence {position}",
        ("RAG", f"Section {position}"),
    )
    return SearchResult(chunk, f"/notes/{position}.md", score)


def test_build_evidence_assigns_labels_and_preserves_retrieval_order() -> None:
    results = [make_result(2, 0.9), make_result(0, 0.7), make_result(1, 0.5)]

    evidence = build_evidence(results)

    assert isinstance(evidence, tuple)
    assert [item.citation_id for item in evidence] == ["S1", "S2", "S3"]
    assert [item.chunk_id for item in evidence] == [
        "chunk-2",
        "chunk-0",
        "chunk-1",
    ]
    assert evidence[0].source == "/notes/2.md"
    assert evidence[0].heading_path == ("RAG", "Section 2")
    assert evidence[0].text == "evidence 2"
    assert evidence[0].retrieval_score == 0.9


def test_build_evidence_returns_empty_tuple_for_no_results() -> None:
    assert build_evidence([]) == ()

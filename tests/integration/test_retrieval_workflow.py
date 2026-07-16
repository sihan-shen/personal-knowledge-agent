"""End-to-end relevance checks using fixed bilingual knowledge fixtures."""

from pathlib import Path

import pytest

from personal_knowledge_agent.application.ingest import IngestService
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.models import Chunk, Document, SearchMode
from personal_knowledge_agent.infrastructure.markdown_chunker import MarkdownChunker
from personal_knowledge_agent.infrastructure.markdown_loader import MarkdownLoader
from personal_knowledge_agent.infrastructure.sqlite_keyword_index import (
    SQLiteKeywordIndex,
)
from tests.helpers import FakeEmbedder

FIXTURES = Path(__file__).parents[1] / "fixtures" / "knowledge"


def make_index(database: Path) -> SQLiteKeywordIndex:
    return SQLiteKeywordIndex(
        database,
        FakeEmbedder({}, default_vector=(1.0, 0.0)),
    )


@pytest.fixture
def populated_database(tmp_path: Path) -> Path:
    database = tmp_path / "knowledge.db"
    index = make_index(database)
    service = IngestService(MarkdownLoader(), MarkdownChunker(), index)
    for name in ("rag_notes.md", "python_agents.md", "unrelated_notes.md"):
        service.ingest(FIXTURES / name)
    return database


@pytest.mark.parametrize(
    ("query", "expected_source"),
    [
        ("检索增强生成", "rag_notes.md"),
        ("retrieval context", "rag_notes.md"),
        ("Agent 工具 decision", "python_agents.md"),
    ],
)
def test_bilingual_queries_rank_relevant_source_first(
    populated_database: Path,
    query: str,
    expected_source: str,
) -> None:
    service = SearchService(make_index(populated_database))

    results = service.search(query)

    assert Path(results[0].source).name == expected_source
    assert results[0].score > 0
    assert results[0].chunk.heading_path


def test_hybrid_keeps_semantic_synonym_and_exact_keyword_candidates(
    tmp_path: Path,
) -> None:
    database = tmp_path / "knowledge.db"
    document = Document("doc", "/notes/hybrid.md", "hybrid")
    semantic_chunk = Chunk(
        "semantic",
        document.document_id,
        0,
        "Ground answers with external knowledge.",
        ("RAG",),
    )
    lexical_chunk = Chunk(
        "lexical",
        document.document_id,
        1,
        "Exact keyword retrieval.",
        ("RAG",),
    )
    embedder = FakeEmbedder(
        {
            "RAG\nGround answers with external knowledge.": (1.0, 0.0),
            "RAG\nExact keyword retrieval.": (0.0, 1.0),
            "减少模型幻觉": (1.0, 0.0),
            "retrieval 减少模型幻觉": (1.0, 0.0),
        }
    )
    index = SQLiteKeywordIndex(database, embedder)
    index.replace_document(document, [semantic_chunk, lexical_chunk])

    assert index.search("减少模型幻觉", mode=SearchMode.LEXICAL) == []
    semantic = index.search("减少模型幻觉", mode=SearchMode.SEMANTIC)
    hybrid = index.search("retrieval 减少模型幻觉", mode=SearchMode.HYBRID)

    assert semantic[0].chunk.chunk_id == semantic_chunk.chunk_id
    assert {result.chunk.chunk_id for result in hybrid} == {
        semantic_chunk.chunk_id,
        lexical_chunk.chunk_id,
    }

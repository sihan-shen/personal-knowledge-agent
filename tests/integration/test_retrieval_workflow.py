"""End-to-end relevance checks using fixed bilingual knowledge fixtures."""

from pathlib import Path

import pytest

from personal_knowledge_agent.application.ingest import IngestService
from personal_knowledge_agent.application.search import SearchService
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

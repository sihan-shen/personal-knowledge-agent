"""Black-box tests for the command-line interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the package module as a user would invoke the CLI."""
    return subprocess.run(
        [sys.executable, "-m", "personal_knowledge_agent.cli", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_health_command_reports_healthy_project() -> None:
    result = run_cli("health")

    assert result.returncode == 0
    assert "Personal Knowledge Agent is healthy." in result.stdout


def test_unknown_command_returns_usage_error() -> None:
    result = run_cli("unknown")

    assert result.returncode != 0
    assert "usage:" in result.stderr.lower()


def test_ingest_and_search_persist_across_processes(tmp_path: Path) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text(
        "# RAG\n\n## Retrieval\n\nRAG 使用检索增强生成来选择 relevant evidence.",
        encoding="utf-8",
    )

    ingested = run_cli("ingest", str(document), "--db", str(database))
    searched = run_cli(
        "search",
        "RAG 检索 evidence",
        "--db",
        str(database),
    )

    assert ingested.returncode == 0
    assert "Indexed 1 chunks" in ingested.stdout
    assert searched.returncode == 0
    assert "[score=" in searched.stdout
    assert str(document.resolve()) in searched.stdout
    assert "RAG > Retrieval" in searched.stdout


def test_reingest_reports_replacement(tmp_path: Path) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text("# RAG\n\nretrieval", encoding="utf-8")
    first = run_cli("ingest", str(document), "--db", str(database))

    second = run_cli("ingest", str(document), "--db", str(database))

    assert first.returncode == 0
    assert second.returncode == 0
    assert "Reindexed 1 chunks" in second.stdout


def test_search_reports_no_matches_as_success(tmp_path: Path) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text("# RAG\n\nretrieval", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(database))

    result = run_cli("search", "gardening", "--db", str(database))

    assert result.returncode == 0
    assert result.stdout.strip() == "No relevant chunks found."


def test_search_empty_database_returns_operational_error(tmp_path: Path) -> None:
    result = run_cli(
        "search",
        "rag",
        "--db",
        str(tmp_path / "empty.db"),
    )

    assert result.returncode == 1
    assert "error: Knowledge base is empty" in result.stderr


def test_ingest_invalid_path_returns_operational_error(tmp_path: Path) -> None:
    result = run_cli(
        "ingest",
        str(tmp_path / "missing.md"),
        "--db",
        str(tmp_path / "knowledge.db"),
    )

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "not a readable file" in result.stderr


def test_search_rejects_limit_outside_cli_range(tmp_path: Path) -> None:
    result = run_cli(
        "search",
        "rag",
        "--limit",
        "0",
        "--db",
        str(tmp_path / "knowledge.db"),
    )

    assert result.returncode == 2
    assert "between 1 and 20" in result.stderr

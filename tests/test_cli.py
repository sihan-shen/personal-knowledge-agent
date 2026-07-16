"""Behavioral tests for the command-line interface."""

from __future__ import annotations

import os
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from personal_knowledge_agent.cli import build_parser, main
from personal_knowledge_agent.domain.errors import LanguageModelError
from personal_knowledge_agent.domain.models import ModelOutput
from personal_knowledge_agent.domain.protocols import LanguageModel
from tests.helpers import FakeEmbedder, FakeLanguageModel, ScriptedLanguageModel


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


def run_cli(
    *arguments: str,
    language_model: LanguageModel | None = None,
    environment: dict[str, str] | None = None,
) -> CliResult:
    """Run the CLI with deterministic embeddings and capture its process contract."""
    stdout = StringIO()
    stderr = StringIO()
    returncode = 0
    embedder = FakeEmbedder(
        {
            "RAG 检索 evidence": (1.0, 0.0),
            "gardening": (0.0, 1.0),
            "rag": (1.0, 0.0),
        },
        default_vector=(1.0, 0.0),
    )
    model = language_model or FakeLanguageModel(
        ModelOutput(
            "fake-provider-model",
            '{"status":"answered","answer":"Grounded answer. [S1]",'
            '"citation_ids":["S1"]}',
            "stop",
        )
    )
    settings = {"PKA_LLM_API_KEY": "test-key"} if environment is None else environment
    with (
        patch(
            "personal_knowledge_agent.cli.E5Embedder",
            return_value=embedder,
        ),
        patch(
            "personal_knowledge_agent.cli.OpenAICompatibleChatModel",
            return_value=model,
        ),
        patch.dict(os.environ, settings, clear=True),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        try:
            main(arguments)
        except SystemExit as error:
            returncode = int(error.code or 0)
    return CliResult(returncode, stdout.getvalue(), stderr.getvalue())


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
    assert "Search mode: hybrid" in searched.stdout
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
    assert result.stdout.splitlines() == [
        "Search mode: hybrid",
        "No relevant chunks found.",
    ]


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


def test_search_rejects_unknown_mode() -> None:
    result = run_cli("search", "rag", "--mode", "vector")

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert "lexical" in result.stderr
    assert "semantic" in result.stderr
    assert "hybrid" in result.stderr


def test_search_reports_explicit_mode(tmp_path: Path) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text("# RAG\n\nretrieval", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(database))

    result = run_cli(
        "search",
        "rag",
        "--mode",
        "semantic",
        "--db",
        str(database),
    )

    assert result.returncode == 0
    assert result.stdout.startswith("Search mode: semantic\n")


def test_ask_parser_uses_safe_defaults() -> None:
    parsed = build_parser().parse_args(["ask", "question"])

    assert parsed.limit == 5
    assert parsed.mode == "hybrid"


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_ask_parser_accepts_each_search_mode(mode: str) -> None:
    parsed = build_parser().parse_args(["ask", "question", "--mode", mode])

    assert parsed.mode == mode


@pytest.mark.parametrize("limit", ["1", "10"])
def test_ask_parser_accepts_cost_boundary(limit: str) -> None:
    parsed = build_parser().parse_args(["ask", "question", "--limit", limit])

    assert parsed.limit == int(limit)


def test_search_parser_still_accepts_twenty_results() -> None:
    parsed = build_parser().parse_args(["search", "question", "--limit", "20"])

    assert parsed.limit == 20


def test_ask_returns_grounded_answer_and_only_cited_local_source(
    tmp_path: Path,
) -> None:
    first = tmp_path / "rag.md"
    second = tmp_path / "other.md"
    database = tmp_path / "knowledge.db"
    first.write_text("# RAG\n\nretrieval evidence", encoding="utf-8")
    second.write_text("# Other\n\nother evidence", encoding="utf-8")
    run_cli("ingest", str(first), "--db", str(database))
    run_cli("ingest", str(second), "--db", str(database))

    result = run_cli(
        "ask",
        "How does RAG work?",
        "--db",
        str(database),
    )

    assert result.returncode == 0
    assert result.stdout.startswith("Answer:\nGrounded answer. [S1]\n")
    assert "\nSources:\n[S1] " in result.stdout
    assert result.stdout.count("\n[S") == 1
    assert "fake-provider-model" not in result.stdout


def test_ask_empty_retrieval_returns_local_refusal_without_model_call(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text("# RAG\n\nretrieval", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(database))
    model = FakeLanguageModel(RuntimeError("must not be called"))

    result = run_cli(
        "ask",
        "gardening",
        "--mode",
        "lexical",
        "--db",
        str(database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("Insufficient evidence:\n")
    assert "Sources:" not in result.stdout
    assert model.calls == []


def test_ask_displays_model_declared_insufficient_evidence(tmp_path: Path) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text("# RAG\n\nretrieval", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(database))
    model = FakeLanguageModel(
        ModelOutput(
            "fake-model",
            '{"status":"insufficient_evidence",'
            '"answer":"The evidence is insufficient.","citation_ids":[]}',
            "stop",
        )
    )

    result = run_cli(
        "ask",
        "question",
        "--db",
        str(database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout == "Insufficient evidence:\nThe evidence is insufficient.\n"


def test_ask_requires_api_key_without_exposing_traceback(tmp_path: Path) -> None:
    result = run_cli(
        "ask",
        "question",
        "--db",
        str(tmp_path / "knowledge.db"),
        environment={},
    )

    assert result.returncode == 1
    assert "PKA_LLM_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("limit", ["0", "11"])
def test_ask_rejects_limit_outside_cost_boundary(limit: str) -> None:
    result = run_cli("ask", "question", "--limit", limit)

    assert result.returncode == 2
    assert "between 1 and 10" in result.stderr


def test_ask_rejects_unknown_mode() -> None:
    result = run_cli("ask", "question", "--mode", "vector")

    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_agent_parser_uses_safe_defaults() -> None:
    parsed = build_parser().parse_args(["agent", "question"])

    assert parsed.limit == 5
    assert parsed.mode == "hybrid"


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_agent_parser_accepts_each_search_mode(mode: str) -> None:
    parsed = build_parser().parse_args(["agent", "question", "--mode", mode])

    assert parsed.mode == mode


@pytest.mark.parametrize("limit", ["1", "10"])
def test_agent_parser_accepts_cost_boundary(limit: str) -> None:
    parsed = build_parser().parse_args(["agent", "question", "--limit", limit])

    assert parsed.limit == int(limit)


@pytest.mark.parametrize("limit", ["0", "11"])
def test_agent_rejects_limit_outside_cost_boundary(limit: str) -> None:
    result = run_cli("agent", "question", "--limit", limit)

    assert result.returncode == 2
    assert "between 1 and 10" in result.stderr


def test_agent_direct_rule_displays_visible_route_without_model_call(
    tmp_path: Path,
) -> None:
    model = FakeLanguageModel(RuntimeError("must not be called"))

    result = run_cli(
        "agent",
        "你好",
        "--db",
        str(tmp_path / "knowledge.db"),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout == (
        "Route: direct (greeting, rule)\n"
        "你好。我可以在需要时检索你的本地知识库，并用引用回答问题。\n"
    )
    assert model.calls == []


def test_agent_llm_retrieve_displays_route_then_grounded_answer(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text("# RAG\n\nretrieval evidence", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(database))
    model = ScriptedLanguageModel(
        [
            ModelOutput(
                "router",
                '{"action":"retrieve","direct_intent":null}',
                "stop",
            ),
            ModelOutput(
                "answer",
                '{"status":"answered","answer":"Grounded. [S1]","citation_ids":["S1"]}',
                "stop",
            ),
        ]
    )

    result = run_cli(
        "agent",
        "How does RAG work?",
        "--db",
        str(database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("Route: retrieve (llm)\nAnswer:\nGrounded. [S1]\n")
    assert "\nSources:\n[S1] " in result.stdout
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    ("router_result", "route_line"),
    [
        (
            ModelOutput("router", "not JSON", "stop"),
            "Route: retrieve (fallback: invalid router output)\n",
        ),
        (
            LanguageModelError("provider-body-secret-123"),
            "Route: retrieve (fallback: provider error)\n",
        ),
    ],
)
def test_agent_router_failure_displays_safe_fallback_and_retrieves(
    tmp_path: Path,
    router_result: ModelOutput | Exception,
    route_line: str,
) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text("# RAG\n\nretrieval evidence", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(database))
    model = ScriptedLanguageModel(
        [
            router_result,
            ModelOutput(
                "answer",
                '{"status":"answered","answer":"Grounded. [S1]","citation_ids":["S1"]}',
                "stop",
            ),
        ]
    )

    result = run_cli(
        "agent",
        "How does RAG work?",
        "--db",
        str(database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout.startswith(route_line + "Answer:\nGrounded. [S1]\n")
    assert "provider-body-secret-123" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_agent_requires_api_key_without_exposing_traceback(tmp_path: Path) -> None:
    result = run_cli(
        "agent",
        "hello",
        "--db",
        str(tmp_path / "knowledge.db"),
        environment={},
    )

    assert result.returncode == 1
    assert "PKA_LLM_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            FakeLanguageModel(
                LanguageModelError("Language-model provider authentication failed")
            ),
            "authentication failed",
        ),
        (
            FakeLanguageModel(ModelOutput("fake-model", "not JSON", "stop")),
            "valid JSON",
        ),
        (
            FakeLanguageModel(
                ModelOutput(
                    "fake-model",
                    '{"status":"answered","answer":"Unsupported [S99]",'
                    '"citation_ids":["S99"]}',
                    "stop",
                )
            ),
            "unknown citation",
        ),
    ],
)
def test_ask_reports_provider_and_validation_failures_without_traceback(
    tmp_path: Path,
    model: FakeLanguageModel,
    message: str,
) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text("# RAG\n\nretrieval", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(database))

    result = run_cli(
        "ask",
        "question",
        "--db",
        str(database),
        language_model=model,
    )

    assert result.returncode == 1
    assert result.stderr.startswith("error: ")
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert "test-key" not in result.stderr

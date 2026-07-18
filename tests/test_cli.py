"""Behavioral tests for the command-line interface."""

from __future__ import annotations

import os
import re
import shlex
import sqlite3
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from personal_knowledge_agent.cli import build_parser, main
from personal_knowledge_agent.domain.errors import LanguageModelError
from personal_knowledge_agent.domain.memory import (
    MemoryProposal,
    MemoryType,
    memory_source_hash,
)
from personal_knowledge_agent.domain.models import ModelOutput
from personal_knowledge_agent.domain.protocols import LanguageModel
from personal_knowledge_agent.infrastructure.sqlite_memory_store import (
    SQLiteMemoryStore,
)
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


def seed_pending_memory(
    database: Path,
    memory_type: MemoryType,
    memory_key: str,
    content: str,
) -> None:
    """Seed one real pending row without exposing its generated ID to a CLI test."""
    store = SQLiteMemoryStore(
        database,
        FakeEmbedder({}, default_vector=(1.0, 0.0)),
    )
    store.create_pending(
        (MemoryProposal(memory_type, memory_key, content),),
        source_hash=memory_source_hash(f"source for {memory_key}"),
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


def test_memory_parser_requires_a_nested_command() -> None:
    result = run_cli("memory")

    assert result.returncode == 2
    assert "usage:" in result.stderr.lower()
    assert "memory_command" in result.stderr
    assert "remember" in result.stderr
    assert "forget" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "action"),
    [
        (
            [
                "memory",
                "remember",
                "--type",
                "goal",
                "--key",
                "goal.ai_engineer",
                "Become an AI Engineer.",
            ],
            "remember",
        ),
        (["memory", "list"], "list"),
        (["memory", "search", "AI Engineer"], "search"),
        (["memory", "approve", "mem_" + "a" * 32], "approve"),
        (["memory", "reject", "mem_" + "a" * 32], "reject"),
        (["memory", "forget", "mem_" + "a" * 32], "forget"),
    ],
)
def test_memory_parser_accepts_each_nested_command(
    arguments: list[str],
    action: str,
) -> None:
    parsed = build_parser().parse_args(arguments)

    assert parsed.command == "memory"
    assert parsed.memory_command == action
    assert parsed.memory_db == Path(".pka/memory.db")


@pytest.mark.parametrize(
    "memory_type",
    ["preference", "goal", "project", "learning_progress"],
)
def test_memory_parser_accepts_each_exact_memory_type(memory_type: str) -> None:
    remember = build_parser().parse_args(
        [
            "memory",
            "remember",
            "--type",
            memory_type,
            "--key",
            "project.pka",
            "content",
        ]
    )
    listed = build_parser().parse_args(["memory", "list", "--type", memory_type])
    searched = build_parser().parse_args(
        ["memory", "search", "query", "--type", memory_type]
    )

    assert remember.memory_type == memory_type
    assert listed.memory_type == memory_type
    assert searched.memory_type == memory_type


@pytest.mark.parametrize("status", ["pending", "active", "superseded"])
def test_memory_list_parser_accepts_each_exact_status(status: str) -> None:
    parsed = build_parser().parse_args(["memory", "list", "--status", status])

    assert parsed.status == status


@pytest.mark.parametrize(
    ("arguments", "expected_choice"),
    [
        (
            [
                "memory",
                "remember",
                "--type",
                "other",
                "--key",
                "key",
                "content",
            ],
            "learning_progress",
        ),
        (["memory", "list", "--type", "other"], "learning_progress"),
        (["memory", "list", "--status", "rejected"], "superseded"),
        (
            ["memory", "search", "query", "--type", "other"],
            "learning_progress",
        ),
    ],
)
def test_memory_parser_rejects_values_outside_exact_enum_choices(
    arguments: list[str],
    expected_choice: str,
) -> None:
    result = run_cli(*arguments)

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert expected_choice in result.stderr


@pytest.mark.parametrize("limit", ["1", "20"])
def test_memory_search_parser_accepts_limit_boundary(limit: str) -> None:
    parsed = build_parser().parse_args(["memory", "search", "query", "--limit", limit])

    assert parsed.limit == int(limit)


@pytest.mark.parametrize("limit", ["0", "21"])
def test_memory_search_parser_rejects_limit_outside_boundary(limit: str) -> None:
    result = run_cli("memory", "search", "query", "--limit", limit)

    assert result.returncode == 2
    assert "between 1 and 20" in result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ["memory", "remember", "--key", "goal.ai_engineer", "content"],
        ["memory", "remember", "--type", "goal", "content"],
        ["memory", "remember", "--type", "goal", "--key", "goal.ai_engineer"],
        ["memory", "search"],
        ["memory", "approve"],
        ["memory", "reject"],
        ["memory", "forget"],
    ],
)
def test_memory_parser_requires_command_inputs(arguments: list[str]) -> None:
    result = run_cli(*arguments)

    assert result.returncode == 2
    assert "required" in result.stderr


@pytest.mark.parametrize("action", ["remember", "approve"])
def test_memory_activation_flags_are_mutually_exclusive(action: str) -> None:
    memory_id = "mem_" + "a" * 32
    if action == "remember":
        arguments = [
            "memory",
            action,
            "--type",
            "goal",
            "--key",
            "goal.ai_engineer",
            "--replace",
            memory_id,
            "--keep-both",
            "content",
        ]
    else:
        arguments = [
            "memory",
            action,
            memory_id,
            "--replace",
            memory_id,
            "--keep-both",
        ]

    result = run_cli(*arguments)

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_memory_parser_accepts_explicit_database_and_activation_actions(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    memory_id = "mem_" + "a" * 32
    replace = build_parser().parse_args(
        [
            "memory",
            "remember",
            "--type",
            "goal",
            "--key",
            "goal.ai_engineer",
            "--replace",
            memory_id,
            "content",
            "--memory-db",
            str(database),
        ]
    )
    keep_both = build_parser().parse_args(
        [
            "memory",
            "approve",
            memory_id,
            "--keep-both",
            "--memory-db",
            str(database),
        ]
    )

    assert replace.replace_id == memory_id
    assert replace.keep_both is False
    assert replace.memory_db == database
    assert keep_both.replace_id is None
    assert keep_both.keep_both is True
    assert keep_both.memory_db == database


def test_existing_database_defaults_remain_unchanged() -> None:
    parser = build_parser()

    assert parser.parse_args(["ingest", "note.md"]).db == Path(".pka/knowledge.db")
    assert parser.parse_args(["search", "query"]).db == Path(".pka/knowledge.db")
    assert parser.parse_args(["ask", "question"]).db == Path(".pka/knowledge.db")
    assert parser.parse_args(["agent", "question"]).db == Path(".pka/knowledge.db")


def test_memory_remember_list_and_search_persist_without_language_model(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    model = FakeLanguageModel(RuntimeError("must not be called"))

    remembered = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.ai_engineer",
        "Become an AI Engineer through a RAG portfolio.",
        "--memory-db",
        str(database),
        language_model=model,
        environment={},
    )
    listed = run_cli(
        "memory",
        "list",
        "--status",
        "active",
        "--type",
        "goal",
        "--memory-db",
        str(database),
        language_model=model,
        environment={},
    )
    searched = run_cli(
        "memory",
        "search",
        "AI Engineer RAG",
        "--type",
        "goal",
        "--memory-db",
        str(database),
        language_model=model,
        environment={},
    )

    identifier = re.search(r"\[(mem_[0-9a-f]{32})\]", remembered.stdout)
    assert remembered.returncode == 0
    assert identifier is not None
    memory_id = identifier.group(1)
    assert remembered.stdout == (
        f"Remembered [{memory_id}] active goal goal.ai_engineer\n"
        "  Become an AI Engineer through a RAG portfolio.\n"
    )
    assert listed.returncode == 0
    assert listed.stdout == (
        f"[{memory_id}] active goal goal.ai_engineer\n"
        "  Become an AI Engineer through a RAG portfolio.\n"
    )
    assert searched.returncode == 0
    assert re.fullmatch(
        rf"1\. \[score=0\.\d{{3}}\] \[{memory_id}\] goal goal\.ai_engineer\n"
        r"  Become an AI Engineer through a RAG portfolio\.\n",
        searched.stdout,
    )
    combined_output = remembered.stdout + listed.stdout + searched.stdout
    assert "hash" not in combined_output.casefold()
    assert "vector" not in combined_output.casefold()
    assert "provider" not in combined_output.casefold()
    assert model.calls == []


def test_memory_search_empty_database_is_success_without_model_settings(
    tmp_path: Path,
) -> None:
    result = run_cli(
        "memory",
        "search",
        "AI Engineer",
        "--memory-db",
        str(tmp_path / "memory.db"),
        environment={},
    )

    assert result.returncode == 0
    assert result.stdout == "No active memories found.\n"


def test_memory_approve_and_reject_pending_candidates_across_processes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    seed_pending_memory(
        database,
        MemoryType.GOAL,
        "goal.ai_engineer",
        "Become an AI Engineer.",
    )
    first_list = run_cli(
        "memory",
        "list",
        "--status",
        "pending",
        "--memory-db",
        str(database),
    )
    first_match = re.search(r"\[(mem_[0-9a-f]{32})\]", first_list.stdout)
    assert first_match is not None
    approved_id = first_match.group(1)

    approved = run_cli(
        "memory",
        "approve",
        approved_id,
        "--memory-db",
        str(database),
        environment={},
    )
    active_list = run_cli(
        "memory",
        "list",
        "--status",
        "active",
        "--memory-db",
        str(database),
    )

    assert approved.returncode == 0
    assert approved.stdout == f"Approved [{approved_id}] active\n"
    assert f"[{approved_id}] active goal goal.ai_engineer" in active_list.stdout

    seed_pending_memory(
        database,
        MemoryType.PROJECT,
        "project.rag",
        "Private candidate content to remove.",
    )
    second_list = run_cli(
        "memory",
        "list",
        "--status",
        "pending",
        "--memory-db",
        str(database),
    )
    second_match = re.search(r"\[(mem_[0-9a-f]{32})\]", second_list.stdout)
    assert second_match is not None
    rejected_id = second_match.group(1)

    rejected = run_cli(
        "memory",
        "reject",
        rejected_id,
        "--memory-db",
        str(database),
        environment={},
    )
    pending_after_rejection = run_cli(
        "memory",
        "list",
        "--status",
        "pending",
        "--memory-db",
        str(database),
    )

    assert rejected.returncode == 0
    assert rejected.stdout == f"Rejected [{rejected_id}]\n"
    assert pending_after_rejection.returncode == 0
    assert "Private candidate content to remove." not in pending_after_rejection.stdout
    assert rejected_id not in pending_after_rejection.stdout
    with sqlite3.connect(database) as connection:
        remaining_rows = connection.execute(
            "SELECT content FROM memories WHERE memory_id = ?",
            (rejected_id,),
        ).fetchall()
    assert remaining_rows == []


def test_memory_replace_keep_both_and_forget_chain_across_processes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    first = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.ai_engineer",
        "Build a portfolio-ready agent.",
        "--memory-db",
        str(database),
    )
    first_match = re.search(r"\[(mem_[0-9a-f]{32})\]", first.stdout)
    assert first_match is not None
    first_id = first_match.group(1)

    replacement = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.ai_engineer",
        "--replace",
        first_id,
        "Build a production-ready agent.",
        "--memory-db",
        str(database),
    )
    replacement_match = re.search(
        r"\[(mem_[0-9a-f]{32})\]",
        replacement.stdout,
    )
    assert replacement.returncode == 0
    assert replacement_match is not None
    replacement_id = replacement_match.group(1)

    after_replacement = run_cli(
        "memory",
        "list",
        "--memory-db",
        str(database),
    )
    assert f"[{first_id}] superseded goal goal.ai_engineer" in (
        after_replacement.stdout
    )
    assert f"[{replacement_id}] active goal goal.ai_engineer" in (
        after_replacement.stdout
    )

    kept = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.ai_engineer",
        "--keep-both",
        "Publish the agent as open source.",
        "--memory-db",
        str(database),
    )
    kept_match = re.search(r"\[(mem_[0-9a-f]{32})\]", kept.stdout)
    assert kept.returncode == 0
    assert kept_match is not None
    kept_id = kept_match.group(1)

    active = run_cli(
        "memory",
        "list",
        "--status",
        "active",
        "--memory-db",
        str(database),
    )
    assert f"[{replacement_id}] active goal goal.ai_engineer" in active.stdout
    assert f"[{kept_id}] active goal goal.ai_engineer" in active.stdout
    assert active.stdout.count(" active goal goal.ai_engineer") == 2

    forgotten = run_cli(
        "memory",
        "forget",
        replacement_id,
        "--memory-db",
        str(database),
    )
    remaining = run_cli(
        "memory",
        "list",
        "--memory-db",
        str(database),
    )

    assert forgotten.returncode == 0
    assert forgotten.stdout == "Forgot 2 memory records.\n"
    assert first_id not in remaining.stdout
    assert replacement_id not in remaining.stdout
    assert f"[{kept_id}] active goal goal.ai_engineer" in remaining.stdout


def test_memory_bare_conflicts_report_ids_and_actionable_follow_up_commands(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory db's" / "custom memory.db"
    quoted_database = shlex.quote(str(database))
    first = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.ai_engineer",
        "Become an AI Engineer.",
        "--memory-db",
        str(database),
    )
    active_match = re.search(r"\[(mem_[0-9a-f]{32})\]", first.stdout)
    assert active_match is not None
    active_id = active_match.group(1)

    remember_conflict = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.ai_engineer",
        "Build a second AI portfolio.",
        "--memory-db",
        str(database),
    )

    assert remember_conflict.returncode == 1
    assert f"Active IDs: {active_id}" in remember_conflict.stderr
    assert (
        "pka memory remember --type goal --key goal.ai_engineer "
        f"--replace {active_id} --memory-db {quoted_database} CONTENT"
    ) in remember_conflict.stderr
    assert (
        "pka memory remember --type goal --key goal.ai_engineer "
        f"--keep-both --memory-db {quoted_database} CONTENT"
    ) in remember_conflict.stderr
    assert "Build a second AI portfolio." not in remember_conflict.stderr
    assert "Traceback" not in remember_conflict.stderr

    seed_pending_memory(
        database,
        MemoryType.GOAL,
        "goal.ai_engineer",
        "Pending AI Engineer candidate.",
    )
    pending = run_cli(
        "memory",
        "list",
        "--status",
        "pending",
        "--memory-db",
        str(database),
    )
    candidate_match = re.search(r"\[(mem_[0-9a-f]{32})\]", pending.stdout)
    assert candidate_match is not None
    candidate_id = candidate_match.group(1)

    approval_conflict = run_cli(
        "memory",
        "approve",
        candidate_id,
        "--memory-db",
        str(database),
    )
    pending_after = run_cli(
        "memory",
        "list",
        "--status",
        "pending",
        "--memory-db",
        str(database),
    )
    active_search = run_cli(
        "memory",
        "search",
        "Pending AI Engineer candidate",
        "--memory-db",
        str(database),
    )

    assert approval_conflict.returncode == 1
    assert f"Active IDs: {active_id}" in approval_conflict.stderr
    assert (
        f"pka memory approve {candidate_id} --replace {active_id} "
        f"--memory-db {quoted_database}" in approval_conflict.stderr
    )
    assert (
        f"pka memory approve {candidate_id} --keep-both --memory-db {quoted_database}"
    ) in approval_conflict.stderr
    assert "Pending AI Engineer candidate." not in approval_conflict.stderr
    assert "Traceback" not in approval_conflict.stderr
    assert candidate_id in pending_after.stdout
    assert "Pending AI Engineer candidate." in pending_after.stdout
    assert candidate_id not in active_search.stdout
    assert "Pending AI Engineer candidate." not in active_search.stdout


def test_memory_conflict_guidance_keeps_default_database_path_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    first = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.ai_engineer",
        "Become an AI Engineer.",
    )
    active_match = re.search(r"\[(mem_[0-9a-f]{32})\]", first.stdout)
    assert active_match is not None
    active_id = active_match.group(1)

    conflict = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.ai_engineer",
        "Private second goal.",
    )

    assert conflict.returncode == 1
    assert (
        "pka memory remember --type goal --key goal.ai_engineer "
        f"--replace {active_id} --memory-db .pka/memory.db CONTENT"
    ) in conflict.stderr
    assert (
        "pka memory remember --type goal --key goal.ai_engineer "
        "--keep-both --memory-db .pka/memory.db CONTENT"
    ) in conflict.stderr
    assert "Private second goal." not in conflict.stderr


def test_memory_invalid_unsafe_unknown_and_conflicting_operations_are_safe(
    tmp_path: Path,
) -> None:
    unknown_id = "mem_" + "f" * 32
    cases = [
        (
            "invalid-key",
            [
                "memory",
                "remember",
                "--type",
                "goal",
                "--key",
                "INVALID KEY",
                "content",
            ],
        ),
        (
            "unsafe-content",
            [
                "memory",
                "remember",
                "--type",
                "goal",
                "--key",
                "goal.secret",
                "api_key=do-not-store-this",
            ],
        ),
        ("unknown-approve", ["memory", "approve", unknown_id]),
        ("unknown-reject", ["memory", "reject", unknown_id]),
        ("unknown-forget", ["memory", "forget", unknown_id]),
        (
            "keep-both-without-conflict",
            [
                "memory",
                "remember",
                "--type",
                "goal",
                "--key",
                "goal.empty",
                "--keep-both",
                "content",
            ],
        ),
        (
            "replace-unknown",
            [
                "memory",
                "remember",
                "--type",
                "goal",
                "--key",
                "goal.empty",
                "--replace",
                unknown_id,
                "content",
            ],
        ),
    ]

    for case_name, arguments in cases:
        result = run_cli(
            *arguments,
            "--memory-db",
            str(tmp_path / f"{case_name}.db"),
            environment={},
        )

        assert result.returncode == 1, case_name
        assert result.stderr.startswith("error: "), case_name
        assert "Traceback" not in result.stderr, case_name
        assert "do-not-store-this" not in result.stderr, case_name


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


def test_agent_llm_direct_displays_route_and_local_template(tmp_path: Path) -> None:
    model = ScriptedLanguageModel(
        [
            ModelOutput(
                "router",
                '{"action":"direct","direct_intent":"capabilities"}',
                "stop",
            )
        ]
    )

    result = run_cli(
        "agent",
        "Tell me how you can help",
        "--db",
        str(tmp_path / "knowledge.db"),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout == (
        "Route: direct (capabilities, llm)\n"
        "I can search your local knowledge base, answer from retrieved evidence, "
        "and show cited sources.\n"
    )
    assert len(model.calls) == 1


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


@pytest.mark.parametrize(
    ("router_result", "route_line"),
    [
        (
            ModelOutput(
                "router",
                '{"action":"retrieve","direct_intent":null}',
                "stop",
            ),
            "Route: retrieve (llm)\n",
        ),
        (
            LanguageModelError("router failed"),
            "Route: retrieve (fallback: provider error)\n",
        ),
    ],
)
def test_agent_keeps_route_visible_when_grounded_generation_fails(
    tmp_path: Path,
    router_result: ModelOutput | Exception,
    route_line: str,
) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text("# RAG\n\nretrieval evidence", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(database))
    model = ScriptedLanguageModel(
        [router_result, LanguageModelError("answer generation failed")]
    )

    result = run_cli(
        "agent",
        "How does RAG work?",
        "--db",
        str(database),
        language_model=model,
    )

    assert result.returncode == 1
    assert result.stdout == route_line
    assert "answer generation failed" in result.stderr
    assert "Traceback" not in result.stderr


def test_agent_keeps_route_visible_when_retrieval_fails(tmp_path: Path) -> None:
    model = ScriptedLanguageModel(
        [
            ModelOutput(
                "router",
                '{"action":"retrieve","direct_intent":null}',
                "stop",
            )
        ]
    )

    result = run_cli(
        "agent",
        "How does RAG work?",
        "--db",
        str(tmp_path / "empty.db"),
        language_model=model,
    )

    assert result.returncode == 1
    assert result.stdout == "Route: retrieve (llm)\n"
    assert "Knowledge base is empty" in result.stderr
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

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
from personal_knowledge_agent.domain.errors import (
    EmbeddingError,
    InvalidMemoryError,
    LanguageModelError,
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryStorageError,
    UnsafeMemoryError,
)
from personal_knowledge_agent.domain.graph import (
    EntityType,
    GraphChunkExtraction,
    GraphEntity,
    GraphRelation,
    GraphRelationEvidence,
    GraphSourceDocument,
    RelationType,
    make_entity_id,
    make_relation_id,
)
from personal_knowledge_agent.domain.memory import (
    MemoryActivationAction,
    MemoryProposal,
    MemoryRecallReport,
    MemoryRecallWarning,
    MemoryRecord,
    MemoryType,
    memory_source_hash,
)
from personal_knowledge_agent.domain.models import Chunk, MessageRole, ModelOutput
from personal_knowledge_agent.domain.protocols import LanguageModel
from personal_knowledge_agent.infrastructure.sqlite_knowledge_graph_store import (
    SQLiteKnowledgeGraphStore,
)
from personal_knowledge_agent.infrastructure.sqlite_memory_store import (
    SQLiteMemoryStore,
)
from tests.helpers import (
    FakeEmbedder,
    FakeLanguageModel,
    FakeMemoryStore,
    ScriptedLanguageModel,
)


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


def run_cli(
    *arguments: str,
    language_model: LanguageModel | None = None,
    embedder: FakeEmbedder | None = None,
    environment: dict[str, str] | None = None,
) -> CliResult:
    """Run the CLI with deterministic embeddings and capture its process contract."""
    stdout = StringIO()
    stderr = StringIO()
    returncode = 0
    selected_embedder = embedder or FakeEmbedder(
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
            return_value=selected_embedder,
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


def seed_active_memory(
    database: Path,
    memory_type: MemoryType,
    memory_key: str,
    content: str,
) -> MemoryRecord:
    """Seed one confirmed memory through the real persistence boundary."""
    return SQLiteMemoryStore(
        database,
        FakeEmbedder({}, default_vector=(1.0, 0.0)),
    ).remember(memory_type, memory_key, content)


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
    assert ingested.stdout == f"Indexed 1 chunks from {document.resolve()}\n"
    assert ingested.stderr == ""
    assert searched.returncode == 0
    assert searched.stdout == (
        "Search mode: hybrid\n"
        f"1. [score=0.033] {document.resolve()} > RAG > Retrieval\n"
        "   RAG 使用检索增强生成来选择 relevant evidence.\n"
    )
    assert searched.stderr == ""


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


def test_memory_pending_list_displays_full_content_before_approval(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    sentinel = "TAIL-SENTINEL"
    content = "x" * (500 - len(sentinel)) + sentinel
    seed_pending_memory(database, MemoryType.GOAL, "goal.long", content)

    pending = run_cli(
        "memory",
        "list",
        "--status",
        "pending",
        "--memory-db",
        str(database),
    )
    identifier = re.search(r"\[(mem_[0-9a-f]{32})\]", pending.stdout)
    assert identifier is not None
    approved = run_cli(
        "memory",
        "approve",
        identifier.group(1),
        "--memory-db",
        str(database),
    )

    assert content in pending.stdout
    assert sentinel in pending.stdout
    assert approved.returncode == 0
    assert approved.stdout == f"Approved [{identifier.group(1)}] active\n"


def test_memory_remember_rejects_chinese_secret_without_writing(
    tmp_path: Path,
) -> None:
    database = tmp_path / "memory.db"
    model = FakeLanguageModel(RuntimeError("must not be called"))

    result = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.credentials",
        "我的 API密钥：cn-provider-secret-123456",
        "--memory-db",
        str(database),
        language_model=model,
        environment={},
    )

    assert result.returncode == 1
    assert "unsafe" in result.stderr.casefold()
    assert model.calls == []
    with sqlite3.connect(database) as connection:
        memory_count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
    assert memory_count == (0,)


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
        assert result.stdout == "", case_name
        assert result.stderr.startswith("error: "), case_name
        assert "Traceback" not in result.stderr, case_name
        assert "do-not-store-this" not in result.stdout + result.stderr, case_name


@pytest.mark.parametrize(
    "error",
    [
        InvalidMemoryError("invalid memory"),
        UnsafeMemoryError("unsafe memory"),
        MemoryConflictError("memory conflict"),
        MemoryNotFoundError("memory not found"),
        MemoryStorageError("memory storage failed"),
        EmbeddingError("memory embedding failed"),
    ],
)
def test_every_expected_explicit_memory_error_exits_one_without_traceback(
    tmp_path: Path,
    error: Exception,
) -> None:
    with patch("personal_knowledge_agent.cli._memory_store", side_effect=error):
        result = run_cli(
            "memory",
            "list",
            "--memory-db",
            str(tmp_path / "memory.db"),
            environment={},
        )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"error: {error}\n"
    assert "Traceback" not in result.stderr


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
    assert result.stdout == (
        f"Answer:\nGrounded answer. [S1]\n\nSources:\n[S1] {first.resolve()} > RAG\n"
    )
    assert result.stderr == ""


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
    assert parsed.memory_db == Path(".pka/memory.db")


def test_agent_parser_accepts_explicit_memory_database(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"

    parsed = build_parser().parse_args(
        ["agent", "question", "--memory-db", str(database)]
    )

    assert parsed.memory_db == database


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

    with patch(
        "personal_knowledge_agent.cli._memory_store",
        side_effect=AssertionError("direct route must not construct memory storage"),
    ):
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


def test_agent_answers_from_active_memory_when_knowledge_base_is_empty(
    tmp_path: Path,
) -> None:
    memory_database = tmp_path / "memory.db"
    memory = seed_active_memory(
        memory_database,
        MemoryType.PREFERENCE,
        "answer_style",
        "Prefer concise answers.",
    )
    model = ScriptedLanguageModel(
        [
            ModelOutput("router", '{"action":"retrieve","direct_intent":null}', "stop"),
            ModelOutput(
                "answer",
                '{"status":"answered","answer":"You prefer concise answers. [M1]",'
                '"citation_ids":["M1"]}',
                "stop",
            ),
        ]
    )

    result = run_cli(
        "agent",
        "What answer style do I prefer?",
        "--db",
        str(tmp_path / "knowledge.db"),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout == (
        "Route: retrieve (llm)\n"
        "Answer:\n"
        "You prefer concise answers. [M1]\n"
        "\nMemories used:\n"
        "[M1] preference.answer_style — Prefer concise answers.\n"
    )
    assert result.stderr == ""
    process_output = result.stdout + result.stderr
    for private_value in (
        memory.memory_id,
        memory.content_hash,
        memory.created_at.isoformat(),
        str(memory_database),
        "test-key",
        "0.01987654321",
    ):
        assert private_value not in process_output
    assert "Traceback" not in process_output


def test_agent_renders_document_and_memory_citations_in_separate_sections(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    memory_database = tmp_path / "memory.db"
    document.write_text("# RAG\n\nretrieval evidence", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(knowledge_database))
    seed_active_memory(
        memory_database,
        MemoryType.GOAL,
        "rag_project",
        "Build a portfolio-ready RAG project.",
    )
    model = ScriptedLanguageModel(
        [
            ModelOutput("router", '{"action":"retrieve","direct_intent":null}', "stop"),
            ModelOutput(
                "answer",
                '{"status":"answered","answer":"Use retrieval for the project. '
                '[S1] [M1]","citation_ids":["S1","M1"]}',
                "stop",
            ),
        ]
    )

    result = run_cli(
        "agent",
        "How should I build my RAG project?",
        "--db",
        str(knowledge_database),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout == (
        "Route: retrieve (llm)\n"
        "Answer:\n"
        "Use retrieval for the project. [S1] [M1]\n"
        "\nSources:\n"
        f"[S1] {document.resolve()} > RAG\n"
        "\nMemories used:\n"
        "[M1] goal.rag_project — Build a portfolio-ready RAG project.\n"
    )


def test_agent_does_not_display_supplied_but_uncited_memory(tmp_path: Path) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    memory_database = tmp_path / "memory.db"
    document.write_text("# RAG\n\nretrieval evidence", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(knowledge_database))
    seed_active_memory(
        memory_database,
        MemoryType.PREFERENCE,
        "answer_style",
        "Prefer concise answers.",
    )
    model = ScriptedLanguageModel(
        [
            ModelOutput("router", '{"action":"retrieve","direct_intent":null}', "stop"),
            ModelOutput(
                "answer",
                '{"status":"answered","answer":"Retrieval uses evidence. [S1]",'
                '"citation_ids":["S1"]}',
                "stop",
            ),
        ]
    )

    result = run_cli(
        "agent",
        "How should retrieval answer concisely?",
        "--db",
        str(knowledge_database),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    assert result.returncode == 0
    assert "\nSources:\n[S1] " in result.stdout
    assert "Memories used:" not in result.stdout
    assert "Prefer concise answers." not in result.stdout


@pytest.mark.parametrize(
    ("warning", "label"),
    [
        (MemoryRecallWarning.UNSAFE_MEMORY, "unsafe stored memory"),
        (MemoryRecallWarning.AMBIGUOUS_MEMORY_SLOT, "ambiguous memory slot"),
    ],
)
def test_agent_displays_safe_policy_recall_warning_once(
    tmp_path: Path,
    warning: MemoryRecallWarning,
    label: str,
) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    document.write_text("# RAG\n\nretrieval evidence", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(database))
    model = ScriptedLanguageModel(
        [
            ModelOutput("router", '{"action":"retrieve","direct_intent":null}', "stop"),
            ModelOutput(
                "answer",
                '{"status":"answered","answer":"Grounded. [S1]","citation_ids":["S1"]}',
                "stop",
            ),
        ]
    )

    with patch(
        "personal_knowledge_agent.cli.MemoryRecallService.recall",
        return_value=MemoryRecallReport((), (warning,)),
    ):
        result = run_cli(
            "agent",
            "How does RAG work?",
            "--db",
            str(database),
            language_model=model,
        )

    assert result.returncode == 0
    assert result.stdout.count(f"Memory recall skipped: {label}\n") == 1
    assert result.stdout.startswith(
        f"Route: retrieve (llm)\nMemory recall skipped: {label}\nAnswer:\n"
    )


def test_agent_never_sends_pending_or_superseded_memory_to_answer_model(
    tmp_path: Path,
) -> None:
    pending_sentinel = "PENDING-MEMORY-PRIVATE"
    superseded_sentinel = "SUPERSEDED-MEMORY-PRIVATE"
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    memory_database = tmp_path / "memory.db"
    document.write_text("# RAG\n\nretrieval evidence", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(knowledge_database))
    store = SQLiteMemoryStore(
        memory_database,
        FakeEmbedder({}, default_vector=(1.0, 0.0)),
    )
    superseded = store.remember(
        MemoryType.GOAL,
        "rag_project",
        superseded_sentinel,
    )
    store.remember(
        MemoryType.GOAL,
        "rag_project",
        "Build the current RAG project.",
        action=MemoryActivationAction.REPLACE,
        replace_id=superseded.memory_id,
    )
    store.create_pending(
        (MemoryProposal(MemoryType.PREFERENCE, "answer_style", pending_sentinel),),
        source_hash=memory_source_hash("lifecycle sentinel source"),
    )
    model = ScriptedLanguageModel(
        [
            ModelOutput("router", '{"action":"retrieve","direct_intent":null}', "stop"),
            ModelOutput(
                "answer",
                '{"status":"answered","answer":"Use retrieval. [S1]",'
                '"citation_ids":["S1"]}',
                "stop",
            ),
        ]
    )

    result = run_cli(
        "agent",
        "How should I build the RAG project?",
        "--db",
        str(knowledge_database),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    answer_request = "\n".join(message.content for message in model.calls[1])
    assert result.returncode == 0
    assert pending_sentinel not in answer_request
    assert superseded_sentinel not in answer_request
    assert pending_sentinel not in result.stdout + result.stderr
    assert superseded_sentinel not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("error", "label"),
    [
        (EmbeddingError("PRIVATE-VECTOR-DETAIL"), "embedding error"),
        (MemoryStorageError("/private/memory.db"), "storage error"),
    ],
)
def test_agent_recall_failure_without_documents_refuses_without_answer_call(
    tmp_path: Path,
    error: Exception,
    label: str,
) -> None:
    model = ScriptedLanguageModel(
        [ModelOutput("router", '{"action":"retrieve","direct_intent":null}', "stop")]
    )

    with patch("personal_knowledge_agent.cli._memory_store", side_effect=error):
        result = run_cli(
            "agent",
            "How does RAG work?",
            "--db",
            str(tmp_path / "knowledge.db"),
            language_model=model,
        )

    assert result.returncode == 0
    assert result.stdout == (
        f"Route: retrieve (llm)\nMemory recall skipped: {label}\n"
        "Insufficient evidence:\n"
        "The knowledge base does not contain enough evidence.\n"
    )
    assert result.stderr == ""
    assert len(model.calls) == 1
    assert "PRIVATE-VECTOR-DETAIL" not in result.stdout + result.stderr
    assert "/private/memory.db" not in result.stdout + result.stderr


def test_agent_prefilter_miss_preserves_phase_2b_output_byte_for_byte(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    database = tmp_path / "knowledge.db"
    memory_database = tmp_path / "memory.db"
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
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout == (
        "Route: retrieve (llm)\n"
        "Answer:\n"
        "Grounded. [S1]\n"
        "\nSources:\n"
        f"[S1] {document.resolve()} > RAG\n"
    )
    assert result.stderr == ""
    assert len(model.calls) == 2
    assert memory_database.exists()


def test_agent_proposes_pending_memory_after_answer_and_keeps_it_out_of_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
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
            ModelOutput(
                "extractor",
                '{"candidates":[{"type":"goal","key":"goal.ai_engineer",'
                '"content":"Build a portfolio-ready Personal Knowledge Agent."}]}',
                "stop",
            ),
        ]
    )

    result = run_cli(
        "agent",
        "My goal is to build a portfolio-ready Personal Knowledge Agent.",
        "--db",
        str(database),
        language_model=model,
    )

    candidate_match = re.search(r"\[(mem_[0-9a-f]{32})\] pending", result.stdout)
    assert result.returncode == 0
    assert result.stderr == ""
    assert candidate_match is not None
    candidate_id = candidate_match.group(1)
    assert result.stdout == (
        "Route: retrieve (llm)\n"
        "Answer:\n"
        "Grounded. [S1]\n"
        "\nSources:\n"
        f"[S1] {document.resolve()} > RAG\n"
        "\nMemory candidates:\n"
        f"[{candidate_id}] pending goal goal.ai_engineer\n"
        "  Build a portfolio-ready Personal Knowledge Agent.\n"
        "Approve with:\n"
        f"  pka memory approve {candidate_id}\n"
    )
    assert len(model.calls) == 3

    pending = run_cli("memory", "list", "--status", "pending")
    active_search = run_cli(
        "memory",
        "search",
        "portfolio-ready Personal Knowledge Agent",
    )

    assert candidate_id in pending.stdout
    assert "Build a portfolio-ready Personal Knowledge Agent." in pending.stdout
    assert active_search.stdout == "No active memories found.\n"
    assert candidate_id not in active_search.stdout


def test_agent_candidate_lists_all_conflicts_and_preserves_memory_database(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    memory_database = tmp_path / "memory db's" / "custom memory.db"
    quoted_memory_database = shlex.quote(str(memory_database))
    document.write_text("# RAG\n\nretrieval evidence", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(knowledge_database))
    first = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.ai_engineer",
        "Become an AI Engineer.",
        "--memory-db",
        str(memory_database),
    )
    first_match = re.search(r"\[(mem_[0-9a-f]{32})\]", first.stdout)
    assert first_match is not None
    first_id = first_match.group(1)
    second = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.ai_engineer",
        "--keep-both",
        "Publish an AI Engineer portfolio.",
        "--memory-db",
        str(memory_database),
    )
    second_match = re.search(r"\[(mem_[0-9a-f]{32})\]", second.stdout)
    assert second_match is not None
    second_id = second_match.group(1)
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
            ModelOutput(
                "extractor",
                '{"candidates":[{"type":"goal","key":"goal.ai_engineer",'
                '"content":"Build a portfolio-ready Personal Knowledge Agent."}]}',
                "stop",
            ),
        ]
    )

    result = run_cli(
        "agent",
        "My goal is to build a portfolio-ready Personal Knowledge Agent.",
        "--db",
        str(knowledge_database),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    candidate_match = re.search(r"\[(mem_[0-9a-f]{32})\] pending", result.stdout)
    assert result.returncode == 0
    assert candidate_match is not None
    candidate_id = candidate_match.group(1)
    assert f"Conflicts with active IDs: {second_id}, {first_id}" in result.stdout
    assert (
        f"pka memory approve {candidate_id} --replace {second_id} "
        f"--memory-db {quoted_memory_database}" in result.stdout
    )
    assert (
        f"pka memory approve {candidate_id} --replace {first_id} "
        f"--memory-db {quoted_memory_database}" in result.stdout
    )
    assert (
        f"pka memory approve {candidate_id} --keep-both "
        f"--memory-db {quoted_memory_database}" in result.stdout
    )

    pending = run_cli(
        "memory",
        "list",
        "--status",
        "pending",
        "--memory-db",
        str(memory_database),
    )
    active_search = run_cli(
        "memory",
        "search",
        "portfolio-ready Personal Knowledge Agent",
        "--memory-db",
        str(memory_database),
    )

    assert candidate_id in pending.stdout
    assert candidate_id not in active_search.stdout
    assert "Build a portfolio-ready Personal Knowledge Agent." not in (
        active_search.stdout
    )


@pytest.mark.parametrize(
    ("query", "safe_category"),
    [
        ("My goal is " + "x" * 3_990, "input too long"),
        (
            "My goal is to keep api_key=sk-provider-secret-123456789.",
            "sensitive input",
        ),
        (
            "我的目标是轮换 API密钥：cn-provider-secret-123456。",
            "sensitive input",
        ),
    ],
    ids=["input-too-long", "english-sensitive-input", "chinese-sensitive-input"],
)
def test_agent_local_memory_guards_skip_before_extraction_call(
    tmp_path: Path,
    query: str,
    safe_category: str,
) -> None:
    assert len(query) == 4_001 or safe_category == "sensitive input"
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    memory_database = tmp_path / "memory.db"
    document.write_text(
        "# RAG\n\nMy goal is to build a portfolio-ready agent with retrieval evidence. "
        "API密钥 rotation.",
        encoding="utf-8",
    )
    run_cli("ingest", str(document), "--db", str(knowledge_database))
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
        query,
        "--mode",
        "lexical",
        "--db",
        str(knowledge_database),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.endswith(f"\nMemory extraction skipped: {safe_category}\n")
    assert len(model.calls) == 2
    assert "sk-provider-secret-123456789" not in result.stdout + result.stderr
    assert memory_database.exists()


def test_agent_candidate_displays_full_content_before_approval(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    memory_database = tmp_path / "memory.db"
    sentinel = "TAIL-SENTINEL"
    content = "x" * (500 - len(sentinel)) + sentinel
    document.write_text("# RAG\n\nretrieval evidence", encoding="utf-8")
    run_cli("ingest", str(document), "--db", str(knowledge_database))
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
            ModelOutput(
                "extractor",
                '{"candidates":[{"type":"goal","key":"goal.long",'
                f'"content":"{content}"}}]}}',
                "stop",
            ),
        ]
    )

    result = run_cli(
        "agent",
        "My goal is to record a durable long-term objective.",
        "--db",
        str(knowledge_database),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    assert result.returncode == 0
    assert content in result.stdout
    assert sentinel in result.stdout


def test_agent_extraction_prompt_contains_only_policy_and_current_message(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    memory_database = tmp_path / "memory.db"
    old_memory = "OLD-MEMORY-MUST-NOT-REACH-PROVIDER"
    knowledge_evidence = "KNOWLEDGE-EVIDENCE-MUST-NOT-REACH-EXTRACTOR"
    query = "My goal is to build a portfolio-ready agent."
    document.write_text(
        f"# RAG\n\nMy goal is to build a portfolio-ready agent. {knowledge_evidence}",
        encoding="utf-8",
    )
    run_cli("ingest", str(document), "--db", str(knowledge_database))
    remembered = run_cli(
        "memory",
        "remember",
        "--type",
        "goal",
        "--key",
        "goal.old",
        old_memory,
        "--memory-db",
        str(memory_database),
    )
    assert remembered.returncode == 0
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
            ModelOutput("extractor", '{"candidates":[]}', "stop"),
        ]
    )

    result = run_cli(
        "agent",
        query,
        "--mode",
        "lexical",
        "--db",
        str(knowledge_database),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    assert result.returncode == 0
    assert len(model.calls) == 3
    extraction_call = model.calls[2]
    assert [message.role for message in extraction_call] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]
    assert extraction_call[1].content == f"CURRENT USER MESSAGE:\n{query}"
    extraction_payload = "\n".join(message.content for message in extraction_call)
    assert old_memory not in extraction_payload
    assert knowledge_evidence not in extraction_payload


@pytest.mark.parametrize(
    ("extraction_result", "safe_category", "private_value"),
    [
        (
            LanguageModelError("provider-body-private api_key=test-key"),
            "provider error",
            "provider-body-private",
        ),
        (
            ModelOutput(
                "extractor",
                "not JSON: COMPLETE-EXTRACTION-PROMPT-PRIVATE",
                "stop",
            ),
            "invalid extractor output",
            "COMPLETE-EXTRACTION-PROMPT-PRIVATE",
        ),
        (
            ModelOutput(
                "extractor",
                '{"candidates":[{"type":"goal","key":"goal.private",'
                '"content":"api_key=unsafe-output-private"}]}',
                "stop",
            ),
            "unsafe extractor output",
            "unsafe-output-private",
        ),
    ],
)
def test_agent_expected_extraction_failures_keep_answer_and_print_safe_category(
    tmp_path: Path,
    extraction_result: ModelOutput | Exception,
    safe_category: str,
    private_value: str,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    document.write_text(
        "# RAG\n\nMy goal is to build a portfolio-ready agent with retrieval evidence.",
        encoding="utf-8",
    )
    run_cli("ingest", str(document), "--db", str(knowledge_database))
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
            extraction_result,
        ]
    )

    result = run_cli(
        "agent",
        "My goal is to build a portfolio-ready agent.",
        "--mode",
        "lexical",
        "--db",
        str(knowledge_database),
        "--memory-db",
        str(tmp_path / "memory.db"),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.startswith("Route: retrieve (llm)\nAnswer:\nGrounded. [S1]\n")
    assert result.stdout.endswith(f"\nMemory extraction skipped: {safe_category}\n")
    private_output = result.stdout + result.stderr
    assert private_value not in private_output
    assert "goal.private" not in private_output
    assert "test-key" not in private_output
    assert "Traceback" not in private_output
    assert len(model.calls) == 3


def test_agent_embedding_failure_keeps_answer_and_hides_private_values(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    private_content = "PRIVATE-CANDIDATE-CONTENT"
    document.write_text(
        "# RAG\n\nMy goal is to build a portfolio-ready agent with retrieval evidence.",
        encoding="utf-8",
    )
    run_cli("ingest", str(document), "--db", str(knowledge_database))
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
            ModelOutput(
                "extractor",
                '{"candidates":[{"type":"goal","key":"goal.private",'
                f'"content":"{private_content}"}}]}}',
                "stop",
            ),
        ]
    )

    result = run_cli(
        "agent",
        "My goal is to build a portfolio-ready agent.",
        "--mode",
        "lexical",
        "--db",
        str(knowledge_database),
        "--memory-db",
        str(tmp_path / "memory.db"),
        language_model=model,
        embedder=FakeEmbedder({}),
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.startswith("Route: retrieve (llm)\nAnswer:\nGrounded. [S1]\n")
    assert result.stdout.endswith("\nMemory extraction skipped: embedding error\n")
    process_output = result.stdout + result.stderr
    assert private_content not in process_output
    assert "goal.private" not in process_output
    assert "(1.0, 0.0)" not in process_output
    assert "Traceback" not in process_output


def test_agent_storage_failure_keeps_answer_and_hides_private_values(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    document.write_text(
        "# RAG\n\nMy goal is to build a portfolio-ready agent with retrieval evidence.",
        encoding="utf-8",
    )
    run_cli("ingest", str(document), "--db", str(knowledge_database))
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
            ModelOutput(
                "extractor",
                '{"candidates":[{"type":"goal","key":"goal.private",'
                '"content":"PRIVATE-STORAGE-CANDIDATE"}]}',
                "stop",
            ),
        ]
    )
    store = FakeMemoryStore(
        create_pending_result=MemoryStorageError(
            "sqlite-body-private vector=(1.0, 0.0)"
        )
    )

    with patch("personal_knowledge_agent.cli._memory_store", return_value=store):
        result = run_cli(
            "agent",
            "My goal is to build a portfolio-ready agent.",
            "--mode",
            "lexical",
            "--db",
            str(knowledge_database),
            "--memory-db",
            str(tmp_path / "memory.db"),
            language_model=model,
        )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.startswith("Route: retrieve (llm)\nAnswer:\nGrounded. [S1]\n")
    assert result.stdout.endswith("\nMemory extraction skipped: storage error\n")
    process_output = result.stdout + result.stderr
    assert "sqlite-body-private" not in process_output
    assert "PRIVATE-STORAGE-CANDIDATE" not in process_output
    assert "(1.0, 0.0)" not in process_output
    assert "Traceback" not in process_output


@pytest.mark.parametrize(
    ("error", "safe_category"),
    [
        (EmbeddingError("model-load-private"), "embedding error"),
        (MemoryStorageError("sqlite-open-private"), "storage error"),
    ],
)
def test_agent_memory_dependency_construction_failure_keeps_answer(
    tmp_path: Path,
    error: Exception,
    safe_category: str,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    document.write_text(
        "# RAG\n\nMy goal is to build a portfolio-ready agent with retrieval evidence.",
        encoding="utf-8",
    )
    run_cli("ingest", str(document), "--db", str(knowledge_database))
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

    with patch("personal_knowledge_agent.cli._memory_store", side_effect=error):
        result = run_cli(
            "agent",
            "My goal is to build a portfolio-ready agent.",
            "--mode",
            "lexical",
            "--db",
            str(knowledge_database),
            "--memory-db",
            str(tmp_path / "memory.db"),
            language_model=model,
        )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.startswith(
        f"Route: retrieve (llm)\nMemory recall skipped: {safe_category}\n"
        "Answer:\nGrounded. [S1]\n"
    )
    assert result.stdout.endswith(f"\nMemory extraction skipped: {safe_category}\n")
    assert "model-load-private" not in result.stdout
    assert "sqlite-open-private" not in result.stdout
    assert "Traceback" not in result.stdout
    assert len(model.calls) == 2


def test_agent_conflict_lookup_storage_failure_prints_only_safe_category(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    document.write_text(
        "# RAG\n\nMy goal is to build a portfolio-ready agent with retrieval evidence.",
        encoding="utf-8",
    )
    run_cli("ingest", str(document), "--db", str(knowledge_database))
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
            ModelOutput(
                "extractor",
                '{"candidates":[{"type":"goal","key":"goal.agent",'
                '"content":"Build an agent."}]}',
                "stop",
            ),
        ]
    )

    with patch(
        "personal_knowledge_agent.cli.MemoryAdminService.list",
        side_effect=MemoryStorageError("conflict-query-private"),
    ):
        result = run_cli(
            "agent",
            "My goal is to build a portfolio-ready agent.",
            "--mode",
            "lexical",
            "--db",
            str(knowledge_database),
            "--memory-db",
            str(tmp_path / "memory.db"),
            language_model=model,
        )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.startswith("Route: retrieve (llm)\nAnswer:\nGrounded. [S1]\n")
    assert result.stdout.endswith("\nMemory extraction skipped: storage error\n")
    process_output = result.stdout + result.stderr
    assert "Memory candidates:" not in process_output
    assert "Build an agent." not in process_output
    assert "conflict-query-private" not in process_output
    assert "Traceback" not in process_output


def test_agent_unexpected_memory_exception_remains_visible_to_tests(
    tmp_path: Path,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    document.write_text(
        "# RAG\n\nMy goal is to build a portfolio-ready agent with retrieval evidence.",
        encoding="utf-8",
    )
    run_cli("ingest", str(document), "--db", str(knowledge_database))
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
            ModelOutput(
                "extractor",
                '{"candidates":[{"type":"goal","key":"goal.agent",'
                '"content":"Build an agent."}]}',
                "stop",
            ),
        ]
    )
    store = FakeMemoryStore(create_pending_result=RuntimeError("programming bug"))

    with (
        patch("personal_knowledge_agent.cli._memory_store", return_value=store),
        pytest.raises(RuntimeError, match="programming bug"),
    ):
        run_cli(
            "agent",
            "My goal is to build a portfolio-ready agent.",
            "--mode",
            "lexical",
            "--db",
            str(knowledge_database),
            "--memory-db",
            str(tmp_path / "memory.db"),
            language_model=model,
        )


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

    with patch(
        "personal_knowledge_agent.cli._memory_store",
        side_effect=AssertionError("fallback route must not construct memory storage"),
    ):
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
            ModelOutput("router", "not JSON", "stop"),
            "Route: retrieve (fallback: invalid router output)\n",
        ),
        (
            LanguageModelError("router-provider-private"),
            "Route: retrieve (fallback: provider error)\n",
        ),
    ],
)
def test_agent_failed_route_does_not_extract_durable_memory(
    tmp_path: Path,
    router_result: ModelOutput | Exception,
    route_line: str,
) -> None:
    document = tmp_path / "rag.md"
    knowledge_database = tmp_path / "knowledge.db"
    memory_database = tmp_path / "memory.db"
    document.write_text(
        "# RAG\n\nMy goal is to build a portfolio-ready agent with retrieval evidence.",
        encoding="utf-8",
    )
    run_cli("ingest", str(document), "--db", str(knowledge_database))
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
        "My goal is to build a portfolio-ready agent.",
        "--mode",
        "lexical",
        "--db",
        str(knowledge_database),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout.startswith(route_line + "Answer:\nGrounded. [S1]\n")
    assert "Memory candidates:" not in result.stdout
    assert "Memory extraction skipped:" not in result.stdout
    assert "router-provider-private" not in result.stdout + result.stderr
    assert len(model.calls) == 2
    assert memory_database.exists() is (route_line == "Route: retrieve (llm)\n")


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
    memory_database = tmp_path / "memory.db"
    document.write_text(
        "# RAG\n\nMy goal is to build a portfolio-ready agent with retrieval evidence.",
        encoding="utf-8",
    )
    run_cli("ingest", str(document), "--db", str(database))
    model = ScriptedLanguageModel(
        [router_result, LanguageModelError("answer generation failed")]
    )

    result = run_cli(
        "agent",
        "My goal is to build a portfolio-ready agent.",
        "--mode",
        "lexical",
        "--db",
        str(database),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    assert result.returncode == 1
    assert result.stdout == route_line
    assert "answer generation failed" in result.stderr
    assert "Traceback" not in result.stderr
    assert len(model.calls) == 2
    assert memory_database.exists() is (route_line == "Route: retrieve (llm)\n")


def test_agent_empty_document_and_memory_context_returns_local_refusal(
    tmp_path: Path,
) -> None:
    memory_database = tmp_path / "memory.db"
    model = ScriptedLanguageModel(
        [ModelOutput("router", '{"action":"retrieve","direct_intent":null}', "stop")]
    )

    result = run_cli(
        "agent",
        "How does RAG work?",
        "--db",
        str(tmp_path / "empty.db"),
        "--memory-db",
        str(memory_database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stdout == (
        "Route: retrieve (llm)\n"
        "Insufficient evidence:\n"
        "The knowledge base does not contain enough evidence.\n"
    )
    assert result.stderr == ""
    assert len(model.calls) == 1
    assert memory_database.exists()


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


class TestGraphParser:
    def test_build_parser(self) -> None:
        from personal_knowledge_agent.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["graph", "build", "/tmp/test.md"])
        assert args.command == "graph"
        assert args.graph_command == "build"
        assert str(args.source) == "/tmp/test.md"

    def test_search_parser(self) -> None:
        from personal_knowledge_agent.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["graph", "search", "RAG"])
        assert args.command == "graph"
        assert args.graph_command == "search"
        assert args.query == "RAG"

    def test_show_parser(self) -> None:
        from personal_knowledge_agent.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["graph", "show", "ent_abc123"])
        assert args.command == "graph"
        assert args.graph_command == "show"
        assert args.entity_id == "ent_abc123"


def test_graph_show_prints_relation_direction_and_complete_provenance(
    tmp_path: Path,
) -> None:
    graph_database = tmp_path / "graph.db"
    rag = GraphEntity(
        make_entity_id(EntityType.METHOD, "RAG"),
        EntityType.METHOD,
        "RAG",
    )
    vector_retrieval = GraphEntity(
        make_entity_id(EntityType.TECHNOLOGY, "Vector Retrieval"),
        EntityType.TECHNOLOGY,
        "Vector Retrieval",
    )
    chunk = Chunk(
        "c1",
        "doc1",
        0,
        "RAG uses vector retrieval.",
        ("Architecture",),
    )
    relation = GraphRelation(
        make_relation_id(
            RelationType.USES,
            rag.entity_id,
            vector_retrieval.entity_id,
        ),
        RelationType.USES,
        rag.entity_id,
        vector_retrieval.entity_id,
    )
    extraction = GraphChunkExtraction(
        chunk,
        (rag, vector_retrieval),
        (GraphRelationEvidence(relation, "RAG uses vector retrieval."),),
    )
    source_document = GraphSourceDocument(
        "doc1",
        Path("/notes/rag.md"),
        "a" * 64,
        (chunk,),
    )
    SQLiteKnowledgeGraphStore(graph_database).replace_document(
        source_document,
        (extraction,),
    )

    result = run_cli(
        "graph",
        "show",
        rag.entity_id,
        "--graph-db",
        str(graph_database),
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == (
        f"Entity: [{rag.entity_id}] method RAG\n"
        "\n"
        "Outgoing:\n"
        f"uses -> [{vector_retrieval.entity_id}] technology Vector Retrieval\n"
        "  [1] /notes/rag.md > Architecture\n"
        "      RAG uses vector retrieval.\n"
    )


def test_graph_search_does_not_create_a_database_when_none_exists(
    tmp_path: Path,
) -> None:
    graph_database = tmp_path / "not-created" / "graph.db"

    result = run_cli(
        "graph",
        "search",
        "RAG",
        "--graph-db",
        str(graph_database),
    )

    assert result.returncode == 1
    assert "empty" in result.stderr.lower()
    assert not graph_database.exists()


def test_graph_build_rejects_an_unindexed_source_before_model_setup(
    tmp_path: Path,
) -> None:
    knowledge_database = tmp_path / "knowledge.db"
    graph_database = tmp_path / "not-created" / "graph.db"
    with sqlite3.connect(knowledge_database) as connection:
        connection.execute(
            "CREATE TABLE documents ("
            "document_id TEXT PRIMARY KEY, source TEXT UNIQUE NOT NULL, "
            "content_hash TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE chunks ("
            "chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, "
            "position INTEGER NOT NULL, text TEXT NOT NULL, "
            "heading_path TEXT NOT NULL, token_count INTEGER NOT NULL)"
        )

    with patch(
        "personal_knowledge_agent.infrastructure.llm_settings.load_llm_settings",
        side_effect=AssertionError("model setup must not run"),
    ):
        result = run_cli(
            "graph",
            "build",
            str(tmp_path / "unindexed.md"),
            "--db",
            str(knowledge_database),
            "--graph-db",
            str(graph_database),
        )

    assert result.returncode == 1
    assert "not found in knowledge index" in result.stderr
    assert not graph_database.exists()


def test_graph_build_extracts_each_indexed_chunk_and_reports_local_counts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rag.md"
    source.write_text("placeholder", encoding="utf-8")
    knowledge_database = tmp_path / "knowledge.db"
    graph_database = tmp_path / "graph.db"
    source_text = str(source.resolve())
    with sqlite3.connect(knowledge_database) as connection:
        connection.execute(
            "CREATE TABLE documents ("
            "document_id TEXT PRIMARY KEY, source TEXT UNIQUE NOT NULL, "
            "content_hash TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE chunks ("
            "chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, "
            "position INTEGER NOT NULL, text TEXT NOT NULL, "
            "heading_path TEXT NOT NULL, token_count INTEGER NOT NULL)"
        )
        connection.execute(
            "INSERT INTO documents(document_id, source, content_hash) VALUES (?, ?, ?)",
            ("document-1", source_text, "a" * 64),
        )
        connection.executemany(
            "INSERT INTO chunks("
            "chunk_id, document_id, position, text, heading_path, token_count"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "chunk-1",
                    "document-1",
                    0,
                    "RAG uses vector retrieval.",
                    '["Architecture"]',
                    4,
                ),
                (
                    "chunk-2",
                    "document-1",
                    1,
                    "RAG uses vector retrieval.",
                    '["Retrieval"]',
                    4,
                ),
            ],
        )
    output = ModelOutput(
        "fake-model",
        '{"entities":[{"local_id":"E1","name":"RAG","type":"method"},'
        '{"local_id":"E2","name":"Vector Retrieval","type":"technology"}],'
        '"relations":[{"source_id":"E1","type":"uses","target_id":"E2",'
        '"evidence_quote":"RAG uses vector retrieval."}]}',
        "stop",
    )
    model = ScriptedLanguageModel((output, output))

    result = run_cli(
        "graph",
        "build",
        str(source),
        "--db",
        str(knowledge_database),
        "--graph-db",
        str(graph_database),
        language_model=model,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == (
        f"Built graph for {source_text}\n"
        "Chunks processed: 2\n"
        "Entities contributed: 2\n"
        "Relations contributed: 1\n"
        "Evidence records: 2\n"
    )
    assert graph_database.is_file()
    assert len(model.calls) == 2

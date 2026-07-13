# Phase 0 Project Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python 3.12+ project foundation exposing a tested `pka health` CLI command.

**Architecture:** The console entry point in `cli.py` accepts command-line arguments and dispatches only the `health` command. The package has future-oriented `application`, `domain`, and `infrastructure` boundaries but contains no RAG implementation or production dependency in Phase 0.

**Tech Stack:** Python 3.12+, uv, pytest, ruff, mypy, standard-library `argparse`.

---

## File structure

- `pyproject.toml`: build metadata, CLI registration, development tools and their configuration.
- `src/personal_knowledge_agent/__init__.py`: package marker and version constant.
- `src/personal_knowledge_agent/cli.py`: the CLI parser and `main()` entry point.
- `src/personal_knowledge_agent/application/__init__.py`: future application-use-case package boundary.
- `src/personal_knowledge_agent/domain/__init__.py`: future core-domain package boundary.
- `src/personal_knowledge_agent/infrastructure/__init__.py`: future adapter package boundary.
- `tests/test_cli.py`: subprocess-level tests of installed CLI behavior.
- `.gitignore`: excludes local/generated data, including secrets.
- `.env.example`: documents the Phase 1 API-key variable without a value.
- `README.md`: setup and quality-gate commands.
- `docs/learning-log/phase-0.md`: short learning notes for the project owner.

### Task 1: Configure the project and quality tools

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/personal_knowledge_agent/__init__.py`
- Create: `src/personal_knowledge_agent/application/__init__.py`
- Create: `src/personal_knowledge_agent/domain/__init__.py`
- Create: `src/personal_knowledge_agent/infrastructure/__init__.py`

- [ ] **Step 1: Create `pyproject.toml` with an editable CLI package and development dependencies**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "personal-knowledge-agent"
version = "0.1.0"
description = "A personal knowledge management system built with RAG and Agent patterns."
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[dependency-groups]
dev = [
    "mypy>=1.16.0",
    "pytest>=8.3.0",
    "ruff>=0.11.0",
]

[project.scripts]
pka = "personal_knowledge_agent.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.12"
files = ["src"]
strict = true
```

- [ ] **Step 2: Create package markers**

```python
# src/personal_knowledge_agent/__init__.py
"""Personal Knowledge Agent package."""

__version__ = "0.1.0"
```

```python
# src/personal_knowledge_agent/application/__init__.py
"""Application use cases and orchestration."""
```

```python
# src/personal_knowledge_agent/domain/__init__.py
"""Core domain models and protocols."""
```

```python
# src/personal_knowledge_agent/infrastructure/__init__.py
"""External-system adapters."""
```

- [ ] **Step 3: Create secret and generated-file exclusions**

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.env
```

```dotenv
# Copy this file to .env before Phase 1 and add your own key locally.
OPENAI_API_KEY=
```

- [ ] **Step 4: Sync the environment and verify the configuration is accepted**

Run: `uv sync --all-groups`

Expected: `uv.lock` is created and uv reports that the project and development dependencies were installed into `.venv`.

- [ ] **Step 5: Commit the configuration foundation**

```bash
git add pyproject.toml uv.lock .gitignore .env.example src/personal_knowledge_agent
git commit -m "chore: configure Python project tooling"
```

### Task 2: Specify the CLI behavior with failing tests

**Files:**
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing subprocess tests**

```python
"""Black-box tests for the command-line interface."""

from __future__ import annotations

import subprocess
import sys


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
```

- [ ] **Step 2: Run the tests to verify the health behavior is not implemented**

Run: `uv run pytest tests/test_cli.py -v`

Expected: FAIL because `personal_knowledge_agent.cli` does not exist.

- [ ] **Step 3: Commit the executable specification**

```bash
git add tests/test_cli.py
git commit -m "test: define CLI health behavior"
```

### Task 3: Implement the minimal CLI

**Files:**
- Create: `src/personal_knowledge_agent/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Implement the parser and entry point**

```python
"""Command-line interface for Personal Knowledge Agent."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the parser for supported user commands."""
    parser = argparse.ArgumentParser(prog="pka")
    parser.add_argument(
        "command",
        choices=("health",),
        help="Command to run. Currently supported: health.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the CLI and print the result of the requested command."""
    parsed_arguments = build_parser().parse_args(arguments)

    if parsed_arguments.command == "health":
        print("Personal Knowledge Agent is healthy.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the focused tests**

Run: `uv run pytest tests/test_cli.py -v`

Expected: both tests PASS.

- [ ] **Step 3: Verify the installed console script**

Run: `uv run pka health`

Expected output: `Personal Knowledge Agent is healthy.`

- [ ] **Step 4: Commit the CLI implementation**

```bash
git add src/personal_knowledge_agent/cli.py tests/test_cli.py
git commit -m "feat: add CLI health command"
```

### Task 4: Document developer workflow and run every quality gate

**Files:**
- Create: `README.md`
- Create: `docs/learning-log/phase-0.md`

- [ ] **Step 1: Create the project README**

```markdown
# Personal Knowledge Agent

A personal knowledge-management system built incrementally with RAG and Agent patterns.

## Phase 0 scope

Phase 0 provides a reproducible Python project, quality checks, and a minimal CLI. It does not yet ingest documents, call an LLM, or retrieve knowledge.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --all-groups
```

## Run

```bash
uv run pka health
```

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```
```

- [ ] **Step 2: Create the Phase 0 learning log**

```markdown
# Phase 0 Learning Log

## What I learned

- `pyproject.toml` centralizes Python project metadata and tool configuration.
- `uv sync --all-groups` creates a reproducible local environment from declared dependencies and `uv.lock`.
- The `src/` layout prevents tests from importing source files accidentally from the repository root.
- Ruff catches common static mistakes and enforces formatting; mypy checks type contracts; pytest verifies behavior.
- `.env` belongs only on the local machine, while `.env.example` documents required variable names safely.

## Commands I can explain

```bash
uv sync --all-groups
uv run pka health
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```
```

- [ ] **Step 3: Run all quality gates**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest`

Expected: every command exits with status 0; pytest reports two passing tests.

- [ ] **Step 4: Commit documentation and the verified Phase 0 baseline**

```bash
git add README.md docs/learning-log/phase-0.md
git commit -m "docs: add Phase 0 workflow guide"
```

### Task 5: Confirm final repository state

**Files:**
- Verify: all files defined in this plan

- [ ] **Step 1: Inspect tracked and untracked changes**

Run: `git status --short`

Expected: no output; all Phase 0 work is committed.

- [ ] **Step 2: Inspect the concise commit history**

Run: `git log --oneline -4`

Expected: the history contains, newest first:

```text
docs: add Phase 0 workflow guide
feat: add CLI health command
test: define CLI health behavior
chore: configure Python project tooling
```

## Verification matrix

| Requirement | Verification |
| --- | --- |
| Reproducible environment | `uv sync --all-groups` creates/synchronizes `.venv` from `pyproject.toml` and `uv.lock`. |
| CLI succeeds | `uv run pka health` prints the health message and exits 0. |
| CLI rejects invalid input | `uv run pytest tests/test_cli.py -v` passes the unknown-command test. |
| Code quality | Ruff formatting, Ruff linting, and mypy all exit 0. |
| No secret leakage | `.gitignore` includes `.env`; `.env.example` contains no secret value. |
| Learning record | `docs/learning-log/phase-0.md` explains the setup choices and commands. |

# Personal Knowledge Agent

A personal knowledge-management system built incrementally with RAG and Agent patterns.

## Phase 0 scope

Phase 0 provides a reproducible Python project, quality checks, and a minimal CLI.
It does not yet ingest documents, call an LLM, or retrieve knowledge.

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

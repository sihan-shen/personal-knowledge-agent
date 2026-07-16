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

## WSL migration note

- The active repository is `/mnt/c/Users/19850/Projects/RAG-Agent`; it is not
  located under the former native-Linux path `/home/sihan/文档/Projects/RAG-Agent`.
- `TMPDIR=/tmp` keeps pytest and Python temporary files off the Windows-mounted
  filesystem, which avoids temporary-file cleanup failures.
- `UV_LINK_MODE=copy` avoids uv's cross-filesystem link warning on `/mnt/c`.
- Both variables are exported by `~/.bashrc`; use `source ~/.bashrc` or open a
  new shell before running the standard commands.

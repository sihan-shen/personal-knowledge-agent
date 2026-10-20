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

## Environment note

- Run the project from the repository root; do not create a second checkout.
- `TMPDIR=/tmp` can keep Python temporary files off a Windows-mounted filesystem.
- `UV_LINK_MODE=copy` avoids uv's cross-filesystem link warning on such a filesystem.
- Set these variables in your shell profile if needed.

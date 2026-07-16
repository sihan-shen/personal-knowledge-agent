# Personal Knowledge Agent contributor guide

## Project intent

This is a learning-first Python project for a personal knowledge agent. Build
features incrementally through the documented phases; do not add later-phase
capabilities merely to make the project look complete.

## Workspace and environment

- Canonical WSL repository: `/mnt/c/Users/19850/Projects/RAG-Agent`.
- A compatibility symlink exists at
  `/home/sihan/文档/Projects/RAG-Agent` for older Codex task contexts. Do not use
  it as a second checkout or move the repository there.
- The project uses `uv` and a local `.venv`. On this Windows-mounted filesystem,
  run commands with `TMPDIR=/tmp` and `UV_LINK_MODE=copy` when the shell has not
  already loaded those exports from `~/.bashrc`.
- Keep generated knowledge bases under `.pka/`; never commit them.

## Common commands

```bash
uv sync --all-groups
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pka health
```

Run targeted tests before the full suite when working TDD. Run the full quality
gate before a feature is merged.

## Engineering rules

- Use Python 3.12+ features, complete type annotations, and strict mypy.
- Keep `domain/` independent of infrastructure libraries. Define replaceable
  boundaries in `domain/protocols.py`.
- Keep application services small: they coordinate domain protocols but do not
  embed storage or model-specific logic.
- Put SQLite, model-runtime, document parsing, and ranking implementation in
  `infrastructure/`.
- Prefer explicit, readable code over framework or factory abstractions.
- Write a failing behavioral test before implementation. Use deterministic fakes
  for external models and network-dependent integrations.
- Preserve existing behavior unless the approved phase design changes it.

## Documentation and planning

- Record approved architecture decisions in `docs/superpowers/specs/`.
- Keep task-level implementation plans in `docs/superpowers/plans/`.
- Update the relevant learning log when a phase produces a new concept worth
  explaining for an AI Engineer portfolio.

## Git and workspace safety

- Use an isolated worktree for feature work when practical.
- Make narrow, coherent commits after tests pass; do not mix user files with
  feature changes.
- Preserve unrelated user-owned directories such as `papers/`, `references/`,
  and `research/` if they appear in a checkout.
- Do not use destructive Git commands (`reset --hard`, checkout-based discard,
  or broad cleanup) without explicit user approval.

## Current architecture and scope

- Phase 1A is complete: Markdown ingestion, heading-aware chunks, mixed
  Chinese/English tokenization, SQLite persistence, and BM25 retrieval.
- Phase 1B is complete: local `multilingual-e5-small` embeddings, exact cosine
  retrieval, and RRF hybrid ranking.
- PDF/web/code ingestion, LLM answer generation, agent routing, memory,
  knowledge graphs, evaluation, and deployment are future phases.

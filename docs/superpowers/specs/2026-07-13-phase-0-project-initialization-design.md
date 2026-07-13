# Phase 0: Project Initialization Design

## Purpose

Create a reproducible, testable Python foundation for Personal Knowledge Agent. This phase intentionally excludes RAG, LLM, vector database, document ingestion, Agent, and Memory implementations.

## Decisions

- Use Python 3.12 or newer.
- Use `uv` with `pyproject.toml` and a committed `uv.lock` for dependency and environment management.
- Use a `src/` package layout.
- Start with a CLI, keeping the domain and application code independent from the delivery interface.
- Keep production dependencies empty in Phase 0. Install only `pytest`, `ruff`, and `mypy` as development dependencies.
- Store secrets only in a local `.env`; include `.env.example` with variable names only.

## Alternatives Considered

1. `uv` plus `pyproject.toml` (chosen): fast, standard, and reproducible.
2. Poetry: capable dependency management but adds a separate workflow with no Phase 0 advantage.
3. `venv` plus `pip`: simple but has weaker dependency locking and splits configuration across files.

For delivery, a CLI was chosen over FastAPI and Streamlit. It exposes the RAG pipeline without adding HTTP or UI concerns. A later API or UI will call the same application layer.

## Structure

```text
RAG-Agent/
├── src/
│   └── personal_knowledge_agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── application/
│       ├── domain/
│       └── infrastructure/
├── tests/
│   └── test_cli.py
├── docs/
│   ├── learning-log/
│   │   └── phase-0.md
│   └── superpowers/specs/
│       └── 2026-07-13-phase-0-project-initialization-design.md
├── pyproject.toml
├── README.md
├── .env.example
└── .gitignore
```

The initially empty directories establish the future boundaries:

- `application`: application use cases and orchestration.
- `domain`: stable data types and protocols.
- `infrastructure`: replaceable adapters such as file loaders, LLM clients, and vector stores.

## Minimal CLI Contract

The package exposes `pka` as a console command.

- `pka health` exits with code 0 and prints an environment/project health message.
- An unknown subcommand exits nonzero and prints helpful usage information.
- `cli.main() -> None` is the command entry point.

No RAG dependencies or behavior are included.

## Quality Gates

Every Phase 0 change must pass:

```bash
uv sync --all-groups
uv run pka health
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

`tests/test_cli.py` tests both the successful health check and unknown-command failure. Ruff checks formatting and common static errors; mypy checks type contracts; pytest verifies runtime behavior.

## Git Strategy

Git should track source, tests, documentation, `pyproject.toml`, and `uv.lock`. It must ignore virtual environments, caches, local environment files, coverage artefacts, and generated build output.

The intended initialization commit is:

```text
chore: initialize Python project
```

## Scope Boundaries

This phase does not add LangChain, LlamaIndex, Chroma, OpenAI SDKs, PDF parsers, embeddings, prompts, retrieval, Agents, long-term memory, knowledge graphs, or evaluation. Each will be selected in its respective phase after its interface and trade-offs are discussed.

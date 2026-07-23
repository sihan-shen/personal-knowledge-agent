# Phase 6A Docker CLI Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Deliver a reproducible, non-root Docker image for the existing pka CLI, with explicit host persistence for .pka data and E5 model caches.

**Architecture:** Track uv.lock and use it in a multi-stage build. The builder creates the production virtual environment; the smaller runtime copies that environment, runs pka as a non-root user, and defaults to the existing offline health command. Runtime data, documents, model caches, and credentials remain outside the image.

**Tech Stack:** Python 3.12, uv 0.11.29, Hatchling, Docker multi-stage builds, pytest, Ruff, mypy, sentence-transformers.

## Global Constraints

- Preserve domain, application, infrastructure, and CLI Python interfaces; this phase packages the application only.
- Generate and track uv.lock; build the image with uv sync --locked.
- Do not copy or commit .env, .pka, model caches, user documents, papers, references, or research.
- The runtime has production dependencies only; test and lint tools stay out of the image.
- The default command and health check are offline pka health; neither may download E5 or call an LLM.
- Use the dedicated non-root pka user.
- Bind-mount application state only at /workspace/.pka and caches only at /home/pka/.cache.
- Keep normal pytest Docker-independent. Docker is an explicit local smoke check.
- On the Windows-mounted WSL checkout, use TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache with uv commands.

---

## File structure

| Path | Responsibility |
| --- | --- |
| uv.lock | Locked dependency graph used locally and in the Docker builder. |
| .gitignore | Permits version control of uv.lock; still ignores secrets and generated data. |
| Dockerfile | Production build, non-root runtime, cache settings, CLI entry point, health check. |
| .dockerignore | Excludes sensitive, generated, and user-owned build-context files. |
| tests/test_container_config.py | Static Docker configuration regression tests. |
| README.md | Build, persistence, ingestion, cache, and runtime-secret instructions. |
| docs/learning-log/phase-6a.md | Learning record for reproducible container runtime boundaries. |

## Task 1: Lock the dependency graph

**Files:**
- Create: uv.lock
- Modify: .gitignore line 18

**Interfaces:**
- Consumes: pyproject.toml and its explicit CPU-only PyTorch index.
- Produces: a tracked lock accepted by uv lock --check and Docker's uv sync --locked.

- [ ] **Step 1: Remove the stale ignore rule**

Delete exactly this line from .gitignore, leaving every other rule unchanged:

    uv.lock

- [ ] **Step 2: Generate the lock file**

Run:

    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv lock

Expected: uv.lock appears at the repository root and resolves torch via the pytorch-cpu source. No .venv, .pka, .env, or user directory changes.

- [ ] **Step 3: Verify the generated contract**

Run:

    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv lock --check

Expected: exit 0 and a valid/up-to-date lock message.

- [ ] **Step 4: Commit only dependency reproducibility changes**

Run:

    git add .gitignore uv.lock
    git commit -m "build: lock runtime dependencies"

Expected: exactly .gitignore and uv.lock are committed; no credentials, generated data, caches, or user-owned directories are staged.

## Task 2: Add a testable Docker runtime contract

**Files:**
- Create: tests/test_container_config.py
- Create: Dockerfile
- Create: .dockerignore

**Interfaces:**
- Consumes: Task 1's uv.lock and the existing pka console-script declaration in pyproject.toml.
- Produces: a container that starts as pka health, plus Python tests that protect its configuration contract without Docker.

- [ ] **Step 1: Write the failing static configuration test**

Create tests/test_container_config.py:

    """Regression tests for the checked-in Docker runtime contract."""

    from __future__ import annotations

    from pathlib import Path


    ROOT = Path(__file__).resolve().parents[1]


    def test_dockerfile_runs_the_offline_cli_as_a_non_root_user() -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        assert "uv.lock" in dockerfile
        assert "uv sync --locked --no-dev" in dockerfile
        assert "useradd --create-home --uid 10001 pka" in dockerfile
        assert "USER pka" in dockerfile
        assert 'ENTRYPOINT ["pka"]' in dockerfile
        assert 'CMD ["health"]' in dockerfile
        assert 'HEALTHCHECK CMD ["pka", "health"]' in dockerfile
        assert "HF_HOME=/home/pka/.cache/huggingface" in dockerfile
        assert "SENTENCE_TRANSFORMERS_HOME=/home/pka/.cache/sentence_transformers" in dockerfile


    def test_dockerignore_excludes_sensitive_and_generated_runtime_state() -> None:
        ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

        assert {".env", ".pka/", ".venv/", ".git/"} <= ignored
        assert {"papers/", "references/", "research/"} <= ignored
        assert {"__pycache__/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/"} <= ignored

- [ ] **Step 2: Prove the test is red before configuration exists**

Run:

    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_container_config.py -v

Expected: FAIL with FileNotFoundError for Dockerfile and .dockerignore.

- [ ] **Step 3: Add the multi-stage Dockerfile**

Create Dockerfile:

    FROM python:3.12-slim-trixie AS builder
    COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

    WORKDIR /build

    ENV UV_COMPILE_BYTECODE=1 \
        UV_LINK_MODE=copy

    COPY pyproject.toml uv.lock README.md ./
    RUN uv sync --locked --no-dev --no-install-project

    COPY src ./src
    RUN uv sync --locked --no-dev --no-editable

    FROM python:3.12-slim-trixie AS runtime

    ENV PATH="/opt/venv/bin:$PATH" \
        HF_HOME=/home/pka/.cache/huggingface \
        SENTENCE_TRANSFORMERS_HOME=/home/pka/.cache/sentence_transformers \
        PYTHONUNBUFFERED=1

    RUN useradd --create-home --uid 10001 pka \
        && mkdir -p /workspace /home/pka/.cache \
        && chown -R pka:pka /workspace /home/pka

    COPY --from=builder --chown=pka:pka /build/.venv /opt/venv

    WORKDIR /workspace
    USER pka

    ENTRYPOINT ["pka"]
    CMD ["health"]
    HEALTHCHECK CMD ["pka", "health"]

The manifest, lock, and README are copied before source so dependency layers are reusable. The final non-editable sync puts package files inside the virtual environment, so only that production environment must reach the runtime stage.

- [ ] **Step 4: Add the restrictive build context**

Create .dockerignore:

    .git/
    .worktrees/
    .venv/
    .env
    .pka/
    .pka-model-cache/
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
    papers/
    references/
    research/

Do not ignore uv.lock, pyproject.toml, README.md, or src: the builder requires them.

- [ ] **Step 5: Make the focused test green**

Run:

    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_container_config.py -v
    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check tests/test_container_config.py
    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/test_container_config.py

Expected: two pytest cases pass and Ruff reports no findings. If necessary, run uv run ruff format tests/test_container_config.py, then repeat the commands.

- [ ] **Step 6: Commit the testable runtime contract**

Run:

    git add Dockerfile .dockerignore tests/test_container_config.py
    git commit -m "feat: add Docker CLI runtime"

Expected: only Docker configuration and its test are committed.

## Task 3: Document safe and persistent container usage

**Files:**
- Modify: README.md lines 7-60
- Create: docs/learning-log/phase-6a.md

**Interfaces:**
- Consumes: Task 2's image name convention, /workspace, /input, /workspace/.pka, and /home/pka/.cache paths.
- Produces: copyable commands that keep secrets and user files out of the image, plus a portfolio-ready explanation of the deployment boundary.

- [ ] **Step 1: Update scope and insert Docker documentation before the Health check section**

Update the current-scope text to include completed Phase 5A/5B evaluation. Then insert this section after the WSL setup section:

    ## Run with Docker

    Docker packages the production pka CLI. It does not contain your knowledge base, source documents, embedding-model cache, or .env file. Build the image from tracked source and the committed dependency lock:

        docker build -t pka:local .
        docker run --rm pka:local

    The default command is the offline health check. It does not call an LLM or download the embedding model.

    Persist application data and the model cache with explicit host directories:

        mkdir -p .pka .pka-model-cache
        docker run --rm \
          -v "$(pwd)/.pka:/workspace/.pka" \
          -v "$(pwd)/.pka-model-cache:/home/pka/.cache" \
          pka:local health

    For ingestion, mount Markdown documents at /input. The read-only flag prevents the container changing source files:

        docker run --rm \
          -v "$(pwd)/.pka:/workspace/.pka" \
          -v "$(pwd)/.pka-model-cache:/home/pka/.cache" \
          -v "$(pwd)/notes:/input:ro" \
          pka:local ingest /input/rag.md

    The first semantic or hybrid query downloads intfloat/multilingual-e5-small. Keep a network connection for that download; later containers reuse .pka-model-cache. If a bind mount is omitted, the corresponding data is deleted with the container.

    LLM-backed commands use runtime variables only. The CLI does not load .env, but Docker can pass it without copying it into the image:

        docker run --rm \
          --env-file .env \
          -v "$(pwd)/.pka:/workspace/.pka" \
          -v "$(pwd)/.pka-model-cache:/home/pka/.cache" \
          pka:local ask "RAG 为什么需要检索？"

    Do not use build arguments for secrets. Remote-answer, Agent, graph, and Judge commands can send retrieved text to the configured provider; choose a provider whose data policy you trust.

Keep local uv instructions unchanged. These are POSIX-shell commands; no host document path becomes part of the image.

- [ ] **Step 2: Write the learning log**

Create docs/learning-log/phase-6a.md:

    # Phase 6A: Docker CLI Runtime

    ## What this phase delivers

    Phase 6A packages the existing production CLI as a Docker image. It does not change retrieval, Agent routing, memory, graph answering, or evaluation. The image defaults to offline pka health and runs as a dedicated non-root user.

    ## Reproducibility is more than a Dockerfile

    A Dockerfile repeats installation instructions, but cannot alone select the same Python dependencies tomorrow. uv.lock records the resolved graph, and Docker builds with uv sync --locked. Dependency changes are therefore deliberate and reviewable.

    ## Image state versus runtime state

    The image contains application code and production dependencies. It excludes .pka, source documents, .env, and model caches. Users pass those explicitly:

    - .pka mounts at /workspace/.pka for SQLite knowledge, memory, graph, and evaluation data.
    - A host cache mounts at /home/pka/.cache so the first E5 download is reused.
    - Documents mount at /input and should be read-only for ingestion.
    - Provider credentials are runtime environment variables, never image layers or build arguments.

    Deleting an unmounted container deletes its local state; deleting a mounted container leaves the host directory intact.

    ## Why health is deliberately narrow

    pka health proves the CLI starts. It does not prove that E5 can download, a model cache is writable, or an LLM provider is configured. Those dependencies run only when the requested command needs them, keeping health deterministic.

- [ ] **Step 3: Check that documentation preserves the secret boundary**

Run:

    rg -n --hidden --glob '!.git/**' 'COPY \.env|--build-arg|\.pka/|\.pka-model-cache|--env-file' README.md Dockerfile .dockerignore docs/learning-log/phase-6a.md

Expected: .dockerignore excludes .pka and cache paths; README uses --env-file .env; Dockerfile never copies .env and no real credential appears.

- [ ] **Step 4: Commit docs and the learning record**

Run:

    git add README.md docs/learning-log/phase-6a.md
    git commit -m "docs: explain Docker CLI usage"

Expected: exactly the README and Phase 6A learning log are committed.

## Task 4: Run quality gates and Docker smoke verification

**Files:**
- Verify only: uv.lock, .gitignore, Dockerfile, .dockerignore, tests/test_container_config.py, README.md, and docs/learning-log/phase-6a.md

**Interfaces:**
- Consumes: all artifacts from Tasks 1-3.
- Produces: evidence that Python gates remain portable and that a Docker-capable machine can build and run the image.

- [ ] **Step 1: Run the full Python quality gate**

Run:

    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest
    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .
    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run mypy src
    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pka health
    TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv lock --check

Expected: pytest passes; Ruff format/lint have no findings; mypy has no errors; health prints Personal Knowledge Agent is healthy.; lock check exits 0.

- [ ] **Step 2: Build the production image**

Run:

    docker build -t pka:local .

Expected: the builder and runtime stages complete. The build context excludes .env, .pka, .pka-model-cache, user-owned directories, and .git.

- [ ] **Step 3: Verify default command and health check**

Run:

    docker run --rm pka:local
    docker inspect --format '{{.Config.Healthcheck.Test}}' pka:local

Expected: the first command prints Personal Knowledge Agent is healthy. and exits 0. The second command contains pka health.

- [ ] **Step 4: Verify the mounted runtime paths**

Run:

    mkdir -p .pka .pka-model-cache
    docker run --rm \
      -v "$(pwd)/.pka:/workspace/.pka" \
      -v "$(pwd)/.pka-model-cache:/home/pka/.cache" \
      pka:local health

Expected: health succeeds as the non-root container user. It intentionally does not download E5 or call an LLM; a later networked semantic/hybrid command exercises cache reuse.

- [ ] **Step 5: Review final scope; commit a formatter-only correction only if needed**

Run:

    git diff --check
    git status --short
    git log --oneline -4

Expected: no whitespace errors; only planned tracked files are in Phase 6A commits; .env, .pka, .pka-model-cache, papers, references, and research are absent. If a tracked formatter correction is necessary, commit it with git commit -m "style: format Docker runtime files"; otherwise create no empty commit.

## Requirements coverage review

| Approved requirement | Plan task |
| --- | --- |
| Tracked deterministic dependency graph | Task 1 |
| Multi-stage production image | Task 2 |
| Non-root CLI entry point and offline health check | Task 2 |
| Explicit data, model-cache, and document boundaries | Tasks 2-3 |
| Secrets, generated data, caches, and user directories excluded | Task 2 |
| No silent failure or retrieval downgrade | Task 3 documents unchanged CLI behavior |
| Docker-independent Python regression test | Task 2 |
| Local Docker smoke validation | Task 4 |
| Learning-first project explanation | Task 3 |

## Self-review

- The plan separates dependency reproducibility, runtime behavior, user-facing boundaries, and release verification into independently reviewable tasks.
- Each configuration task includes exact paths, a red/green or executable verification cycle, and a narrow commit.
- No new Python API is introduced; the plan preserves existing CLI failure semantics.
- uv.lock is generated by uv lock rather than copied inline because its resolved content is intentionally environment-sized.


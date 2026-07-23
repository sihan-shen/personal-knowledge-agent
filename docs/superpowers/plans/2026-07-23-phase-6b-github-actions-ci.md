# Phase 6B GitHub Actions CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SHA-pinned GitHub Actions workflow that runs the complete locked Python gate and the Phase 6A offline Docker health check.

**Architecture:** One CI workflow triggers on PRs and `main` pushes. A `quality` job runs deterministic Python checks first; a dependent `container` job builds the image with GHA caching, runs its default command, and inspects its health-check contract. Static pytest protects the workflow text locally; GitHub validates runner Docker integration.

**Tech Stack:** GitHub Actions, Python 3.12, uv 0.11.29, Docker Buildx, BuildKit GHA cache, pytest.

## Global Constraints

- Use `.github/workflows/ci.yml` only; do not publish images, deploy, or add a release workflow.
- Trigger only on `pull_request` and pushes to `main`.
- Use `contents: read` only, full Action SHAs, and readable version comments.
- CI must not use `.env`, `PKA_*` secrets, `.pka/`, model caches, user documents, registry login, or image push.
- Quality commands are exactly `uv sync --locked --all-groups`, pytest, Ruff format/check, mypy, `pka health`, and `uv lock --check`.
- Container runs only after quality succeeds and must stay offline.
- Local WSL push instructions use the configured `git push` command.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `.github/workflows/ci.yml` | Trigger, least privilege, concurrency, quality gate, Docker build/run. |
| `tests/test_ci_workflow.py` | Dependency-free workflow contract regression test. |
| `README.md` | CI scope and WSL Git push instructions. |
| `docs/learning-log/phase-6b.md` | CI/CD, cache, token, SHA pin learning notes. |

### Task 1: Define CI behavior with a failing local contract test

**Files:**
- Create: `tests/test_ci_workflow.py`

**Interfaces:**
- Consumes: repository-root `.github/workflows/ci.yml` text.
- Produces: three deterministic pytest checks for trigger/security, quality gate, and container gate.

- [ ] **Step 1: Create the failing test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_uses_approved_triggers_permissions_and_sha_pins() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "main" in workflow
    assert "contents: read" in workflow
    assert "cancel-in-progress: true" in workflow
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in workflow
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in workflow
    assert "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b" in workflow
    assert "docker/setup-buildx-action@e468171a9de216ec08956ac3ada2f0791b6bd435" in workflow
    assert "docker/build-push-action@263435318d21b8e681c14492fe198d362a7d2c83" in workflow
    assert "PKA_" not in workflow


def test_ci_runs_the_locked_python_quality_gate() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for command in (
        "uv sync --locked --all-groups",
        "uv run pytest",
        "uv run ruff format --check .",
        "uv run ruff check .",
        "uv run mypy src",
        "uv run pka health",
        "uv lock --check",
    ):
        assert command in workflow


def test_ci_builds_and_runs_the_offline_container_after_quality() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "needs: quality" in workflow
    assert "cache-from: type=gha" in workflow
    assert "cache-to: type=gha,mode=max" in workflow
    assert "docker/build-push-action@263435318d21b8e681c14492fe198d362a7d2c83" in workflow
    assert "context: ." in workflow
    assert "load: true" in workflow
    assert "tags: pka-ci" in workflow
    assert "docker run --rm pka-ci" in workflow
    expected_health_check = (
        "test \"$(docker inspect --format '{{json .Config.Healthcheck.Test}}' "
        "pka-ci)\" = '[\"CMD\",\"pka\",\"health\"]'"
    )
    assert expected_health_check in " ".join(workflow.split())
    assert "docker push" not in workflow
    assert "docker login" not in workflow
```

- [ ] **Step 2: Verify RED**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_ci_workflow.py -v`

Expected: FAIL with `FileNotFoundError` because the workflow does not exist.

### Task 2: Implement the pinned, cached workflow

**Files:**
- Create: `.github/workflows/ci.yml`
- Test: `tests/test_ci_workflow.py`

**Interfaces:**
- Consumes: Task 1 tests and existing `Dockerfile` default command.
- Produces: one non-publishing workflow with `quality` and dependent `container` jobs.

- [ ] **Step 1: Add the workflow**

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.1.7
      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
        with:
          python-version: "3.12"
      - uses: astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b # v8.1.0
        with:
          version: "0.11.29"
          enable-cache: true
      - run: uv sync --locked --all-groups
      - run: uv run pytest
      - run: uv run ruff format --check .
      - run: uv run ruff check .
      - run: uv run mypy src
      - run: uv run pka health
      - run: uv lock --check

  container:
    needs: quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.1.7
      - uses: docker/setup-buildx-action@e468171a9de216ec08956ac3ada2f0791b6bd435 # v3.11.1
      - uses: docker/build-push-action@263435318d21b8e681c14492fe198d362a7d2c83 # v6.18.0
        with:
          context: .
          load: true
          tags: pka-ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - run: docker run --rm pka-ci
      - run: >-
          test "$(docker inspect --format '{{json .Config.Healthcheck.Test}}' pka-ci)"
          = '["CMD","pka","health"]'
```

- [ ] **Step 2: Verify GREEN and format**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_ci_workflow.py -v`

Expected: all three tests pass. Then run Ruff format/check for the test file; expected no findings.

- [ ] **Step 3: Commit the CI contract**

Run: `git add .github/workflows/ci.yml tests/test_ci_workflow.py && git commit -m "ci: add quality and container checks"`

Expected: only workflow and its test are committed.

### Task 3: Document operator and learning boundaries

**Files:**
- Modify: `README.md`
- Create: `docs/learning-log/phase-6b.md`

**Interfaces:**
- Consumes: CI's offline, non-publishing contract and WSL Git workflow.
- Produces: concise user documentation that distinguishes CI from release/deployment.

- [ ] **Step 1: Document CI and WSL Git push**

Add a README CI section explaining that Actions checks the locked Python gate plus an offline Docker build/default health command, and explicitly excludes provider calls, E5 download, mounted data, and image publication. Add this WSL push command:

```bash
git push
```

- [ ] **Step 2: Add learning log**

Create `docs/learning-log/phase-6b.md` covering CI versus CD, lock/cache distinction, SHA-pinned Actions, least privilege, and why Docker health remains offline.

- [ ] **Step 3: Commit documentation**

Run: `git add README.md docs/learning-log/phase-6b.md && git commit -m "docs: explain CI workflow"`

Expected: only documentation files are committed.

### Task 4: Verify and publish the branch intentionally

**Files:**
- Verify only: all Task 1-3 files.

- [ ] **Step 1: Run the complete local quality gate**

Run the existing pytest, Ruff format/check, mypy, health, and lock-check commands with the WSL uv environment variables.

Expected: all pass. Docker is validated by GitHub-hosted CI because this environment lacks Docker.

- [ ] **Step 2: Push with WSL Git only after user authorization**

Run: `git push -u origin <feature-branch>`.

Expected: the remote branch triggers GitHub Actions. Do not run this command until the user explicitly requests push.

## Self-review

- Task 1 protects every approved workflow contract before the YAML exists.
- Task 2 uses only pinned, non-publishing Actions and has no secret path.
- Task 3 documents the CI scope and WSL Git operator boundary.
- Task 4 separates local deterministic validation from remote runner Docker validation.

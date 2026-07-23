# Phase 6B: GitHub Actions CI Design

**Date:** 2026-07-23

**Status:** Approved design

## Goal and scope

Add one GitHub Actions workflow for pull requests and pushes to `main`. It runs the
locked Python quality gate, builds the Phase 6A Docker image, and executes its offline
default health command. It does not publish an image, release, deploy, call a provider,
download E5, or use private knowledge data.

## Constraints and decisions

- Create one `.github/workflows/ci.yml` workflow with a Python 3.12 baseline on
  `ubuntu-latest`.
- Pin every Action to a complete commit SHA and annotate its release version. Pin uv
  at 0.11.29.
- Use top-level `contents: read` only and a workflow/ref concurrency group with
  `cancel-in-progress: true`.
- Use `uv sync --locked --all-groups`, setup-uv's cache, and BuildKit GHA cache.
- Run `quality` first; `container` declares `needs: quality` and has no `if: always()`.
- CI receives no `.env`, provider secret, `.pka/`, model cache, user document, or
  registry credential.
- Local WSL pushes use the user's configured WSL Git; remote runners use their own
  checkout tooling.

## Architecture

```text
pull request or push to main
             |
             v
quality: checkout -> Python 3.12 -> pinned uv/cache
uv sync --locked --all-groups
pytest -> Ruff format -> Ruff lint -> mypy -> pka health -> uv lock --check
             |
             | success only
             v
container: checkout -> pinned Buildx -> GHA BuildKit cache
docker build -t pka-ci
docker run --rm pka-ci
docker inspect pka-ci health-check configuration
```

The quality commands are exactly:

```bash
uv sync --locked --all-groups
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pka health
uv lock --check
```

The container job builds `pka-ci` with `cache-from` and `cache-to` set to `type=gha`,
then runs:

```bash
docker run --rm pka-ci
test "$(docker inspect --format '{{json .Config.Healthcheck.Test}}' pka-ci)" \
  = '["CMD","pka","health"]'
```

The image is not pushed or loaded into a registry. The default `pka health` command is
offline; no mount, environment file, provider setting, E5 cache, or user data is
supplied.

## Cache and failure behavior

Caches accelerate downloads and layers only. `uv sync --locked` still validates the
lock on every run; a cache miss only redownloads packages. BuildKit cache misses run a
full build. A quality failure prevents container validation. Docker setup, build, run,
health-check assertion, or cache failure fails the active job; no partial success is
reported.

## Static regression test

Add dependency-free `tests/test_ci_workflow.py`, reading the controlled workflow text.
It verifies PR/main triggers, `contents: read`, concurrency cancellation, that every
`uses:` reference is a complete SHA-pinned member of the approved Action set,
Python 3.12, uv 0.11.29, cache settings, locked installation, every quality command,
`container` depending on `quality`, Buildx/GHA cache, Docker build/default run/health
inspection, and absence of secrets, `.env`, `.pka/` mounts, registry login/push, and
model download instructions.

This is fast local regression feedback. A GitHub Actions run is the integration check
for the runner image, Action execution, cache service, and Docker daemon.

## Documentation and acceptance

README will state that CI validates only the offline quality/container contract, not
provider calls, E5 download, or host-mount persistence. It will add the WSL Git push
example. A Phase 6B learning log explains CI versus CD, cache
correctness, least-privilege tokens, SHA pins, and offline container checks.

Acceptance requires PR and `main` push triggers; the full Python gate; a dependent
container build/default-run/health inspection; full SHA pins and `contents: read`;
no private data, secrets, registry login, or publish path; passing local static tests;
and one post-merge GitHub Actions run.

## Out of scope

Registry publishing, releases, deployment/CD, coverage upload, Dependabot, a Python
matrix, E5/RAG/LLM/evaluation/provider testing, host mount persistence, Git
configuration changes, and automatic pushes are out of scope.

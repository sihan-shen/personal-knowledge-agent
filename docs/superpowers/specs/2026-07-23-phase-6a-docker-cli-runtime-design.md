# Phase 6A: Docker CLI Runtime Design

**Date:** 2026-07-23

**Status:** Approved design

## Goal

Package the existing `pka` command-line application in a reproducible Docker
image. The image must run the production CLI without bundling user knowledge,
model downloads, credentials, or other runtime state.

Phase 6A is deployment packaging only. It does not add an HTTP API, Docker
Compose, GPU support, a baked embedding model, or a new RAG, Agent, memory,
graph, or evaluation capability.

## Context and constraints

- The application is a Python 3.12+ CLI with the `pka` console entry point.
- Its local persistent data defaults to paths below `.pka/`, relative to the
  current working directory.
- `E5Embedder` lazily loads `intfloat/multilingual-e5-small` through
  `sentence-transformers`. The first semantic or hybrid operation can download
  model artifacts.
- The repository currently has `pyproject.toml`, but no tracked `uv.lock`.
  A reproducible container build therefore requires generating and committing a
  lock file.
- `.env`, `.pka/`, model caches, and user-owned `papers/`, `references/`, and
  `research/` directories must never be copied to the image or committed.

## Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Dependency resolution | Generate and commit `uv.lock`; build with the locked production dependency set | Identical source revisions resolve to identical dependency versions. |
| Image layout | Multi-stage build | Keeps build tooling out of the runtime image. |
| Runtime identity | A dedicated non-root `pka` user | Avoids writing application data as root into bind-mounted host directories. |
| Embedding model | Download at runtime; persist the cache with an explicit bind mount | Keeps image size and build time practical while avoiding repeated downloads. |
| Persistent application data | Explicit host bind mount for `.pka/` | Makes the data location, backup responsibility, and lifecycle visible. |
| Container entry point | `pka`, with default command `health` | Makes normal CLI usage concise and gives a safe offline default. |
| Health check | `pka health` only | It verifies CLI availability without downloading models or calling an LLM. |
| Test strategy | Static Docker-contract test plus opt-in local Docker smoke commands | Python quality gates stay deterministic when Docker is absent. |

## Alternatives considered

### Bake the E5 model into the image

This gives an offline first semantic query, but makes image builds network-bound,
slow, and large. Every model update requires rebuilding and redistributing the
image. Phase 6A instead downloads the model on first use and mounts its cache.

### Install dependencies directly from `pyproject.toml` on every build

This is shorter, but an unconstrained resolver can select different dependency
versions in future builds. A committed `uv.lock` is the deployment contract.

### Use a named Docker volume

A Docker-managed volume is simple once Docker is the only environment. An
explicit bind mount is more transparent for a learning project: users can
inspect, back up, and delete their `.pka/` data and model cache directly.

### Add Docker integration tests to `pytest`

Such tests would make the normal development quality gate depend on a Docker
daemon, image-build permissions, and possibly a network. The repository will
test the Dockerfile's invariant text in Python; Docker commands remain explicit
release smoke checks.

## Runtime architecture

```text
host directory                          container
--------------                          ---------
./.pka/            -- bind mount -->   /workspace/.pka/
./.pka-model-cache/ -- bind mount -->  /home/pka/.cache/
./documents/       -- bind mount -->   /input/ (read-only recommended)

                                            +-----------------------+
                                            | Python runtime image  |
                                            | - application package |
                                            | - production venv     |
                                            | - non-root pka user   |
                                            | ENTRYPOINT: pka       |
                                            | CMD: health           |
                                            +-----------------------+
```

The Dockerfile has two stages:

1. A builder stage copies only the dependency manifest and lock file, installs
   production dependencies into a virtual environment, then installs the
   application from the checked-out source.
2. A smaller runtime stage copies the virtual environment and application
   artifacts, sets `/workspace` as the working directory, exposes the `pka`
   executable on `PATH`, and switches to the `pka` user.

The runtime sets the Hugging Face and Sentence Transformers cache locations
below `/home/pka/.cache/`. Mounting that parent directory therefore preserves
all embedding-model artifacts across containers without assuming a particular
internal cache layout.

## Data and command flow

1. The user builds an image from tracked source, `pyproject.toml`, and
   `uv.lock`. `.dockerignore` excludes generated and sensitive paths.
2. The user creates host directories for `.pka/` and the model cache, then
   binds them into the container.
3. `docker run ... pka:local health` invokes the default command and does not
   use a model or remote provider.
4. For ingestion, the user binds a document directory at `/input` and invokes
   `pka ingest /input/example.md`. SQLite files are written through the
   `/workspace/.pka/` mount.
5. On the first hybrid or semantic query, `sentence-transformers` downloads E5
   into `/home/pka/.cache/`; later containers reuse the mounted cache.
6. For an LLM-backed command, the user supplies variables at runtime, for
   example with `--env-file .env`. The image never contains `.env`.

## Error handling and security boundaries

Docker does not alter existing CLI error semantics.

- If the E5 download or cache write fails, the existing embedding error reaches
  the CLI and produces a non-zero exit. The container must not silently replace
  hybrid or semantic retrieval with keyword-only retrieval.
- If `.pka/` is missing, the CLI can still operate with container-local data.
  That data is intentionally lost when the container is removed; the README
  states this explicitly.
- If a mounted `.pka/` path is not writable, existing SQLite or storage errors
  produce a non-zero exit.
- A missing LLM configuration or API key affects only commands that already
  require a provider. `pka health` remains offline.
- The container runs as `pka`, not root. The documentation requires a writable
  host mount that is compatible with that user; users can choose a different
  output path through the existing CLI flags if needed.
- Credentials are runtime environment variables only. They must not be copied,
  supplied as build arguments, emitted in documentation output, or committed.
- User documents are supplied by a mount and can be mounted read-only. Remote
  LLM commands retain the existing privacy boundary: evidence sent to a
  configured provider leaves the local machine.

## Files and interfaces

| Path | Change | Responsibility |
| --- | --- | --- |
| `uv.lock` | Add and track | Deterministic dependency graph for local and image builds. |
| `.gitignore` | Stop ignoring `uv.lock` | Allows the lock file to be version controlled. |
| `Dockerfile` | Add | Defines builder/runtime stages, non-root execution, CLI entry point, and health check. |
| `.dockerignore` | Add | Removes credentials, generated data, caches, VCS metadata, and user files from the build context. |
| `tests/test_container_config.py` | Add | Checks durable Dockerfile and `.dockerignore` configuration contracts without Docker. |
| `README.md` | Update | Provides build, health, ingest, cache, persistence, and LLM environment examples. |

No domain, application, or infrastructure Python interface changes are needed.

## Test and verification strategy

The new Python test reads the checked-in Docker configuration and asserts these
contracts:

- `Dockerfile` creates and selects a non-root `pka` user;
- the runtime exposes `pka` as its entry point, defaults to `health`, and uses
  `pka health` for the health check;
- the build uses `uv.lock` rather than an unconstrained production install;
- `.dockerignore` excludes `.env`, `.pka/`, virtual environments, caches,
  `.git`, and `papers/`, `references/`, and `research/`.

The normal quality gate remains:

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pka health
```

When Docker is available, release verification additionally runs:

```bash
docker build -t pka:local .
docker run --rm pka:local
```

The README's mounted health and ingest commands are manual acceptance checks.
The first semantic or hybrid operation may require network access to obtain E5;
the cache mount makes subsequent operations independent of a repeat download.

## Acceptance criteria

- A fresh clone can resolve dependencies from a tracked `uv.lock` and build the
  image without reading user-generated or secret files.
- `docker run --rm pka:local` invokes `pka health`, exits successfully, and
  never calls an LLM or downloads E5.
- The image executes as a non-root user and has an offline health check.
- The documented `.pka/` and cache mounts preserve SQLite data and model
  artifacts across container deletion and recreation.
- A document mounted at `/input` can be ingested into the mounted `.pka/`
  directory.
- `.env`, user documents, `.pka/`, and model caches do not enter the image.
- All existing Python quality gates pass, and Docker smoke commands pass where
  Docker is available.

## Out of scope

- HTTP service, authentication, multi-user tenancy, or a browser UI;
- Docker Compose, Kubernetes, cloud deployment, CI publishing, or image
  registries;
- GPU images or CUDA dependency variants;
- pre-downloading or baking E5 into an image;
- encrypting SQLite data, changing the memory safety policy, or changing
  provider privacy behavior.

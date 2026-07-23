# Phase 6A: Docker CLI Runtime

## What this phase delivers

Phase 6A packages the existing production CLI as a Docker image. It does not change
retrieval, Agent routing, memory, graph answering, or evaluation. The image defaults
to offline `pka health` and runs as a dedicated non-root user.

## Reproducibility is more than a Dockerfile

A Dockerfile repeats installation instructions, but cannot alone select the same
Python dependencies tomorrow. `uv.lock` records the resolved graph, and Docker builds
with `uv sync --locked`. Dependency changes are therefore deliberate and reviewable.

## Image state versus runtime state

The image contains application code and production dependencies. It excludes `.pka/`,
source documents, `.env`, and model caches. Users pass those explicitly:

- `.pka/` mounts at `/workspace/.pka/` for SQLite knowledge, memory, graph, and
  evaluation data.
- A host cache mounts at `/home/pka/.cache/` so the first E5 download is reused.
- Documents mount at `/input` and should be read-only for ingestion.
- Provider credentials are runtime environment variables, never image layers or build
  arguments.

Deleting an unmounted container deletes its local state; deleting a mounted container
leaves the host directory intact.

## Why health is deliberately narrow

`pka health` proves the CLI starts. It does not prove that E5 can download, a model
cache is writable, or an LLM provider is configured. Those dependencies run only when
the requested command needs them, keeping health deterministic.

# Personal Knowledge Agent

A personal knowledge-management system built incrementally with RAG and Agent patterns.

## Current scope

Phase 1A provides persistent Markdown ingestion and Chinese/English BM25 keyword
retrieval. It does not yet call an LLM, generate answers, or use embeddings and a
vector database.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --all-groups
```

## Health check

```bash
uv run pka health
```

## Build a local knowledge base

By default, the CLI stores its SQLite index in `.pka/knowledge.db` relative to the
current directory:

```bash
uv run pka ingest notes/rag.md
uv run pka search "RAG 如何检索 knowledge"
```

Use `--db` to select an explicit knowledge base and `--limit` to return between
1 and 20 chunks:

```bash
uv run pka ingest notes/rag.md --db data/knowledge.db
uv run pka search "retrieval context" --limit 3 --db data/knowledge.db
```

Ingesting the same resolved source path replaces its previous chunks atomically.
Search output contains BM25 scores, source paths, heading paths, and text snippets.
It returns evidence chunks rather than an LLM-written answer.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

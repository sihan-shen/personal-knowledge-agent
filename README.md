# Personal Knowledge Agent

A personal knowledge-management system built incrementally with RAG and Agent patterns.

## Current scope

Phase 1B provides persistent Markdown ingestion plus Chinese/English lexical,
semantic, and hybrid retrieval. Embeddings run locally on CPU and are stored in
SQLite alongside the BM25 index. The project does not yet call an LLM or generate
answers.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --all-groups
```

The dependency lock selects the CPU-only PyTorch build. The first ingest command
downloads `intfloat/multilingual-e5-small` from Hugging Face; later commands reuse
the default Hugging Face cache (normally `~/.cache/huggingface/hub`). Model
inference is explicitly configured for CPU, so a GPU is not required. Keep the
machine online for the first model download.

### WSL environment

This repository's current WSL location is:

```bash
/mnt/c/Users/19850/Projects/RAG-Agent
```

Run project commands from that directory (not the previous native-Linux path
under `/home/sihan/文档/...`). On this Windows-mounted filesystem, keep Python
temporary files in WSL's Linux filesystem and ask uv to copy cached packages:

```bash
export TMPDIR=/tmp
export UV_LINK_MODE=copy
```

These exports are already configured in `~/.bashrc`. Open a new shell or run
`source ~/.bashrc` after changing that file.

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

Search defaults to hybrid retrieval. Select a mode explicitly when comparing
retrieval behavior:

```bash
uv run pka search "exact keyword" --mode lexical
uv run pka search "a related idea using different words" --mode semantic
uv run pka search "RAG 如何减少幻觉" --mode hybrid
```

- `lexical` ranks exact Chinese/English token matches with BM25 and does not run
  the embedding model for the query.
- `semantic` embeds the query and exact-scans the stored vectors by cosine
  similarity.
- `hybrid` combines the BM25 and semantic rankings with Reciprocal Rank Fusion
  (RRF). It is the default because either retriever may contribute a useful
  candidate.

E5 uses different inputs for the two sides of retrieval: indexed chunk text is
encoded as `passage: <heading and body>`, while user text is encoded as
`query: <query>`. The adapter adds these prefixes automatically.

Use `--db` to select an explicit knowledge base and `--limit` to return between
1 and 20 chunks:

```bash
uv run pka ingest notes/rag.md --db data/knowledge.db
uv run pka search "retrieval context" --limit 3 --db data/knowledge.db
```

Ingesting the same resolved source path replaces its previous chunks atomically.
The same transaction replaces postings and float32 embedding vectors. A Phase 1A
database is migrated automatically, but its existing documents must be reingested
before semantic retrieval can search them. Search output labels the mode and
contains retrieval scores, source paths, heading paths, and text snippets. It
returns evidence chunks rather than an LLM-written answer.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

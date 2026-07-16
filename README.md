# Personal Knowledge Agent

A personal knowledge-management system built incrementally with RAG and Agent patterns.

## Current scope

Phase 2A provides persistent Markdown ingestion, Chinese/English lexical, semantic,
and hybrid retrieval, plus evidence-grounded answers through an OpenAI-compatible
Chat Completions endpoint. Embeddings run locally on CPU and are stored in SQLite
alongside the BM25 index. Answer generation is deliberately not an Agent yet:
`pka ask` always retrieves, while later routing will decide whether retrieval or a
different tool is needed.

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

## Configure answer generation

`pka ask` requires an API key in the shell. The defaults target DeepSeek's
OpenAI-compatible endpoint and the `deepseek-v4-flash` model:

```bash
export PKA_LLM_API_KEY="your-key"
export PKA_LLM_BASE_URL="https://api.deepseek.com"  # optional
export PKA_LLM_MODEL="deepseek-v4-flash"            # optional
```

`.env.example` documents these names, but the CLI does not automatically load a
`.env` file. Never commit the real key. The endpoint and model are configurable so
the application service remains independent of one provider.

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

## Ask an evidence-grounded question

After ingesting a document, ask a question against the same database:

```bash
uv run pka ask \
  "RAG 为什么能减少模型脱离资料的猜测？" \
  --mode hybrid \
  --limit 5 \
  --db data/knowledge.db
```

`ask` defaults to hybrid retrieval and five evidence chunks. Its evidence limit is
1 to 10, a smaller boundary than `search` because every selected chunk consumes
remote-model context and can add API cost. An accepted answer contains inline
citations such as `[S1]` and a `Sources` section mapping only those cited labels
back to local paths and heading paths. If retrieval returns nothing, the CLI refuses
locally without calling the model. The model can also declare that the retrieved
evidence is insufficient; this produces no `Sources` section.

The provider receives the question plus each selected chunk's request-local citation
ID, heading path, and text. Absolute source paths, stable chunk IDs, retrieval scores,
the API key, and the embedding vectors remain local. Retrieved text can still be
private, so configure only a provider you trust and understand its data policy before
using `ask`.

The adapter requests one non-streaming JSON object through Chat Completions. JSON
mode only helps the provider produce syntactically valid JSON; the project still
validates the exact fields, answer status, citation IDs, body citation order, and
membership in the supplied evidence. It does not repair malformed responses.
Retrieved documents are serialized as untrusted evidence data and the system prompt
tells the model not to follow instructions inside them. That boundary, data
minimization, and local citation validation are defence in depth, not a mathematical
guarantee against prompt injection.

The SDK may retry transient failures up to two times, so one CLI command can make
more than one billable request. Authentication, permission, bad-request, rate-limit,
timeout, connection, server, truncation, schema, and citation failures become concise
CLI errors without echoing provider payloads. Automated tests use deterministic
fakes and never call DeepSeek or incur API cost.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

# Phase 1B Learning Log

## What this phase built

The persistent Markdown knowledge base now stores one local E5 vector per chunk,
supports exact cosine semantic search, and combines semantic and BM25 rankings with
Reciprocal Rank Fusion. Users can compare `lexical`, `semantic`, and `hybrid` modes;
hybrid is the CLI default.

## Embedding boundary and E5 inputs

`EmbeddingModel` separates retrieval from a particular inference library. Production
uses a lazy `sentence-transformers` adapter for
`intfloat/multilingual-e5-small`; tests use a deterministic fake and never download
model weights.

E5 expects asymmetric prefixes. Documents are encoded as
`passage: <heading and chunk body>`, and questions are encoded as
`query: <question>`. Both sides are L2-normalized. Their dot product therefore equals
cosine similarity and can be compared without renormalizing every stored vector.

The adapter loads once on first use, forces CPU inference, and converts model loading,
encoding, shape, finite-value, and zero-vector failures into a project-level
`EmbeddingError`.

## SQLite schema and consistency

Schema version 2 adds an `embeddings` table keyed by `chunk_id`. Each vector stores
its model identifier, dimension, and explicit little-endian float32 BLOB. The model
identifier prevents meaningless comparisons between vectors from different models;
the dimension and BLOB-length checks detect corrupt or incompatible records.

Embedding generation finishes before the replacement transaction begins. The
transaction then replaces the document, chunks, postings, and vectors together. A
model failure cannot delete the old document, while an SQL failure rolls back all
new lexical and semantic data. Foreign-key cascades remove obsolete vectors during
successful reingestion.

A version 1 database migrates without losing its BM25 rows. Existing chunks have no
vectors until they are reingested, so lexical search remains available during that
transition.

## Exact semantic retrieval

The index embeds a query once, reads compatible vectors, and computes normalized dot
products in Python. Exact scanning is intentionally transparent: every stored vector
is considered, ranking is deterministic, and no approximate-index parameters can
hide retrieval behavior.

This is appropriate for a small personal corpus. An ANN index or vector database
becomes worth evaluating when measurements show either roughly tens of thousands of
chunks or semantic-search p95 latency above about 200 ms on the target machine. The
decision should be driven by a benchmark rather than the availability of a more
complex dependency.

## Reciprocal Rank Fusion

BM25 and cosine scores have unrelated scales, so adding them directly would require
calibration. RRF uses rank positions instead. For each retriever it adds
`1 / (60 + rank)` to a candidate; a candidate found by only one retriever remains
eligible. Final ties break by stable chunk ID.

The deterministic synonym regression demonstrates the intended benefit: lexical
search can miss a relevant chunk expressed with different words, semantic search can
recover it, and hybrid search can retain both that semantic candidate and a separate
exact-keyword candidate.

## Real-model CPU smoke test

The manual smoke test ran successfully on Python 3.14 with the locked
`torch 2.13.0+cpu` build. Ingesting `rag_notes.md` loaded the real
`intfloat/multilingual-e5-small` model and persisted three vectors. Database
inspection confirmed all three rows used the expected model identifier and 384
dimensions.

The semantic query `如何用外部知识减少模型幻觉` ranked the Chinese RAG explanation
first with cosine score `0.839`. The hybrid query `RAG 检索` also ranked that chunk
first with RRF score `0.033`. This verifies that the production adapter, prefixes,
CPU inference, float32 persistence, exact scan, and fusion work together rather than
only behind fakes.

Two operational limitations were visible. Hugging Face warns that unauthenticated
downloads have lower rate limits, although no token is required for this public
model. Each CLI process also reloads weights from the local cache; the download is
one-time, but startup and CPU inference are noticeably slower than lexical search.

On this WSL repository, a previously interrupted dependency sync left the
Windows-mounted `.venv` with a partial PyTorch install. Recreating the same locked
environment under `/tmp` completed normally and exposed `torch.distributed`; the
failure was an interrupted environment installation, not a model or code
incompatibility. Dependency installation on the native Linux filesystem is much
faster than writing PyTorch's many files to `/mnt/c`.

## Validation completed

Automated tests cover lazy E5 construction and prefixes, vector validation, schema
migration, float32 persistence, replacement rollback, cosine ordering, model and
dimension incompatibility, RRF behavior, application mode forwarding, and all CLI
modes. The separate real-model CPU smoke test also passed with the observations
above.

## Commands I can explain

```bash
uv sync --all-groups
uv run pka ingest tests/fixtures/knowledge/rag_notes.md --db /tmp/pka-e5.db
uv run pka search "如何用外部知识减少模型幻觉" --mode semantic --db /tmp/pka-e5.db
uv run pka search "RAG 检索" --mode hybrid --db /tmp/pka-e5.db
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

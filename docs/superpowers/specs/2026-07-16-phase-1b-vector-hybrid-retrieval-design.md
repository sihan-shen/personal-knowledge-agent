# Phase 1B: Local Vector and Hybrid Retrieval Design

## Context and goal

Phase 1A stores Markdown chunks in SQLite and ranks lexical matches with BM25.
This phase adds local semantic retrieval for the same chunks, then combines both
signals. The result remains a local, persistent command-line knowledge base.

The phase is intentionally limited to Markdown sources and retrieval. PDF,
web ingestion, answer generation, agents, and evaluation remain later phases.

## Decisions

| Concern | Decision | Reason |
| --- | --- | --- |
| Embedding model | `intfloat/multilingual-e5-small` | It supports Chinese and English and is practical on the available CPU-only WSL environment. |
| Embedding runtime | `sentence-transformers` | It exposes the model's document/query encoding workflow without hiding it behind a hosted API. |
| Document input | `passage: <heading and chunk text>` | E5 was trained with distinct query and passage prefixes. |
| Query input | `query: <user query>` | This is required for compatible E5 query embeddings. |
| Vector normalization | L2-normalize each vector before storage and querying | The dot product then equals cosine similarity and avoids repeated normalization at search time. |
| Persistence | An `embeddings` table in the existing SQLite database | A single database keeps chunk identity, deletion, and replacement understandable and durable. |
| Vector format | Little-endian float32 BLOB plus dimension and model identifier | It is compact, explicit, and avoids an extra serialization dependency. |
| Vector search | Exact cosine scan in Python | The expected personal corpus is small; this makes ranking and tests transparent. |
| Hybrid fusion | Reciprocal Rank Fusion (RRF), with `k = 60` | It combines differently scaled BM25 and cosine scores without fragile score normalization. |
| CLI modes | `lexical`, `semantic`, and `hybrid`; `hybrid` is the default | This permits comparison and makes the system's decision visible. |

## Alternatives rejected for this phase

### Dedicated vector database

Chroma and LanceDB would provide a richer vector-oriented API. They also add
storage concepts and dependency complexity before the project has enough data
to need them. They are suitable candidates once exact scan latency becomes a
measured problem.

### Approximate nearest-neighbour indexes

FAISS and HNSW make large collections fast but introduce index build,
parameter-tuning, and platform-dependency concerns. Exact scanning is easier
to inspect and is adequate for the initial personal corpus.

### Direct score interpolation

Adding a weighted BM25 score to a weighted cosine score requires calibrating
two incomparable scales. RRF only relies on rank and is therefore the safer
initial hybrid strategy. Weighted scoring can be evaluated later.

## Data model and consistency

The database schema is upgraded from version 1 to version 2. A migration adds
the table below; existing BM25 documents and postings remain usable.

```sql
CREATE TABLE embeddings (
    chunk_id TEXT PRIMARY KEY
        REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX idx_embeddings_model_id ON embeddings(model_id);
```

`chunk_id` is the stable identity already created by the chunker. When a source
is re-ingested, the existing `documents` row is deleted inside the transaction.
SQLite cascade deletion removes its chunks, postings, and embeddings. The new
chunks, postings, and their prepared vectors are then inserted before commit.
Thus a successful replacement contains matching lexical and semantic data; a
failed SQLite transaction leaves the previous version intact.

Embedding generation happens before the transaction because it can be slow and
does not mutate the database. If generation fails, ingestion stops before any
stored document data is deleted.

The index accepts only one configured `model_id`. Searching a database that
contains a different model identifier raises a domain storage error with a
clear re-ingestion instruction. Silently comparing vectors from different
models would be invalid.

## Components and interfaces

### Domain

`EmbeddingModel` is a protocol with two explicit operations:

```python
embed_documents(texts: Sequence[str]) -> list[Embedding]
embed_query(query: str) -> Embedding
```

`Embedding` is an immutable record containing the `model_id` and normalized
float values. The constructor validates that the vector is non-empty and has
finite values.

`SearchMode` is a string enum: `lexical`, `semantic`, or `hybrid`. Search
results retain the existing `SearchResult` shape so callers do not need to know
which ranking implementation produced a result.

### Infrastructure

`E5Embedder` lazily loads `intfloat/multilingual-e5-small`. It supplies the
E5 prefixes, normalizes model output, converts it to float values, and wraps
runtime/model failures in a project domain error.

`SQLiteKeywordIndex` remains the concrete index name during this phase to
avoid an unrelated public rename. It receives an `EmbeddingModel`, prepares
chunk embeddings during replacement, persists them with BM25 records, performs
semantic ranking, and applies RRF. The name will be reconsidered when Phase 1
has a broader retrieval abstraction.

### Application and CLI

`IngestService` remains responsible for load, chunk, and replace. The index
owns embedding generation because that work must be aligned exactly with its
database transaction.

`SearchService.search` accepts a search mode, validates it, and delegates to
the index. `pka search` gains `--mode`, defaulting to `hybrid`. It keeps
`--limit` and `--db` unchanged. Output labels the chosen mode so a user can
compare modes honestly.

## Data flow

```text
Markdown path
  -> MarkdownLoader
  -> MarkdownChunker
  -> E5Embedder.embed_documents("passage: ...")
  -> SQLite transaction: documents + chunks + postings + embeddings

Query
  -> E5Embedder.embed_query("query: ...")
  -> BM25 ranked chunk IDs
  -> cosine-similarity ranked chunk IDs
  -> RRF fusion (or selected single mode)
  -> SearchResult list
```

For hybrid retrieval, each retriever receives an internal candidate limit at
least as large as the requested result limit. RRF sums `1 / (60 + rank)` for
each ranking in which a chunk appears, then breaks ties by chunk ID. A chunk
may appear in one ranking only; it still remains eligible.

## Error handling

- An empty or invalid query raises `InvalidQueryError` before model work.
- A missing knowledge base raises `KnowledgeBaseEmptyError` consistently for
  lexical, semantic, and hybrid modes.
- Model loading, encoding, non-finite values, invalid BLOB lengths, and mixed
  model identifiers raise a project `EmbeddingError` or `IndexStorageError`.
- A schema version other than 1 or 2 is rejected rather than guessed.
- SQLite errors roll back the active replacement transaction and include the
  source or database path in the message.

## Test strategy

Unit tests use a deterministic fake embedder rather than downloading a model.
They cover E5 prefix construction, normalization validation, BLOB round trips,
cosine ordering, RRF ordering and deterministic ties, model mismatch, and
schema migration.

Integration tests create a temporary SQLite database and verify that:

1. one ingestion writes embeddings for every chunk;
2. re-ingestion removes old vectors as well as old postings;
3. semantic retrieval can return a synonym-oriented fixture that lexical
   retrieval misses;
4. hybrid retrieval includes useful lexical and semantic candidates;
5. the CLI validates and reports all three modes.

One optional manual smoke test runs the real E5 model after dependencies are
installed. It is deliberately not part of the default test suite because model
download and inference should not be required for deterministic unit tests.

## Non-goals

- PDF, webpage, and code-document ingestion.
- Hosted embedding APIs and API key configuration.
- GPU acceleration, batching/throughput tuning, or ANN indexing.
- LLM answer generation, agent routing, memory, knowledge graphs, and answer
  evaluation.

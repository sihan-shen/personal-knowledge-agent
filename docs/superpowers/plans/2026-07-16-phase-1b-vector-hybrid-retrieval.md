# Phase 1B Vector and Hybrid Retrieval Implementation Plan

**Goal:** Add persistent local E5 embeddings, semantic cosine retrieval, and RRF
hybrid retrieval to the Markdown knowledge base.

**Architecture:** `EmbeddingModel` separates model inference from storage. The
existing SQLite index owns one transaction that writes BM25 and float32 vector
records for each chunk. It can then rank lexical, semantic, or RRF-fused
results. The CLI composes the real lazy E5 adapter while all automated tests
use a deterministic fake.

**Tech stack:** Python 3.12+, SQLite, standard-library `math` and `struct`,
`sentence-transformers`, pytest, Ruff, and mypy.

> Run commands in this WSL environment with `TMPDIR=/tmp` and
> `UV_LINK_MODE=copy` when necessary.

## File map

- Modify `pyproject.toml`: add the embedding runtime.
- Modify `domain/errors.py`: add embedding-specific project failures.
- Modify `domain/models.py`: add validated immutable embedding and search mode
  records.
- Modify `domain/protocols.py`: make the embedding and retrieval-mode boundary
  explicit.
- Create `infrastructure/e5_embedder.py`: lazy real-model adapter and E5 input
  prefixes.
- Create `infrastructure/retrieval_fusion.py`: deterministic RRF ranking.
- Modify `infrastructure/sqlite_keyword_index.py`: schema v1-to-v2 migration,
  transactional vector persistence, cosine search, and mode dispatch.
- Modify `application/search.py`: validate and forward the requested mode.
- Modify `cli.py`: compose E5 and expose `--mode`.
- Modify `tests/conftest.py` or create an equivalent test helper: deterministic
  fake embedder shared by unit and integration tests.
- Modify/add unit, integration, CLI, fixture, README, and learning-log files.

## Task 1: Declare the embedding boundary and deterministic fake

**Files:**
- Modify `src/personal_knowledge_agent/domain/errors.py`
- Modify `src/personal_knowledge_agent/domain/models.py`
- Modify `src/personal_knowledge_agent/domain/protocols.py`
- Create `tests/helpers.py`
- Modify `tests/unit/test_models.py`

1. Write failing tests for a frozen `Embedding(model_id, values)` record:
   reject blank model IDs, empty vectors, and non-finite values; retain valid
   tuples of floats. Test the string enum values `lexical`, `semantic`, and
   `hybrid`.
2. Add `EmbeddingError(KnowledgeError)` for model loading, encoding, and
   incompatible vectors.
3. Add `SearchMode(str, Enum)` and `Embedding` to domain models. A helper may
   normalize vectors, but it must reject a zero vector rather than divide by
   zero.
4. Add `EmbeddingModel` protocol methods:

   ```python
   @property
   def model_id(self) -> str: ...

   def embed_documents(self, texts: Sequence[str]) -> list[Embedding]: ...
   def embed_query(self, query: str) -> Embedding: ...
   ```

   Extend `KnowledgeIndex.search` with `mode: SearchMode`.
5. Implement a typed `FakeEmbedder` in tests. It maps known text to specified
   vectors, records passage/query calls, and normalizes inputs just like the
   production boundary. It must never import or download the real model.
6. Run RED then GREEN:

   ```bash
   uv run pytest tests/unit/test_models.py -v
   uv run mypy src
   ```

7. Commit: `feat: define embedding retrieval contracts`.

## Task 2: Implement an isolated E5 adapter

**Files:**
- Modify `pyproject.toml`
- Create `src/personal_knowledge_agent/infrastructure/e5_embedder.py`
- Create `tests/unit/test_e5_embedder.py`

1. Add `sentence-transformers` as a runtime dependency with a modern v5 lower
   bound. Run `uv lock`/`uv sync` only after the tests establish the adapter
   boundary.
2. Write RED tests using a fake SentenceTransformer constructor/model object:
   construction is lazy; document calls contain `passage: `; query calls
   contain `query: `; outputs become normalized `Embedding` values; empty,
   wrong-shaped, zero, non-finite, and runtime failures become `EmbeddingError`.
3. Implement `E5Embedder` with model ID `intfloat/multilingual-e5-small`.
   Import the heavy optional dependency inside the loader, set CPU device
   explicitly, and load once per adapter instance. Convert encoder output to
   Python floats before domain validation. Do not put Hugging Face calls in
   module import time.
4. Run:

   ```bash
   uv run pytest tests/unit/test_e5_embedder.py -v
   uv run ruff check src tests
   uv run mypy src
   ```

5. Commit: `feat: add local E5 embedding adapter`.

## Task 3: Upgrade SQLite storage and write vectors atomically

**Files:**
- Modify `src/personal_knowledge_agent/infrastructure/sqlite_keyword_index.py`
- Modify `tests/integration/test_sqlite_keyword_index.py`

1. Change index construction to require an `EmbeddingModel`; update existing
   integration helpers to inject `FakeEmbedder`.
2. Write failing integration tests for:
   - a fresh database at `PRAGMA user_version = 2` with an `embeddings` table;
   - v1 migration preserving existing BM25 rows and adding the table;
   - one stored vector per chunk with expected `model_id` and dimension;
   - BLOB float32 round-trip and invalid byte length handling;
   - replacement deleting old embedding rows along with chunks/postings;
   - embedding-generation failure preserving the previously stored document;
   - duplicate/failing SQL insertion rolling back both postings and vectors.
3. Implement a deliberate v1-to-v2 migration. Fresh schema creation includes
   `embeddings`; version 1 adds it and advances the pragma; unsupported versions
   still fail clearly.
4. Encode each chunk using the same searchable passage text used for retrieval:
   headings followed by chunk body. Generate all embeddings before `BEGIN`.
   Validate count, model ID, and common dimension before deleting a source.
5. Serialize vectors with `struct.pack` as explicit little-endian float32.
   On reading, validate byte length against the stored dimension before
   unpacking. Add rows under the existing replacement transaction so foreign-key
   cascade removes them with old chunks.
6. Run:

   ```bash
   uv run pytest tests/integration/test_sqlite_keyword_index.py -v
   uv run mypy src
   ```

7. Commit: `feat: persist embeddings with SQLite chunks`.

## Task 4: Implement semantic ranking and RRF fusion

**Files:**
- Create `src/personal_knowledge_agent/infrastructure/retrieval_fusion.py`
- Create `tests/unit/test_retrieval_fusion.py`
- Modify `src/personal_knowledge_agent/infrastructure/sqlite_keyword_index.py`
- Modify `tests/integration/test_sqlite_keyword_index.py`

1. Add RED unit tests for RRF:
   both rankings contribute `1 / (60 + rank)`; a chunk from only one ranking is
   eligible; equal scores break by chunk ID; requested limit is respected.
2. Implement a pure `reciprocal_rank_fusion` function over ranked
   `SearchResult` sequences. Do not mix database reads or model calls into it.
3. Add RED integration tests with fixed two-dimensional embeddings. Verify
   semantic search orders by normalized dot product, handles no semantic match,
   rejects a stored incompatible `model_id`, and keeps a deterministic tie.
4. Extend `SQLiteKeywordIndex.search(query, *, limit, mode)`:
   - lexical: preserve current BM25 ranking;
   - semantic: encode the query once and exact-scan compatible stored vectors;
   - hybrid: call both rankings then RRF-fuse them.

   Raise the existing empty-index error before model work. Reject punctuation
   queries in all modes. Query a candidate set at least as large as `limit`.
5. Add a regression test with a semantic synonym fixture where BM25 returns no
   lexical result but the fake vector ranks the relevant chunk; then assert
   hybrid retains the semantic candidate and an exact keyword candidate.
6. Run unit/integration checks:

   ```bash
   uv run pytest tests/unit/test_retrieval_fusion.py tests/integration/test_sqlite_keyword_index.py -v
   uv run ruff check src tests
   uv run mypy src
   ```

7. Commit: `feat: add semantic and hybrid retrieval`.

## Task 5: Expose retrieval modes through application and CLI

**Files:**
- Modify `src/personal_knowledge_agent/application/search.py`
- Modify `src/personal_knowledge_agent/cli.py`
- Modify `tests/unit/test_application_services.py`
- Modify `tests/test_cli.py`

1. Update typed fakes and write RED service tests that prove mode forwarding
   and invalid mode rejection.
2. Extend `SearchService.search` with `mode: SearchMode = SearchMode.HYBRID`.
3. Add `pka search --mode {lexical,semantic,hybrid}`. Default to `hybrid`,
   construct one lazy `E5Embedder` for ingest/search, and print the selected
   mode before results. Keep existing `--limit` and `--db` behavior.
4. Ensure parser errors explain invalid modes. Test the CLI with dependency
   injection or a patched adapter so automated tests do not download E5.
5. Run:

   ```bash
   uv run pytest tests/unit/test_application_services.py tests/test_cli.py -v
   uv run ruff check src tests
   uv run mypy src
   ```

6. Commit: `feat: expose hybrid retrieval modes in CLI`.

## Task 6: Document and validate the real local workflow

**Files:**
- Modify `README.md`
- Create `docs/learning-log/phase-1b.md`

1. Document first-run model download, E5 prefixes, default model cache
  behavior, CPU expectation, the three modes, and the persistent database
   location.
2. Explain why exact SQLite scan and RRF are chosen now, plus the threshold at
   which to consider an ANN/vector database.
3. Run the full deterministic quality gate:

   ```bash
   uv run pytest
   uv run ruff check src tests
   uv run mypy src
   uv run pka health
   ```

4. Run the manual smoke test once with the real model, after the user approves
   the model download:

   ```bash
   uv sync --all-groups
   uv run pka ingest tests/fixtures/knowledge/rag_notes.md --db /tmp/pka-e5.db
   uv run pka search "如何用外部知识减少模型幻觉" --mode semantic --db /tmp/pka-e5.db
   uv run pka search "RAG 检索" --mode hybrid --db /tmp/pka-e5.db
   ```

5. Record observed results and limitations in the learning log. Do not commit
   the generated `.db` or downloaded model artifacts.
6. Commit: `docs: explain Phase 1B hybrid retrieval`.

## Final review

Before merging, inspect the complete diff; confirm only Phase 1B code, tests,
dependency lock, and documentation are staged. Re-run the quality gate. Merge
the feature branch into `main` only after all checks and the real smoke test
pass.

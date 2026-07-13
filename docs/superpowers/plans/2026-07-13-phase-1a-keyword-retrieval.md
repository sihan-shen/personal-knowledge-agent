# Phase 1A Keyword Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent Markdown ingestion and deterministic Chinese/English BM25 keyword search to the `pka` CLI.

**Architecture:** Frozen domain models and protocols define the retrieval boundary. Application services orchestrate concrete Markdown and SQLite adapters assembled in `cli.py`; SQLite persists documents, chunks, and an inspectable inverted index while Python computes BM25.

**Tech Stack:** Python 3.12+, standard-library `argparse`, `dataclasses`, `hashlib`, `json`, `re`, `sqlite3`, pytest, Ruff, mypy.

---

## File map

- `src/personal_knowledge_agent/domain/errors.py`: expected user and storage failures.
- `src/personal_knowledge_agent/domain/models.py`: immutable document, chunk, result, and ingestion records plus stable ID helpers.
- `src/personal_knowledge_agent/domain/protocols.py`: loader, chunker, and index contracts.
- `src/personal_knowledge_agent/infrastructure/mixed_tokenizer.py`: NFKC English-word and CJK-bigram tokenization.
- `src/personal_knowledge_agent/infrastructure/markdown_loader.py`: safe UTF-8 Markdown loading.
- `src/personal_knowledge_agent/infrastructure/markdown_chunker.py`: heading-aware paragraph and fenced-code grouping.
- `src/personal_knowledge_agent/infrastructure/sqlite_keyword_index.py`: schema, atomic replacement, persistence, and BM25.
- `src/personal_knowledge_agent/application/ingest.py`: ingestion orchestration.
- `src/personal_knowledge_agent/application/search.py`: search-limit validation and delegation.
- `src/personal_knowledge_agent/cli.py`: health, ingest, and search commands and dependency composition.
- `tests/fixtures/knowledge/*.md`: fixed bilingual knowledge and distractor documents.
- `tests/unit/*.py`: isolated model, tokenizer, loader, and chunker behavior.
- `tests/integration/test_sqlite_keyword_index.py`: real temporary SQLite behavior.
- `tests/test_cli.py`: subprocess-level CLI workflow.
- `.gitignore`, `README.md`, `docs/learning-log/phase-1a.md`: local database safety and developer documentation.

### Task 1: Domain errors and immutable models

**Files:**
- Create: `src/personal_knowledge_agent/domain/errors.py`
- Create: `src/personal_knowledge_agent/domain/models.py`
- Create: `tests/unit/test_models.py`

- [ ] **Step 1: Write failing tests for stable IDs, immutability, and validation**

Tests must assert that identical source/position inputs create identical SHA-256 IDs, different positions create different chunk IDs, frozen records reject mutation, blank identifiers raise `ValueError`, negative positions raise `ValueError`, and blank chunk text raises `ValueError`.

The public model signatures are:

```python
def make_document_id(source: str) -> str: ...
def make_chunk_id(document_id: str, position: int) -> str: ...

@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    source: str
    content: str

@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    position: int
    text: str
    heading_path: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    source: str
    score: float

@dataclass(frozen=True, slots=True)
class IngestionResult:
    source: str
    chunk_count: int
    replaced: bool
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/unit/test_models.py -v`

Expected: collection fails because `domain.models` does not exist.

- [ ] **Step 3: Implement errors, ID helpers, and models**

`errors.py` defines `KnowledgeError` and the seven approved subclasses. `models.py` uses `hashlib.sha256`, frozen slotted dataclasses, and `__post_init__` validation. Hash inputs use UTF-8; chunk hashing uses `f"{document_id}:{position}"`.

- [ ] **Step 4: Run GREEN and static checks**

Run: `uv run pytest tests/unit/test_models.py -v && uv run mypy src`

Expected: all model tests pass and mypy reports no issues.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/domain tests/unit/test_models.py
git commit -m "feat: add retrieval domain models"
```

### Task 2: Domain protocols and mixed-language tokenizer

**Files:**
- Create: `src/personal_knowledge_agent/domain/protocols.py`
- Create: `src/personal_knowledge_agent/infrastructure/mixed_tokenizer.py`
- Create: `tests/unit/test_mixed_tokenizer.py`

- [ ] **Step 1: Write tokenizer behavior tests**

Cover NFKC normalization, lowercase English, apostrophe/hyphen words, version tokens, CJK bigrams, a single CJK character, mixed input, punctuation omission, and whitespace-only input. Exact examples include:

```python
assert tokenizer.tokenize("Retrieval with RAG") == ["retrieval", "with", "rag"]
assert tokenizer.tokenize("检索增强生成") == ["检索", "索增", "增强", "强生", "生成"]
assert tokenizer.tokenize("RAG 检索 Python 3.12") == ["rag", "检索", "python", "3.12"]
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/unit/test_mixed_tokenizer.py -v`

Expected: collection fails because `MixedLanguageTokenizer` does not exist.

- [ ] **Step 3: Implement protocols and tokenizer**

Protocols use `Path`, `Sequence`, and the exact model types:

```python
class DocumentLoader(Protocol):
    def load(self, path: Path) -> Document: ...

class Chunker(Protocol):
    def split(self, document: Document) -> list[Chunk]: ...

class KnowledgeIndex(Protocol):
    def replace_document(self, document: Document, chunks: Sequence[Chunk]) -> bool: ...
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]: ...
```

The tokenizer normalizes using `unicodedata.normalize("NFKC", text).lower()`. One compiled regex extracts Latin words, dotted numeric versions, and contiguous CJK sequences. CJK sequences become overlapping bigrams; one character remains a unigram.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/unit/test_mixed_tokenizer.py -v`

Expected: all tokenizer tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/domain/protocols.py src/personal_knowledge_agent/infrastructure/mixed_tokenizer.py tests/unit/test_mixed_tokenizer.py
git commit -m "feat: add mixed-language tokenization"
```

### Task 3: Safe Markdown loading

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/markdown_loader.py`
- Create: `tests/unit/test_markdown_loader.py`

- [ ] **Step 1: Write loader tests**

Use `tmp_path` to cover `.md`, case-insensitive `.markdown`, absolute resolved source, deterministic document ID, missing path, directory input, unsupported `.txt`, invalid UTF-8 bytes, and whitespace-only Markdown. Expected project errors are `UnsupportedDocumentError`, `DocumentReadError`, and `EmptyDocumentError`.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/unit/test_markdown_loader.py -v`

Expected: collection fails because `MarkdownLoader` does not exist.

- [ ] **Step 3: Implement the loader**

Resolve `path.expanduser().resolve()`, validate the suffix and `is_file()`, read with UTF-8, convert `OSError` and `UnicodeError` into `DocumentReadError`, reject whitespace-only text, and return `Document(make_document_id(str(resolved)), str(resolved), content)`.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/unit/test_markdown_loader.py -v`

Expected: all loader tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/infrastructure/markdown_loader.py tests/unit/test_markdown_loader.py
git commit -m "feat: add Markdown document loader"
```

### Task 4: Heading-aware Markdown chunking

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/markdown_chunker.py`
- Create: `tests/unit/test_markdown_chunker.py`

- [ ] **Step 1: Write chunker tests**

Cover nested ATX headings, heading transitions, paragraph aggregation below the target, splitting before a block that exceeds the target, preserving a fenced Python block containing `#` and blank lines, keeping one oversized block intact, ignoring empty sections, deterministic positions/IDs, and no overlap.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/unit/test_markdown_chunker.py -v`

Expected: collection fails because `MarkdownChunker` does not exist.

- [ ] **Step 3: Implement block scanning and aggregation**

Use private `_Block(heading_path, text)` records. A line scanner tracks the opening fence marker (backticks or tildes), recognizes `#{1,6}` headings only outside fences, flushes paragraphs at blank lines, and treats a fenced region as one block. Aggregate adjacent blocks with equal headings while `len(current + "\n\n" + next) <= target_chars`; default `target_chars=800` and reject non-positive targets.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/unit/test_markdown_chunker.py -v`

Expected: all chunker tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/infrastructure/markdown_chunker.py tests/unit/test_markdown_chunker.py
git commit -m "feat: add heading-aware Markdown chunking"
```

### Task 5: SQLite schema and atomic replacement

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/sqlite_keyword_index.py`
- Create: `tests/integration/test_sqlite_keyword_index.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write schema, persistence, replacement, and rollback tests**

With a temporary database, assert schema `user_version == 1`, persistence after constructing a second index, first replacement returns `False`, second replacement returns `True`, old chunks disappear, repeat ingestion produces no duplicates, foreign keys are enabled, version 2 raises `IndexStorageError`, and duplicate chunk IDs force a failed transaction that preserves the prior document.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/integration/test_sqlite_keyword_index.py -v`

Expected: collection fails because `SQLiteKeywordIndex` does not exist.

- [ ] **Step 3: Implement schema initialization and replacement**

`SQLiteKeywordIndex(db_path: Path, tokenizer: MixedLanguageTokenizer | None = None)` creates parent directories, opens connections with `PRAGMA foreign_keys = ON`, initializes the approved three-table schema at user version 1, JSON-encodes headings, calculates a SHA-256 content hash, tokenizes `" ".join(heading_path) + "\n" + text`, and writes `Counter` frequencies in a single explicit transaction. Use `contextlib.closing` so each connection closes. Convert `OSError`, `sqlite3.Error`, and invalid stored JSON into `IndexStorageError` with exception chaining.

- [ ] **Step 4: Ignore local knowledge databases**

Add `.pka/`, `*.db-wal`, and `*.db-shm` to `.gitignore`.

- [ ] **Step 5: Run GREEN**

Run: `uv run pytest tests/integration/test_sqlite_keyword_index.py -v`

Expected: schema and replacement tests pass.

- [ ] **Step 6: Commit**

```bash
git add .gitignore src/personal_knowledge_agent/infrastructure/sqlite_keyword_index.py tests/integration/test_sqlite_keyword_index.py
git commit -m "feat: persist keyword index in SQLite"
```

### Task 6: BM25 search

**Files:**
- Modify: `src/personal_knowledge_agent/infrastructure/sqlite_keyword_index.py`
- Modify: `tests/integration/test_sqlite_keyword_index.py`

- [ ] **Step 1: Add failing ranking tests**

Test Chinese, English, and mixed queries; higher term relevance; heading-term matching; top-k; deterministic chunk-ID tie breaks; reopen persistence; punctuation-only `InvalidQueryError`; empty-index `KnowledgeBaseEmptyError`; and populated no-match returning `[]`.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/integration/test_sqlite_keyword_index.py -v`

Expected: new search tests fail because search is absent.

- [ ] **Step 3: Implement BM25**

Deduplicate query tokens while preserving order. Read `N`, `avgdl`, matching postings, and joined chunk/document data. Compute `df` from unique postings, `IDF = log(1 + (N - df + 0.5) / (df + 0.5))`, and BM25 with `k1=1.5`, `b=0.75`. Accumulate scores by chunk ID, reconstruct `Chunk` and `SearchResult`, retain positive scores, sort by `(-score, chunk_id)`, and apply `limit`.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/integration/test_sqlite_keyword_index.py -v`

Expected: all SQLite and BM25 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/infrastructure/sqlite_keyword_index.py tests/integration/test_sqlite_keyword_index.py
git commit -m "feat: rank keyword matches with BM25"
```

### Task 7: Application services

**Files:**
- Create: `src/personal_knowledge_agent/application/ingest.py`
- Create: `src/personal_knowledge_agent/application/search.py`
- Create: `tests/unit/test_application_services.py`

- [ ] **Step 1: Write service tests using typed fakes**

Assert ingestion calls loader, chunker, and index in order; returns source/count/replaced; zero chunks raise `EmptyDocumentError`; search delegates query and limit; limits below 1 or above 20 raise `InvalidQueryError`.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/unit/test_application_services.py -v`

Expected: collection fails because the service classes do not exist.

- [ ] **Step 3: Implement services**

Constructors accept domain protocols. `IngestService.ingest(path: Path) -> IngestionResult` orchestrates load/split/replace. `SearchService.search(query: str, *, limit: int = 5) -> list[SearchResult]` validates the limit and delegates.

- [ ] **Step 4: Run GREEN and mypy**

Run: `uv run pytest tests/unit/test_application_services.py -v && uv run mypy src`

Expected: tests and strict type checking pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/application tests/unit/test_application_services.py
git commit -m "feat: orchestrate ingestion and search"
```

### Task 8: CLI ingest and search workflow

**Files:**
- Modify: `src/personal_knowledge_agent/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing subprocess tests**

Preserve health and unknown-command tests. Add temporary-database tests for first ingest, reingest, search from a separate subprocess, score/source/heading output, no match, empty database, invalid path, and `--limit` outside 1..20 returning argparse exit code 2.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_cli.py -v`

Expected: ingest and search tests fail because the parser only supports health.

- [ ] **Step 3: Refactor CLI to subcommands and compose dependencies**

Use required subparsers for `health`, `ingest`, and `search`. Both new commands accept `--db` with default `.pka/knowledge.db`; search accepts an argparse type validator for 1..20. Construct loader, chunker, tokenizer, index, and services at the CLI boundary. Catch only `KnowledgeError`, use `parser.exit(1, f"error: {error}\n")`, print indexed/reindexed counts, and format results with three decimal score, ` > ` heading path, collapsed whitespace, and 160-character snippets.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/test_cli.py -v`

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/cli.py tests/test_cli.py
git commit -m "feat: expose persistent keyword search CLI"
```

### Task 9: Fixed bilingual fixtures and end-to-end relevance

**Files:**
- Create: `tests/fixtures/knowledge/rag_notes.md`
- Create: `tests/fixtures/knowledge/python_agents.md`
- Create: `tests/fixtures/knowledge/unrelated_notes.md`
- Create: `tests/integration/test_retrieval_workflow.py`

- [ ] **Step 1: Create fixed synthetic fixtures**

`rag_notes.md` contains Chinese RAG definitions, English retrieval notes, and a fenced Python example. `python_agents.md` contains tool, state, and decision sections in both languages. `unrelated_notes.md` contains coffee and gardening distractors without RAG or Agent vocabulary.

- [ ] **Step 2: Write end-to-end ranking tests**

Ingest all three fixtures through `IngestService`, reopen the SQLite index, and assert Chinese `检索增强生成`, English `retrieval context`, and mixed `Agent 工具 decision` queries rank the intended source first and retain non-empty headings and positive scores.

- [ ] **Step 3: Run the integration suite**

Run: `uv run pytest tests/integration -v`

Expected: all persistence, ranking, and workflow tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures tests/integration/test_retrieval_workflow.py
git commit -m "test: add bilingual retrieval fixtures"
```

### Task 10: Documentation and final quality gates

**Files:**
- Modify: `README.md`
- Create: `docs/learning-log/phase-1a.md`

- [ ] **Step 1: Document usage and boundaries**

README commands must show ingest, search, `--db`, and the fact that Phase 1A is lexical retrieval without LLM answers. The learning log explains Document/Chunk, heading-aware chunking, inverted indexes, mixed tokenization, BM25, SQLite transactions, and lexical-versus-vector trade-offs.

- [ ] **Step 2: Run formatting and apply only mechanical changes**

Run: `uv run ruff format .`

Expected: Ruff formats Python files without errors.

- [ ] **Step 3: Run every quality gate**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest`

Expected: formatting, lint, and mypy succeed; all tests pass with zero failures.

- [ ] **Step 4: Run a fresh CLI smoke test**

Run:

```bash
tmp_db=/tmp/pka-phase-1a.db
uv run pka ingest tests/fixtures/knowledge/rag_notes.md --db "$tmp_db"
uv run pka search "RAG 如何检索 knowledge" --db "$tmp_db"
```

Expected: ingestion reports a positive chunk count and search returns at least one scored result from `rag_notes.md` with a heading path.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/learning-log/phase-1a.md
git commit -m "docs: explain Phase 1A keyword retrieval"
```

### Task 11: Final repository review

**Files:**
- Verify all files in this plan.

- [ ] **Step 1: Check scope and secrets**

Run: `git status --short && git diff main...HEAD --stat && rg -n "OPENAI_API_KEY=.+" . --glob '!uv.lock' || true`

Expected: worktree is clean, the diff contains only Phase 1A files, and no populated API key is found.

- [ ] **Step 2: Re-run merged-quality commands immediately before integration**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest`

Expected: all commands exit zero with no test failures.

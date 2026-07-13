# Phase 1A: Persistent Keyword Retrieval Design

## Purpose

Build the first independently testable retrieval slice of Personal Knowledge Agent. A user can ingest Markdown files into a persistent local knowledge base and search for relevant chunks in Chinese, English, or mixed-language queries without calling an LLM.

This phase establishes a transparent lexical retrieval baseline. Later vector and hybrid retrievers will be compared against the same behavior and interfaces.

## Scope

Phase 1A includes:

- UTF-8 Markdown loading;
- heading-aware, paragraph-preserving chunking;
- deterministic English-word and Chinese-bigram tokenization;
- a persistent SQLite inverted index;
- BM25 ranking implemented in Python;
- `pka ingest` and `pka search` commands;
- fixed synthetic test documents in Chinese and English;
- unit, integration, and CLI tests.

Phase 1A excludes PDF, web, and code loaders; embeddings; vector databases; LLM generation; prompts; Agent decisions; reranking; chunk overlap; concurrent writers; and a database migration framework.

## Alternatives Considered

### Retrieval approach

1. Offline keyword retrieval was selected because it has no model dependency and makes tokenization, indexing, and ranking observable.
2. Embedding retrieval is closer to a typical production RAG system but introduces model and vector-index variables before a baseline exists.
3. Hybrid retrieval has greater eventual recall but is premature until lexical and vector retrieval can be evaluated independently.

### Persistence approach

1. SQLite with a custom inverted index was selected for transactions, persistence, inspectable storage, and direct exposure to BM25 mechanics.
2. JSON is easier to inspect but requires coarse rewrites and provides weak consistency as the knowledge base grows.
3. SQLite FTS5 provides built-in ranking but makes the central indexing and scoring logic less visible and still requires custom mixed-language tokenization.

No ORM is used. Python's standard-library `sqlite3` module is sufficient for the selected schema and transactions.

## Architecture

```text
CLI
├── pka ingest PATH [--db PATH]
└── pka search QUERY [--limit N] [--db PATH]
          ↓
Application Services
├── IngestService
└── SearchService
          ↓
Domain Interfaces and Models
├── DocumentLoader
├── Chunker
├── KnowledgeIndex
├── Document
├── Chunk
└── SearchResult
          ↓
Infrastructure
├── MarkdownLoader
├── MarkdownChunker
├── MixedLanguageTokenizer
└── SQLiteKeywordIndex
```

Dependencies point inward:

- `domain` does not depend on the application, CLI, Markdown, or SQLite;
- `application` depends on domain models and protocols;
- `infrastructure` implements domain protocols using concrete technology;
- `cli.py` is the composition root that constructs infrastructure and application services.

## Proposed File Structure

```text
src/personal_knowledge_agent/
├── application/
│   ├── ingest.py
│   └── search.py
├── domain/
│   ├── errors.py
│   ├── models.py
│   └── protocols.py
├── infrastructure/
│   ├── markdown_chunker.py
│   ├── markdown_loader.py
│   ├── mixed_tokenizer.py
│   └── sqlite_keyword_index.py
└── cli.py

tests/
├── fixtures/knowledge/
│   ├── python_agents.md
│   ├── rag_notes.md
│   └── unrelated_notes.md
├── integration/
│   └── test_sqlite_keyword_index.py
├── unit/
│   ├── test_markdown_chunker.py
│   ├── test_markdown_loader.py
│   ├── test_models.py
│   └── test_mixed_tokenizer.py
└── test_cli.py
```

## Domain Models

### Document

- `document_id: str`: a deterministic hash derived from the normalized absolute source path;
- `source: str`: the normalized absolute source path;
- `content: str`: decoded Markdown source.

### Chunk

- `chunk_id: str`: a deterministic hash derived from `document_id` and zero-based `position`;
- `document_id: str`;
- `position: int`;
- `text: str`;
- `heading_path: tuple[str, ...]`.

Models are frozen dataclasses. They reject blank identifiers, negative positions, and blank chunk text in `__post_init__`. A document may reach the loader with whitespace-only content only long enough to be rejected as `EmptyDocumentError` before indexing.

### SearchResult

- `chunk: Chunk`;
- `source: str`;
- `score: float`.

Search results are ordered by descending score and then ascending `chunk_id`, producing deterministic ties.

## Protocols

```text
DocumentLoader.load(path) -> Document
Chunker.split(document) -> list[Chunk]
KnowledgeIndex.replace_document(document, chunks) -> bool
KnowledgeIndex.search(query, limit) -> list[SearchResult]
```

`replace_document` returns `True` when a source already existed and was replaced, and `False` for its first ingestion. This supports accurate CLI output without exposing SQLite to the application layer.

`KnowledgeIndex.search` accepts the original query string. The concrete index owns the matching tokenizer so indexing and query tokenization cannot accidentally diverge. A later vector implementation can accept the same query without leaking embeddings through the application interface.

## Markdown Loading

`MarkdownLoader`:

1. normalizes and resolves the input path;
2. rejects extensions other than `.md` and `.markdown` case-insensitively;
3. reads the file as UTF-8;
4. rejects missing, non-file, unreadable, undecodable, or whitespace-only input with project-specific errors;
5. derives a stable `document_id` from the normalized absolute path.

The loader does not parse Markdown structure. Structural parsing belongs to `MarkdownChunker`.

## Markdown Chunking

The chunker uses an 800-character soft target:

1. scan Markdown line by line while tracking fenced code blocks;
2. recognize ATX headings (`#` through `######`) only outside code fences;
3. maintain the current heading path by level;
4. group non-heading content into blocks separated by blank lines;
5. treat a complete fenced code block as one indivisible block;
6. aggregate adjacent blocks with the same heading path until adding the next block would exceed 800 characters;
7. emit the current chunk and begin another when the soft target would be exceeded.

An individual paragraph or fenced code block may exceed 800 characters and remains intact. Headings are stored in `heading_path` rather than duplicated in `text`. Heading terms are included once when the chunk is tokenized for indexing. Empty sections do not emit chunks. Phase 1A uses no overlap.

Chunk positions are assigned in document order starting at zero. Given the same source path and content, chunk IDs, positions, text, and heading paths are deterministic.

## Mixed-Language Tokenization

Tokenization is deterministic and has no external dependency:

1. normalize text with Unicode NFKC;
2. lowercase Latin text;
3. extract Latin words, version-like numeric tokens, and contiguous CJK sequences;
4. retain Latin and numeric tokens as extracted;
5. convert each CJK sequence of length two or more into overlapping character bigrams;
6. preserve a single CJK character as a unigram;
7. omit punctuation and whitespace.

Examples:

```text
"Retrieval with RAG" -> ["retrieval", "with", "rag"]
"检索增强生成" -> ["检索", "索增", "增强", "强生", "生成"]
"RAG 检索 Python 3.12" -> ["rag", "检索", "python", "3.12"]
```

No stop-word filtering or stemming is used in this baseline. During indexing, the effective searchable text is the heading path joined with spaces followed by the chunk text.

## SQLite Schema

The default database is `.pka/knowledge.db`, relative to the current working directory. Both `.pka/` and SQLite sidecar files remain untracked through `.gitignore`. `--db PATH` overrides the default for testing and explicit knowledge bases.

```text
documents
├── document_id TEXT PRIMARY KEY
├── source TEXT UNIQUE NOT NULL
└── content_hash TEXT NOT NULL

chunks
├── chunk_id TEXT PRIMARY KEY
├── document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE
├── position INTEGER NOT NULL
├── text TEXT NOT NULL
├── heading_path TEXT NOT NULL
└── token_count INTEGER NOT NULL

postings
├── token TEXT NOT NULL
├── chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE
└── frequency INTEGER NOT NULL

PRIMARY KEY (token, chunk_id)
INDEX postings(token)
UNIQUE (document_id, position)
```

`heading_path` is stored as a JSON array. On connection, the index enables `PRAGMA foreign_keys = ON`.

Database lifecycle:

- parent directories are created before connecting;
- `PRAGMA user_version = 0` creates the schema and sets version 1;
- version 1 opens normally;
- any other version raises `IndexStorageError` rather than silently reading an incompatible schema;
- connections are closed by context managers.

## Atomic Document Replacement

`replace_document` performs one transaction:

1. determine whether the source already exists;
2. delete the prior document row when present, cascading to chunks and postings;
3. insert the current document metadata;
4. tokenize every chunk, count term frequencies, and insert chunks and postings;
5. commit only after all writes succeed.

Any constraint or storage error rolls back the whole transaction, preserving the previously indexed version. Reingesting an unchanged file still performs deterministic replacement in Phase 1A; content-hash short-circuiting is deferred until performance requires it.

## BM25 Ranking

The index gathers postings for unique query tokens and computes BM25 in Python:

```text
IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

score(q, d) = Σ IDF(t) ×
  tf(t,d) × (k1 + 1)
  ─────────────────────────────────────
  tf(t,d) + k1 × (1 - b + b × |d| / avgdl)
```

Constants are `k1 = 1.5` and `b = 0.75`. `|d|` and `avgdl` use indexed token counts, including heading terms. Repeated query tokens are treated once in Phase 1A. Results with a score greater than zero are returned, capped by the requested limit.

## Application Services

`IngestService` loads one path, splits its document, rejects a zero-chunk result, and atomically replaces the source in the index. It returns an ingestion result containing source, chunk count, and whether the source was replaced.

`SearchService` validates `1 <= limit <= 20`, delegates search to the index, and returns ordered results. It does not format terminal output.

## CLI Behavior

Commands:

```bash
pka health
pka ingest PATH [--db PATH]
pka search QUERY [--limit 5] [--db PATH]
```

Successful first ingestion:

```text
Indexed 4 chunks from /absolute/path/notes/rag.md
```

Successful replacement:

```text
Reindexed 5 chunks from /absolute/path/notes/rag.md
```

Search result:

```text
1. [score=2.184] /absolute/path/notes/rag.md > RAG > Retrieval
   Retrieval selects relevant evidence before generation.
```

Search snippets replace internal whitespace with single spaces and are truncated to 160 characters for terminal display. Stored chunk text remains unchanged.

No matches print `No relevant chunks found.` and exit zero. An empty knowledge base prints `error: Knowledge base is empty. Ingest a document first.` to stderr and exits one.

Exit codes:

- 0: successful operation, including no matches;
- 1: document, knowledge-state, or storage failure;
- 2: invalid CLI syntax or arguments handled by `argparse`.

## Error Handling

`domain/errors.py` defines:

- `KnowledgeError`;
- `UnsupportedDocumentError`;
- `DocumentReadError`;
- `EmptyDocumentError`;
- `InvalidQueryError`;
- `KnowledgeBaseEmptyError`;
- `IndexStorageError`.

Expected decoding, file-system, and SQLite failures are converted to these types with safe messages. The CLI catches `KnowledgeError`, writes `error: <message>` to stderr, and exits one. Unexpected programming errors are not converted, ensuring their tracebacks remain visible during development.

An empty or punctuation-only query raises `InvalidQueryError`. A database with no chunks raises `KnowledgeBaseEmptyError`. A populated database with no matching postings returns an empty list.

## Testing Strategy

Fixed synthetic fixtures are committed once rather than generated at test runtime:

- `rag_notes.md`: Chinese RAG explanation, English retrieval notes, and a fenced Python example;
- `python_agents.md`: Agent tools, state, and decisions;
- `unrelated_notes.md`: coffee brewing and gardening distractors.

Unit coverage:

- models reject invalid values and produce stable IDs;
- tokenizer covers normalization, English, numbers, Chinese bigrams, mixed text, punctuation, and empty input;
- loader covers successful Markdown, missing input, non-files, unsupported extensions, invalid UTF-8, and empty documents;
- chunker covers nested headings, paragraph aggregation, fenced-code preservation, an oversized indivisible block, empty sections, and deterministic output.

SQLite integration coverage:

- schema initialization and version rejection;
- persistence after closing and reopening;
- BM25 ordering and deterministic ties;
- top-k limits;
- first ingestion versus replacement;
- no duplicate chunks after repeated ingestion;
- rollback preserves the previous document after an insertion failure;
- empty knowledge base, invalid query, and no-match behavior.

Application and CLI coverage:

- ingestion and search orchestration use protocol-compatible fakes where isolation is valuable;
- existing `health` behavior remains intact;
- `ingest` writes a temporary database;
- a separate CLI subprocess searches that same database;
- Chinese, English, and mixed queries rank relevant fixtures above distractors;
- invalid paths, limits, empty databases, and unknown commands return the specified output and exit codes.

## Acceptance Criteria

Phase 1A is complete when:

```bash
uv run pka ingest tests/fixtures/knowledge/rag_notes.md --db /tmp/pka-test.db
uv run pka search "RAG 如何检索 knowledge" --db /tmp/pka-test.db
```

returns ranked chunks with scores, source paths, and heading paths; a second process can read the persisted index; reingestion replaces rather than duplicates data; all quality gates pass; and no LLM, embedding model, vector database, ORM, or external tokenizer dependency has been added.

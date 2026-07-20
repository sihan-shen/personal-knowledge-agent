# Phase 4A Provenance-First Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an explicit, chunk-by-chunk knowledge-graph pipeline for indexed Markdown documents, with strict structured extraction, atomic provenance-aware SQLite replacement, local entity search, and one-hop inspection.

**Architecture:** `GraphBuildService` reads one indexed source through a domain protocol, invokes a provider-neutral graph extractor once per ordered chunk, and submits the complete validated result to one atomic graph-store replacement. A dedicated SQLite graph database shares exact entity and relation identities across documents while retaining document-scoped mentions and evidence; separate local query services expose BM25 entity search and one-hop neighborhoods without constructing LLM or embedding dependencies.

**Tech Stack:** Python 3.12+, stdlib dataclasses/enums/hashlib/json/sqlite3, existing OpenAI-compatible Chat Completions `LanguageModel`, existing mixed Chinese/English tokenizer, pytest, Ruff, strict mypy, uv.

## Global Constraints

- Implement only the approved spec at `docs/superpowers/specs/2026-07-19-phase-4a-provenance-first-knowledge-graph-design.md`.
- Keep `domain/` independent of SQLite, model SDKs, and other infrastructure libraries.
- Use a dedicated graph database defaulting to `.pka/graph.db`; never commit generated `.pka/` data.
- `pka graph build` is the only graph command allowed to construct an LLM; it makes one logical request per current indexed chunk.
- `pka graph search` and `pka graph show` make zero remote calls and do not load E5.
- Preserve exact existing behavior for `ingest`, `search`, `ask`, `agent`, and `memory`.
- Use controlled entity types `concept`, `technology`, `method`, `project`, and `resource`.
- Use controlled relation types `is_a`, `part_of`, `uses`, `depends_on`, `implements`, and `related_to`.
- Enforce at most 15 entities and 20 relations per chunk, 120 code points per entity name, and 300 code points per evidence quote.
- Entity identity is exact `(entity_type, NFKC/casefold/whitespace-normalized name)`; do not add aliases, fuzzy matching, or embedding-based merging.
- Every relation must have an exact quote substring from its source chunk and complete local provenance.
- Use complete type annotations, strict mypy, deterministic fakes, TDD, and narrow commits.
- On the Windows-mounted repository, run uv commands with `TMPDIR=/tmp` and `UV_LINK_MODE=copy`; execution may use an isolated `UV_PROJECT_ENVIRONMENT` under `/tmp`.
- Preserve untracked user-owned `papers/`, `references/`, and `research/` directories.

## Planned File Structure

- Create `src/personal_knowledge_agent/domain/graph.py`: controlled ontology, identity helpers, immutable extraction/build/query values, and graph limits.
- Modify `src/personal_knowledge_agent/domain/errors.py`: graph-specific user-facing error hierarchy.
- Modify `src/personal_knowledge_agent/domain/protocols.py`: source-reader, extractor, and graph-store boundaries.
- Create `src/personal_knowledge_agent/infrastructure/structured_graph_parser.py`: strict JSON/reference/evidence validation.
- Create `src/personal_knowledge_agent/infrastructure/graph_prompt_builder.py`: untrusted per-chunk extraction prompt.
- Create `src/personal_knowledge_agent/infrastructure/llm_knowledge_graph_extractor.py`: provider-neutral prompt/generate/parse adapter.
- Create `src/personal_knowledge_agent/infrastructure/sqlite_graph_source_reader.py`: read one current indexed source without E5 or reparsing Markdown.
- Create `src/personal_knowledge_agent/application/graph_build.py`: all-or-nothing per-document extraction orchestration.
- Create `src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py`: graph schema, source replacement, BM25 entity search, and one-hop reads.
- Create `src/personal_knowledge_agent/application/graph_query.py`: local query validation and delegation.
- Modify `src/personal_knowledge_agent/cli.py`: graph command parser, lazy composition, and stable output.
- Create focused unit/integration tests matching each file, then extend `tests/test_cli.py` for end-to-end and compatibility behavior.
- Modify `README.md`, create `docs/learning-log/phase-4a.md`, and modify `AGENTS.md` only after implementation passes.

---

### Task 1: Define graph domain contracts and deterministic identities

**Files:**
- Create: `src/personal_knowledge_agent/domain/graph.py`
- Modify: `src/personal_knowledge_agent/domain/errors.py`
- Modify: `src/personal_knowledge_agent/domain/protocols.py`
- Create: `tests/unit/test_graph_models.py`

**Interfaces:**
- Consumes: existing `Chunk`, `ModelOutput`, and `LanguageModel` conventions.
- Produces: `EntityType`, `RelationType`, `GraphEntity`, `GraphRelation`, `GraphRelationEvidence`, `GraphChunkExtraction`, `GraphSourceDocument`, `GraphBuildResult`, `GraphProvenance`, `GraphNeighbor`, `EntityNeighborhood`, `EntitySearchResult`, `normalize_entity_name`, `make_entity_id`, `make_relation_id`, `GraphSourceReader`, `KnowledgeGraphExtractor`, and `KnowledgeGraphStore`.

- [ ] **Step 1: Write failing domain behavior tests**

Create table-driven tests that require:

```python
assert normalize_entity_name("  ＲＡＧ  Model ") == "rag model"
assert make_entity_id(EntityType.METHOD, "RAG") == make_entity_id(
    EntityType.METHOD, "rag"
)
assert make_entity_id(EntityType.METHOD, "RAG") != make_entity_id(
    EntityType.CONCEPT, "RAG"
)
```

Also prove these exact invariants:

- entity IDs match `ent_[0-9a-f]{32}` and relation IDs match
  `rel_[0-9a-f]{32}`;
- entity display names are whitespace/NFKC normalized but preserve case;
- `related_to` reversals produce the same canonical endpoints and relation ID;
- directed `uses` reversals produce different IDs;
- all relation self-loops fail;
- `GraphSourceDocument` rejects blank metadata, a non-64-hex content hash,
  duplicate/nonascending chunk positions, or chunks from another document;
- evidence, search score, build count, neighborhood ordering container, length, and
  frozen-dataclass invariants are enforced;
- all five entity and six relation enum values are exact.

- [ ] **Step 2: Run the domain tests and verify RED**

Run:

```bash
uv run pytest tests/unit/test_graph_models.py -q
```

Expected: collection fails because `domain.graph` does not exist.

- [ ] **Step 3: Implement the domain values and errors**

Define these exact limits and identity shapes in `domain/graph.py`:

```python
MAX_GRAPH_ENTITIES_PER_CHUNK = 15
MAX_GRAPH_RELATIONS_PER_CHUNK = 20
MAX_GRAPH_ENTITY_NAME_LENGTH = 120
MAX_GRAPH_EVIDENCE_QUOTE_LENGTH = 300
MAX_GRAPH_SEARCH_LIMIT = 20

class EntityType(StrEnum):
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    METHOD = "method"
    PROJECT = "project"
    RESOURCE = "resource"

class RelationType(StrEnum):
    IS_A = "is_a"
    PART_OF = "part_of"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    RELATED_TO = "related_to"
```

Use NFKC, whitespace collapse, and `casefold()` for identity. Use the first 32 hex
characters of SHA-256 over unambiguous null-separated identity fields, prefixed by
`ent_` or `rel_`. Canonicalize only `related_to` by sorting endpoint IDs before
hashing and constructing `GraphRelation`.

Add to `domain/errors.py`:

```python
class GraphFeatureError(KnowledgeError): ...
class InvalidGraphError(GraphFeatureError): ...
class GraphSourceNotFoundError(GraphFeatureError): ...
class GraphEmptyError(GraphFeatureError): ...
class GraphEntityNotFoundError(GraphFeatureError): ...
class GraphStorageError(GraphFeatureError): ...
class InvalidGraphSchemaError(GraphFeatureError): ...
```

Define frozen, slotted dataclasses with the names in **Interfaces**. Use
`GraphRelationEvidence(relation, evidence_quote)`,
`GraphChunkExtraction(chunk, entities, relation_evidence)`, and
`GraphSourceDocument(document_id, source, content_hash, chunks)`. Query provenance
must contain source, chunk ID, chunk position, heading path, and exact quote.

- [ ] **Step 4: Add protocol signatures and run static checks**

In `domain/protocols.py`, import graph values and add:

```python
class GraphSourceReader(Protocol):
    def load(self, source: Path) -> GraphSourceDocument: ...

class KnowledgeGraphExtractor(Protocol):
    def extract(self, chunk: Chunk) -> GraphChunkExtraction: ...

class KnowledgeGraphStore(Protocol):
    def replace_document(
        self,
        source_document: GraphSourceDocument,
        extractions: Sequence[GraphChunkExtraction],
    ) -> GraphBuildResult: ...

    def search_entities(
        self, query: str, *, limit: int = 10
    ) -> list[EntitySearchResult]: ...

    def get_neighborhood(self, entity_id: str) -> EntityNeighborhood: ...
```

Run:

```bash
uv run pytest tests/unit/test_graph_models.py -q
uv run ruff format --check src/personal_knowledge_agent/domain tests/unit/test_graph_models.py
uv run ruff check src/personal_knowledge_agent/domain tests/unit/test_graph_models.py
uv run mypy src
```

Expected: all pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/personal_knowledge_agent/domain/graph.py src/personal_knowledge_agent/domain/errors.py src/personal_knowledge_agent/domain/protocols.py tests/unit/test_graph_models.py
git commit -m "feat: define knowledge graph contracts"
```

---

### Task 2: Parse strict per-chunk graph extraction output

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/structured_graph_parser.py`
- Create: `tests/unit/test_structured_graph_parser.py`

**Interfaces:**
- Consumes: `ModelOutput`, `Chunk`, entity/relation enums and identity helpers from Task 1.
- Produces: `parse_graph_extraction(output: ModelOutput, chunk: Chunk) -> GraphChunkExtraction`.

- [ ] **Step 1: Write failing happy-path and empty-result tests**

Use a chunk containing `RAG uses vector retrieval.` and require the exact JSON:

```json
{
  "entities": [
    {"local_id": "E1", "name": "RAG", "type": "method"},
    {"local_id": "E2", "name": "Vector Retrieval", "type": "technology"}
  ],
  "relations": [
    {
      "source_id": "E1",
      "type": "uses",
      "target_id": "E2",
      "evidence_quote": "RAG uses vector retrieval."
    }
  ]
}
```

Assert the parser creates two deterministic global entities, one directed relation,
one exact quote, and preserves the original `Chunk`. Also require
`{"entities":[],"relations":[]}` to return an empty extraction.

- [ ] **Step 2: Add failing adversarial schema tests and verify RED**

Cover every strict boundary with concrete payloads:

- non-JSON, fenced JSON, array root, missing/extra/duplicate fields;
- non-list collections and non-object entries;
- nonsequential, duplicate, malformed, or nonstring local IDs;
- blank/over-120 names, duplicate normalized identities, unknown entity type;
- 16 entities and 21 relations;
- unknown endpoints, same normalized endpoint, unknown relation type;
- blank/over-300 quote, paraphrased quote not in `chunk.text`;
- duplicate `(relation identity, evidence_quote)` entries;
- hostile chunk text containing `SYSTEM`, fake JSON delimiters, and `E99` does not
  alter endpoint validation.

Run:

```bash
uv run pytest tests/unit/test_structured_graph_parser.py -q
```

Expected: collection fails because the parser module does not exist.

- [ ] **Step 3: Implement exact decoding, local reference mapping, and evidence validation**

Implement duplicate-key rejection using `json.loads(..., object_pairs_hook=...)`.
Require root fields `entities` and `relations`; entity fields `local_id`, `name`,
`type`; relation fields `source_id`, `type`, `target_id`, `evidence_quote`.

Build entities first in list order, requiring local IDs to equal `E1...En`. Map each
local ID to one `GraphEntity`, rejecting repeated normalized `(type, name)` identity.
Build relations second, resolve both endpoints from that map, call the Task 1
relation constructor for direction/canonicalization, require the quote to satisfy
`quote in chunk.text`, and reject exact duplicate relation-plus-quote pairs. Convert
all domain `ValueError` failures into `InvalidGraphSchemaError` without including the
provider payload.

- [ ] **Step 4: Run focused and compatibility parser gates**

```bash
uv run pytest tests/unit/test_structured_graph_parser.py tests/unit/test_structured_answer_parser.py tests/unit/test_structured_contextual_answer_parser.py tests/unit/test_structured_memory_parser.py tests/unit/test_structured_route_parser.py -q
uv run ruff format --check src/personal_knowledge_agent/infrastructure/structured_graph_parser.py tests/unit/test_structured_graph_parser.py
uv run ruff check src/personal_knowledge_agent/infrastructure/structured_graph_parser.py tests/unit/test_structured_graph_parser.py
uv run mypy src
```

Expected: all pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/personal_knowledge_agent/infrastructure/structured_graph_parser.py tests/unit/test_structured_graph_parser.py
git commit -m "feat: validate structured graph extraction"
```

---

### Task 3: Build the graph extraction prompt and LLM adapter

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/graph_prompt_builder.py`
- Create: `src/personal_knowledge_agent/infrastructure/llm_knowledge_graph_extractor.py`
- Create: `tests/unit/test_graph_prompt_builder.py`
- Create: `tests/unit/test_llm_knowledge_graph_extractor.py`

**Interfaces:**
- Consumes: `Chunk`, `ChatMessage`, `LanguageModel`, and `parse_graph_extraction`.
- Produces: `build_graph_extraction_messages(chunk: Chunk) -> tuple[ChatMessage, ChatMessage]` and `LLMKnowledgeGraphExtractor.extract(chunk) -> GraphChunkExtraction`.

- [ ] **Step 1: Write failing prompt privacy and policy tests**

Require one system and one user message. The user message must serialize exactly:

```text
HEADING_PATH_JSON:
["Architecture", "Retrieval"]

CHUNK_TEXT:
RAG uses vector retrieval.
```

Assert the combined outgoing messages contain all enum values, maximum 15/20,
exact-substring evidence instructions, local `E1...En` rules, self-loop rejection,
and untrusted-data language. Assert they omit the chunk ID, document ID, source path,
graph path, API-key sentinel, memory sentinel, score, and vector sentinel.

- [ ] **Step 2: Write failing adapter call and error-preservation tests**

Use `FakeLanguageModel` to require exactly one immutable message call and the exact
parsed extraction. Require `LanguageModelError` and `InvalidGraphSchemaError` to
propagate without a second model call.

Run:

```bash
uv run pytest tests/unit/test_graph_prompt_builder.py tests/unit/test_llm_knowledge_graph_extractor.py -q
```

Expected: collection fails because both modules are absent.

- [ ] **Step 3: Implement the prompt builder and thin adapter**

The prompt builder must construct no provider/model objects. Use
`json.dumps(chunk.heading_path, ensure_ascii=False)` for headings and include raw
chunk text only after the fixed `CHUNK_TEXT` marker. The system prompt distinguishes
data from instructions and lists the complete schema rather than linking to local
documentation.

The adapter is deliberately thin:

```python
class LLMKnowledgeGraphExtractor:
    def __init__(self, language_model: LanguageModel) -> None:
        self._language_model = language_model

    def extract(self, chunk: Chunk) -> GraphChunkExtraction:
        messages = build_graph_extraction_messages(chunk)
        output = self._language_model.generate(messages)
        return parse_graph_extraction(output, chunk)
```

- [ ] **Step 4: Run extraction boundary checks**

```bash
uv run pytest tests/unit/test_graph_prompt_builder.py tests/unit/test_llm_knowledge_graph_extractor.py tests/unit/test_openai_compatible_chat_model.py -q
uv run ruff format --check src/personal_knowledge_agent/infrastructure tests/unit/test_graph_prompt_builder.py tests/unit/test_llm_knowledge_graph_extractor.py
uv run ruff check src/personal_knowledge_agent/infrastructure tests/unit/test_graph_prompt_builder.py tests/unit/test_llm_knowledge_graph_extractor.py
uv run mypy src
```

Expected: all pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/personal_knowledge_agent/infrastructure/graph_prompt_builder.py src/personal_knowledge_agent/infrastructure/llm_knowledge_graph_extractor.py tests/unit/test_graph_prompt_builder.py tests/unit/test_llm_knowledge_graph_extractor.py
git commit -m "feat: extract graph facts from one chunk"
```

---

### Task 4: Read one graph source from the existing knowledge index

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/sqlite_graph_source_reader.py`
- Create: `tests/integration/test_sqlite_graph_source_reader.py`

**Interfaces:**
- Consumes: the Phase 1 SQLite `documents` and `chunks` schema.
- Produces: `SQLiteGraphSourceReader(database_path: Path).load(source: Path) -> GraphSourceDocument` without an embedder or Markdown parser.

- [ ] **Step 1: Write failing read-only integration tests**

Seed a real temporary knowledge DB through `SQLiteKeywordIndex` and `FakeEmbedder`,
then require the reader to:

- resolve relative/absolute forms of the same source;
- return document ID, persisted 64-hex content hash, and chunks ordered by position;
- preserve chunk text and JSON-decoded heading paths;
- avoid calling the fake embedder during reads;
- raise `GraphSourceNotFoundError` for a valid DB without that source;
- raise `GraphSourceNotFoundError` without creating a missing knowledge DB;
- map missing tables, corrupt heading JSON/types, duplicate positions, bad hashes,
  and SQLite failures to `GraphStorageError` without raw SQL/path details.

- [ ] **Step 2: Run the reader integration tests and verify RED**

```bash
uv run pytest tests/integration/test_sqlite_graph_source_reader.py -q
```

Expected: collection fails because `SQLiteGraphSourceReader` does not exist.

- [ ] **Step 3: Implement a read-only SQLite adapter**

Resolve both database and source paths. Check `database_path.is_file()` before
opening so SQLite cannot create a missing knowledge DB. Execute one document lookup
by exact resolved source and one ordered chunk query inside a read transaction.
Decode heading JSON as a list of strings and construct the Task 1 value; do not
instantiate `SQLiteKeywordIndex`, `E5Embedder`, `MarkdownLoader`, or
`MarkdownChunker`.

Map `sqlite3.Error`, invalid stored scalar types, invalid JSON, and domain invariant
failures to `GraphStorageError("Could not read graph source from knowledge index")`.

- [ ] **Step 4: Run source-reader and existing index gates**

```bash
uv run pytest tests/integration/test_sqlite_graph_source_reader.py tests/integration/test_sqlite_keyword_index.py tests/integration/test_retrieval_workflow.py -q
uv run ruff format --check src/personal_knowledge_agent/infrastructure/sqlite_graph_source_reader.py tests/integration/test_sqlite_graph_source_reader.py
uv run ruff check src/personal_knowledge_agent/infrastructure/sqlite_graph_source_reader.py tests/integration/test_sqlite_graph_source_reader.py
uv run mypy src
```

Expected: all pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/personal_knowledge_agent/infrastructure/sqlite_graph_source_reader.py tests/integration/test_sqlite_graph_source_reader.py
git commit -m "feat: read indexed graph sources"
```

---

### Task 5: Orchestrate all-or-nothing document graph builds

**Files:**
- Create: `src/personal_knowledge_agent/application/graph_build.py`
- Create: `tests/unit/test_graph_build_service.py`

**Interfaces:**
- Consumes: `GraphSourceReader`, `KnowledgeGraphExtractor`, and `KnowledgeGraphStore`.
- Produces: `GraphBuildService.build(source: Path) -> GraphBuildResult`.

- [ ] **Step 1: Create recording fakes and failing orchestration tests**

Require:

- reader receives the resolved source exactly once;
- chunks are extracted once each in ascending position order;
- the store receives the original `GraphSourceDocument` and the complete immutable
  extraction tuple exactly once;
- the store's `GraphBuildResult` is returned unchanged;
- an empty chunk tuple still calls replacement once with an empty tuple;
- reader failure calls neither extractor nor store;
- failure on chunk two stops chunk three and never calls store;
- store errors propagate without retrying extraction.

- [ ] **Step 2: Run the application tests and verify RED**

```bash
uv run pytest tests/unit/test_graph_build_service.py -q
```

Expected: collection fails because `GraphBuildService` is absent.

- [ ] **Step 3: Implement minimal orchestration**

Use constructor injection only:

```python
class GraphBuildService:
    def __init__(
        self,
        source_reader: GraphSourceReader,
        extractor: KnowledgeGraphExtractor,
        store: KnowledgeGraphStore,
    ) -> None: ...

    def build(self, source: Path) -> GraphBuildResult:
        resolved = source.expanduser().resolve()
        source_document = self._source_reader.load(resolved)
        extractions = tuple(
            self._extractor.extract(chunk) for chunk in source_document.chunks
        )
        return self._store.replace_document(source_document, extractions)
```

The CLI parser owns the required-source check; an arbitrary supplied path is
resolved and delegated so the reader can distinguish indexed from unindexed source.
Before replacement, assert each returned extraction preserves the exact input chunk;
map a mismatched extractor result to `InvalidGraphError` so a buggy adapter cannot
attach facts to another provenance record.

- [ ] **Step 4: Run graph application tests and static checks**

```bash
uv run pytest tests/unit/test_graph_build_service.py tests/unit/test_application_services.py -q
uv run ruff format --check src/personal_knowledge_agent/application/graph_build.py tests/unit/test_graph_build_service.py
uv run ruff check src/personal_knowledge_agent/application/graph_build.py tests/unit/test_graph_build_service.py
uv run mypy src
```

Expected: all pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/personal_knowledge_agent/application/graph_build.py tests/unit/test_graph_build_service.py
git commit -m "feat: orchestrate atomic graph builds"
```

---

### Task 6: Persist and atomically replace provenance-aware graph contributions

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py`
- Create: `tests/integration/test_sqlite_knowledge_graph_store.py`

**Interfaces:**
- Consumes: `GraphSourceDocument` and complete `GraphChunkExtraction` tuples.
- Produces: `SQLiteKnowledgeGraphStore.replace_document(...) -> GraphBuildResult` plus a versioned graph schema used by Tasks 7–8.

- [ ] **Step 1: Write failing schema and first-build integration tests**

Require a new DB to create schema version 1 with foreign keys and these logical
tables/indexes:

```text
schema_metadata
graph_documents
entities
entity_mentions
relations
relation_evidence
entity_postings
```

Seed two chunks whose extractions share an entity and relation. Assert one source
contribution stores unique entity/relation rows, both chunk mentions/evidence where
appropriate, JSON heading snapshots, stable IDs, exact quotes, and document-local
counts. Close and reopen the store, then assert durable rows are unchanged.

- [ ] **Step 2: Add failing lifecycle and rollback tests**

Cover:

- two documents reuse exact entity/relation identities while retaining separate
  mentions/evidence;
- replacing one source removes only its contribution and preserves the other;
- replacing with an empty extraction removes true orphan relations/entities;
- entities with another mention and relations with another evidence row survive;
- display name recomputes to the lexicographically smallest active mention,
  independent of build order;
- exact duplicate mention/evidence input is rejected before writes;
- extraction chunks not belonging to the source document are rejected;
- injected SQLite failure rolls back every table;
- corrupt schema version or stored enum/ID/heading/count maps to `GraphStorageError`.

- [ ] **Step 3: Run lifecycle tests and verify RED**

```bash
uv run pytest tests/integration/test_sqlite_knowledge_graph_store.py -q
```

Expected: collection fails because the store module does not exist.

- [ ] **Step 4: Implement schema initialization and atomic replacement**

The constructor resolves the graph path, creates only its parent directory, opens
SQLite with row factory, enables foreign keys, and creates or validates schema
version 1. Accept an injected UTC clock for deterministic `built_at` tests.

Prepare and validate every row before `BEGIN IMMEDIATE`. Within one transaction:

1. delete the old `graph_documents` row by source so contribution foreign keys
   cascade;
2. insert the current document ID, source, content hash, and UTC timestamp;
3. upsert entity identities and insert per-chunk mentions;
4. upsert relation identities and insert exact evidence rows;
5. delete relations with no evidence;
6. delete entities with no mentions and no incident relations;
7. recompute each affected entity's display name from `MIN(mention_name)`;
8. commit.

Use explicit SQL column lists and translate `sqlite3.Error`, stored-data conversion
failures, and invariant failures to `GraphStorageError` with fixed messages. Never
include evidence or provider content in error text.

- [ ] **Step 5: Run graph-store lifecycle and project static gates**

```bash
uv run pytest tests/integration/test_sqlite_knowledge_graph_store.py -q
uv run ruff format --check src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py tests/integration/test_sqlite_knowledge_graph_store.py
uv run ruff check src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py tests/integration/test_sqlite_knowledge_graph_store.py
uv run mypy src
```

Expected: all pass.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py tests/integration/test_sqlite_knowledge_graph_store.py
git commit -m "feat: persist provenance-aware graph facts"
```

---

### Task 7: Add local BM25 entity search and query-service validation

**Files:**
- Modify: `src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py`
- Create: `src/personal_knowledge_agent/application/graph_query.py`
- Modify: `tests/integration/test_sqlite_knowledge_graph_store.py`
- Create: `tests/unit/test_graph_query_service.py`

**Interfaces:**
- Consumes: persisted entities and `KnowledgeGraphStore.search_entities`.
- Produces: `GraphQueryService.search(query: str, *, limit: int = 10) -> list[EntitySearchResult]` and working SQLite BM25 search.

- [ ] **Step 1: Write failing application validation tests**

Require blank/whitespace query and limits 0/21 to raise `InvalidGraphError` before
the fake store. Require limits 1/20 and normal queries to delegate exactly once and
return the exact result list. Do not catch `GraphEmptyError` or `GraphStorageError`.

- [ ] **Step 2: Write failing SQLite search/ranking tests**

After replacement, require:

- mixed Chinese/English token matching over entity display/normalized names;
- BM25 rewards repeated/exact terms and excludes zero lexical matches;
- limit and stable entity-ID tie-break behavior;
- rebuilding a source refreshes postings and token counts;
- an empty graph raises `GraphEmptyError` without creating model/E5 dependencies;
- punctuation-only queries raise `InvalidGraphError`;
- corrupt postings/counts map to `GraphStorageError`.

Run:

```bash
uv run pytest tests/unit/test_graph_query_service.py tests/integration/test_sqlite_knowledge_graph_store.py -q
```

Expected: failures because the service and search implementation are absent.

- [ ] **Step 3: Implement postings and BM25 search**

Reuse `MixedLanguageTokenizer` and the existing BM25 constants `k1=1.5`, `b=0.75`.
During every replacement, rebuild postings for affected surviving entities from
`f"{display_name} {normalized_name}"`; store positive token counts and frequencies.

Search active entity rows only, calculate document frequency and BM25 in Python or
bounded SQL using the graph tables, exclude scores `<= 0`, and return at most the
requested limit ordered by `(-score, entity_id)`. A graph with zero entities raises
`GraphEmptyError("Knowledge graph is empty. Build a document graph first.")`.

Implement `GraphQueryService` as a small validator/delegator with no tokenizer or
SQL knowledge.

- [ ] **Step 4: Run local-query gates**

```bash
uv run pytest tests/unit/test_graph_query_service.py tests/integration/test_sqlite_knowledge_graph_store.py tests/unit/test_mixed_tokenizer.py -q
uv run ruff format --check src/personal_knowledge_agent/application/graph_query.py src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py tests/unit/test_graph_query_service.py tests/integration/test_sqlite_knowledge_graph_store.py
uv run ruff check src/personal_knowledge_agent/application/graph_query.py src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py tests/unit/test_graph_query_service.py tests/integration/test_sqlite_knowledge_graph_store.py
uv run mypy src
```

Expected: all pass.

- [ ] **Step 5: Commit Task 7**

```bash
git add src/personal_knowledge_agent/application/graph_query.py src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py tests/unit/test_graph_query_service.py tests/integration/test_sqlite_knowledge_graph_store.py
git commit -m "feat: search graph entities locally"
```

---

### Task 8: Return deterministic one-hop neighborhoods with complete evidence

**Files:**
- Modify: `src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py`
- Modify: `src/personal_knowledge_agent/application/graph_query.py`
- Modify: `tests/integration/test_sqlite_knowledge_graph_store.py`
- Modify: `tests/unit/test_graph_query_service.py`

**Interfaces:**
- Consumes: `KnowledgeGraphStore.get_neighborhood(entity_id)`.
- Produces: `GraphQueryService.show(entity_id: str) -> EntityNeighborhood` and complete SQLite inbound/outbound reads.

- [ ] **Step 1: Write failing service and integration tests**

Require malformed IDs (`""`, `"E1"`, wrong prefix/hex length) to fail before the
store and a valid `ent_` ID to delegate unchanged. For SQLite, construct one entity
with directed inbound/outbound edges, one symmetric `related_to`, multiple evidence
rows from different documents, and one isolated entity. Assert:

- the selected entity header data is exact;
- outgoing and incoming sides are correct;
- `related_to` is stored once and appears on the appropriate side of its canonical
  endpoints without duplication;
- relations order by relation type, neighbor display name, then neighbor ID;
- evidence order by source, chunk position, then quote;
- headings, chunk IDs, positions, sources, and exact quotes survive reopening;
- an isolated entity returns two empty tuples;
- a valid unknown ID raises `GraphEntityNotFoundError`;
- corrupt relation/evidence rows raise `GraphStorageError`.

- [ ] **Step 2: Run neighborhood tests and verify RED**

```bash
uv run pytest tests/unit/test_graph_query_service.py tests/integration/test_sqlite_knowledge_graph_store.py -q
```

Expected: failures because `show`/`get_neighborhood` are not implemented.

- [ ] **Step 3: Implement one-hop read assembly**

Validate the ID with the domain entity-ID grammar. Load the selected entity first,
then query outgoing and incoming relation rows with neighbor data. Load all evidence
for each relation, convert stored heading JSON to tuples of strings, sort by the
approved keys, and return frozen domain values. Do not truncate evidence and do not
deduplicate different provenance records.

For symmetric `related_to`, keep the canonical persisted direction; the CLI can
still label it as outgoing or incoming relative to the selected stored endpoint.
Do not synthesize a second relation.

- [ ] **Step 4: Run complete graph-store/query gates**

```bash
uv run pytest tests/unit/test_graph_query_service.py tests/integration/test_sqlite_knowledge_graph_store.py -q
uv run ruff format --check src/personal_knowledge_agent/application/graph_query.py src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py tests/unit/test_graph_query_service.py tests/integration/test_sqlite_knowledge_graph_store.py
uv run ruff check src/personal_knowledge_agent/application/graph_query.py src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py tests/unit/test_graph_query_service.py tests/integration/test_sqlite_knowledge_graph_store.py
uv run mypy src
```

Expected: all pass.

- [ ] **Step 5: Commit Task 8**

```bash
git add src/personal_knowledge_agent/application/graph_query.py src/personal_knowledge_agent/infrastructure/sqlite_knowledge_graph_store.py tests/unit/test_graph_query_service.py tests/integration/test_sqlite_knowledge_graph_store.py
git commit -m "feat: inspect one-hop graph neighborhoods"
```

---

### Task 9: Compose explicit graph builds in the CLI

**Files:**
- Modify: `src/personal_knowledge_agent/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 3–6 and existing LLM settings/model adapter.
- Produces: `pka graph build SOURCE --db PATH --graph-db PATH` and stable build output.

- [ ] **Step 1: Write failing parser and end-to-end build tests**

Require parser defaults `.pka/knowledge.db` and `.pka/graph.db`, explicit path
acceptance, mandatory source, and rejection of unknown options. Build a real
temporary knowledge DB with two chunks and use a scripted model with two graph JSON
outputs. Assert:

```text
Built graph for <resolved source>
Chunks processed: 2
Entities contributed: <document-unique count>
Relations contributed: <document-unique count>
Evidence records: <row count>
```

Also require exactly two model calls, per-call payload isolation, selected custom
graph DB creation, and an unindexed source failure before any model call or graph DB
creation.

- [ ] **Step 2: Add failing atomic failure and privacy CLI tests**

Seed an old successful graph, then make chunk two return provider failure or invalid
schema. Require exit 1, no success counts, route-independent safe error text, no
provider-body/API/vector/graph-ID sentinel in stdout/stderr, and unchanged old graph
rows. Require an empty extraction for every chunk to succeed and replace the old
contribution with zero counts.

- [ ] **Step 3: Run focused CLI tests and verify RED**

```bash
uv run pytest tests/test_cli.py -q -k "graph_build or graph_parser"
```

Expected: failures because the graph parser branch and composition root are absent.

- [ ] **Step 4: Add parser and lazy build composition**

Add one `graph` command with required nested subcommands. `build` receives `SOURCE`,
`--db`, and `--graph-db`. In `_graph_build`, construct in this order:

```python
settings = load_llm_settings(os.environ)
language_model = OpenAICompatibleChatModel(...)
source_reader = SQLiteGraphSourceReader(knowledge_database)
store = SQLiteKnowledgeGraphStore(graph_database)
service = GraphBuildService(
    source_reader,
    LLMKnowledgeGraphExtractor(language_model),
    store,
)
result = service.build(source)
```

Print the exact four counts from `GraphBuildResult`. Let existing `KnowledgeError`
handling produce exit 1 and safe stderr. Do not alter `ingest`, `search`, `ask`,
`agent`, or `memory` branches.

- [ ] **Step 5: Run build and compatibility CLI gates**

```bash
uv run pytest tests/test_cli.py -q
uv run ruff format --check src/personal_knowledge_agent/cli.py tests/test_cli.py
uv run ruff check src/personal_knowledge_agent/cli.py tests/test_cli.py
uv run mypy src
```

Expected: all pass.

- [ ] **Step 6: Commit Task 9**

```bash
git add src/personal_knowledge_agent/cli.py tests/test_cli.py
git commit -m "feat: build document knowledge graphs"
```

---

### Task 10: Expose local graph search and show commands

**Files:**
- Modify: `src/personal_knowledge_agent/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `GraphQueryService` and `SQLiteKnowledgeGraphStore` only.
- Produces: `pka graph search QUERY` and `pka graph show ENTITY_ID` output without model/settings/E5 construction.

- [ ] **Step 1: Write failing search CLI tests**

Require default/custom graph DB and limit parsing with the 1–20 boundary. Against a
real seeded graph, assert exact output:

```text
[ent_ab12...] method RAG score=0.0328
```

Require stable ordering, `No graph entities found.` only for a valid nonempty graph
with no lexical matches, safe empty-graph behavior, and no empty heading. Patch
`load_llm_settings`, `OpenAICompatibleChatModel`, and `E5Embedder` to raise if called.

- [ ] **Step 2: Write failing show CLI tests**

Seed inbound, outbound, multiple evidence, and isolated cases. Assert the exact
entity header, arrows, source/heading line, indented full quote, omitted empty
sections, evidence ordering, safe unknown ID behavior, and no model/settings/E5
construction.

- [ ] **Step 3: Run local CLI tests and verify RED**

```bash
uv run pytest tests/test_cli.py -q -k "graph_search or graph_show"
```

Expected: failures because query subcommands/output are absent.

- [ ] **Step 4: Implement local-only composition and rendering**

For both commands construct exactly:

```python
service = GraphQueryService(SQLiteKnowledgeGraphStore(graph_database))
```

`search` prints scores with four decimal places. `show` prints `Entity:`, then only
nonempty `Outgoing:`/`Incoming:` sections. Render relation arrows relative to the
selected entity and print every provenance row as source plus ` > `-joined headings,
then the complete evidence quote on an indented next line. Do not abbreviate quotes.

- [ ] **Step 5: Run the full CLI file and static checks**

```bash
uv run pytest tests/test_cli.py -q
uv run ruff format --check src/personal_knowledge_agent/cli.py tests/test_cli.py
uv run ruff check src/personal_knowledge_agent/cli.py tests/test_cli.py
uv run mypy src
```

Expected: all pass.

- [ ] **Step 6: Commit Task 10**

```bash
git add src/personal_knowledge_agent/cli.py tests/test_cli.py
git commit -m "feat: query local knowledge graphs"
```

---

### Task 11: Harden graph disclosure, lifecycle, and compatibility boundaries

**Files:**
- Modify: `tests/unit/test_structured_graph_parser.py`
- Modify: `tests/unit/test_graph_prompt_builder.py`
- Modify: `tests/unit/test_graph_build_service.py`
- Modify: `tests/integration/test_sqlite_graph_source_reader.py`
- Modify: `tests/integration/test_sqlite_knowledge_graph_store.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: the complete Phase 4A workflow.
- Produces: adversarial proof that graph output cannot escape its source, schema, atomicity, or provider-disclosure boundaries.

- [ ] **Step 1: Add adversarial tests before production changes**

Cover at least:

- a hostile chunk containing `SYSTEM`, fake section delimiters, SQL, fake `E99`,
  source-path, API-key, memory-ID, score, and vector sentinels remains untrusted data;
- provider messages contain the chunk text sentinel by design but omit every local
  metadata sentinel not present in the chunk;
- an evidence quote copied from a different chunk is rejected;
- a relation endpoint cannot borrow an entity from a previous chunk;
- two chunks with exact entity identity share one global entity, while `LLM` and
  `Large Language Model` stay separate;
- one source rebuild cannot remove another source's shared evidence;
- a failed final chunk causes no store call and an injected transaction failure
  preserves every prior row;
- search/show combined stdout/stderr omit document/chunk/graph IDs except the
  intentionally displayed entity ID and source provenance;
- search/show do not create missing knowledge DBs or load any remote/local model;
- existing non-graph CLI exact-output tests remain unchanged.

- [ ] **Step 2: Run the combined graph gate and inspect any RED**

```bash
uv run pytest tests/unit/test_graph_models.py tests/unit/test_structured_graph_parser.py tests/unit/test_graph_prompt_builder.py tests/unit/test_llm_knowledge_graph_extractor.py tests/unit/test_graph_build_service.py tests/unit/test_graph_query_service.py tests/integration/test_sqlite_graph_source_reader.py tests/integration/test_sqlite_knowledge_graph_store.py tests/test_cli.py -q
```

Expected: all new tests pass. If a real gap fails, use `systematic-debugging`, record
an explicit fix subtask in this plan with the failing assertion and exact production
path, then apply the smallest TDD fix; do not make generic hardening edits.

- [ ] **Step 3: Run the Phase 0–4 compatibility gate**

```bash
uv run pytest tests/integration tests/unit tests/test_cli.py -q
uv run ruff format --check .
uv run ruff check .
uv run mypy src
```

Expected: all pass with no network calls.

- [ ] **Step 4: Commit Task 11 as tests-only unless RED justified production files**

```bash
git add tests/unit/test_structured_graph_parser.py tests/unit/test_graph_prompt_builder.py tests/unit/test_graph_build_service.py tests/integration/test_sqlite_graph_source_reader.py tests/integration/test_sqlite_knowledge_graph_store.py tests/test_cli.py
git diff --cached --name-only
git commit -m "test: harden knowledge graph boundaries"
```

Before committing, confirm `.env`, `.pka`, `papers/`, `references/`, and `research/`
are absent. If a verified fix changed production, stage only its exact files and use
a separate `fix:` commit after its focused test passes.

---

### Task 12: Document Phase 4A and complete the merge gate

**Files:**
- Modify: `README.md`
- Create: `docs/learning-log/phase-4a.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: completed Phase 4A behavior.
- Produces: accurate usage/privacy/cost documentation, portfolio learning notes, and merge-ready verification evidence.

- [ ] **Step 1: Update README usage and boundaries**

Document exact build/search/show commands and output, controlled ontology, per-chunk
remote calls and retries, exact quote provenance, independent `.pka/graph.db`,
rebuild-after-ingest requirement, unencrypted storage/erasure limitations, local-only
query behavior, and every Phase 4A exclusion. State clearly that graph data does not
influence `ask`, `agent`, routing, or memory.

- [ ] **Step 2: Write the Phase 4A learning log**

Explain with small English/Chinese examples:

- ontology control versus open information extraction;
- entity identity versus similarity and why false duplicates are safer than false
  merges;
- relation direction and symmetric canonicalization;
- exact quote provenance and why validated structure is not semantic truth;
- per-chunk isolation, cost, and atomic document replacement;
- shared cross-document evidence and orphan cleanup;
- relational SQLite as a sufficient graph substrate for one-hop queries;
- why GraphRAG and evaluation remain future work.

- [ ] **Step 3: Update contributor scope only after behavior is complete**

Mark Phase 4A complete in `AGENTS.md` with controlled extraction, provenance-aware
SQLite persistence, local lexical entity search, and one-hop inspection. Keep
GraphRAG/multi-hop, PDF/web/code ingestion, formal evaluation, and deployment in
future scope.

- [ ] **Step 4: Run the complete quality gate**

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pka health
git status --short
```

Expected: every test passes; Ruff and mypy are clean; health prints
`Personal Knowledge Agent is healthy.`; only the three documentation files are
modified/untracked for this task, while user-owned directories remain untouched.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/learning-log/phase-4a.md AGENTS.md
git commit -m "docs: explain phase 4a knowledge graphs"
```

- [ ] **Step 6: Request final review and finish the branch**

Use `requesting-code-review` with the Phase 4A base/head SHAs and approved spec. Fix
all Critical and Important findings with focused RED/GREEN tests and rerun the full
quality gate. Then use `finishing-a-development-branch` to offer local merge, PR,
keep, or discard; never merge with failing checks.

## Final acceptance checklist

- [ ] `graph build` reads only an indexed source and calls the LLM exactly once per chunk.
- [ ] Every relation has a validated exact quote and full current provenance.
- [ ] Per-source replacement is atomic and preserves facts supported by other sources.
- [ ] Exact entity identity is deterministic; no fuzzy, alias, or embedding merge exists.
- [ ] Entity BM25 search and one-hop show are deterministic and model-free.
- [ ] Graph data never enters RAG, Agent, router, or memory paths.
- [ ] Provider payload omits source paths, stable IDs, scores, vectors, memories, and credentials.
- [ ] Existing Phase 0–3 behavior and exact CLI contracts remain compatible.
- [ ] README, Phase 4A learning log, and `AGENTS.md` match implemented behavior.
- [ ] Full pytest, Ruff format/check, strict mypy, and health gates pass.

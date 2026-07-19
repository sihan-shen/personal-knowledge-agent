# Phase 4A: Provenance-First Knowledge Graph Foundation

**Status:** Approved design
**Date:** 2026-07-19
**Phase:** 4A

## Objective

Phase 4A derives a small, inspectable knowledge graph from Markdown documents that
are already present in the local knowledge index. A user explicitly builds one
document at a time, an LLM proposes typed entities and relations per chunk, local
code strictly validates every proposal, and SQLite atomically replaces that
document's graph contribution.

The result is queryable through local entity search and one-hop neighborhood
commands. Every displayed relation retains exact supporting text and source
provenance. The graph does not yet influence `pka ask`, `pka agent`, routing,
memory, or answer generation.

## Learning objectives

The implementation should make these AI Engineer concepts explicit:

- a knowledge graph is a typed, provenance-bearing representation rather than a
  collection of unverified LLM claims;
- ontology design trades coverage for consistency and explainability;
- LLM extraction proposes structured data, while deterministic code owns schema,
  referential, cardinality, and evidence validation;
- entity identity and entity similarity are different problems;
- document-level replacement requires provenance-aware many-to-many lifecycle
  rules;
- graph construction can use an external model while graph lookup remains local;
- GraphRAG should be built only after graph quality can be inspected independently.

## Approved product decisions

| Decision | Approved behavior | Reason |
| --- | --- | --- |
| First Phase 4 deliverable | Build and inspect a graph from indexed documents | Establish graph quality before graph-assisted answering. |
| Build trigger | Explicit `pka graph build` | Preserve deterministic local `ingest` and make remote cost intentional. |
| Build unit | One already-indexed resolved source path | Bound cost and support atomic source replacement. |
| Extraction unit | One LLM request per chunk | Keep context, failure, and provenance boundaries precise. |
| Graph database | Dedicated SQLite database | Avoid a service dependency while one-hop queries remain simple. |
| Ontology | Small controlled AI-learning schema | Prevent type and relation vocabulary drift. |
| Entity identity | Exact type plus normalized canonical name | Prefer visible duplicates over irreversible false merges. |
| Entity retrieval | Local lexical BM25 | Avoid hiding identity defects behind semantic matching. |
| Graph lookup | Entity search plus one-hop inbound/outbound relations | Provide enough inspection to evaluate extraction quality. |
| Provenance | Exact source quote, chunk, heading, and document source | Keep every relation auditable. |
| Replacement | All-or-nothing per document | Never expose a partially rebuilt graph contribution. |

## Scope

### In scope

- reading one resolved, already-indexed Markdown source and its current chunks;
- per-chunk structured entity and relation extraction through the existing
  provider-neutral language-model boundary;
- controlled entity and relation types;
- strict JSON, reference, size, cardinality, and exact-quote validation;
- deterministic identity for entities and relations;
- cross-chunk and cross-document exact entity reuse;
- a dedicated SQLite graph schema with source contribution provenance;
- atomic document contribution replacement and orphan cleanup;
- local BM25 entity search;
- local one-hop inbound and outbound relation display;
- CLI commands for build, search, and show;
- privacy, cost, failure, compatibility, and lifecycle documentation;
- a Phase 4A learning log.

### Out of scope

- graph context in `pka ask`, `pka agent`, or router prompts;
- GraphRAG, graph query planning, or natural-language graph queries;
- multi-hop traversal, shortest paths, path ranking, or cycle presentation;
- aliases, fuzzy matching, entity merging, or embedding-based entity resolution;
- user editing, deleting, or manually merging graph records;
- Neo4j or another external graph service;
- graph visualization or a web UI;
- extracting a graph from long-term memory;
- automatic graph rebuild during `pka ingest`;
- background synchronization or stale-graph detection commands;
- PDF, web, or code-document ingestion;
- formal evaluation or deployment.

## Existing boundaries to preserve

- `pka ingest` parses Markdown locally and atomically replaces the document's
  knowledge chunks, lexical postings, and embeddings.
- `pka search`, `pka ask`, and `pka agent` operate through the knowledge index and
  must not query the graph in Phase 4A.
- memory persists in its own SQLite database and must not enter graph extraction.
- the domain layer remains independent of SQLite and provider libraries.
- provider-specific behavior remains behind `LanguageModel`.
- existing CLI output and all established Phase 0–3 behavior remain compatible.

## Architecture

```text
pka graph build SOURCE
        |
        v
GraphSourceReader
  read current indexed source metadata + ordered Chunks from knowledge.db
        |
        v
GraphBuildService
        |
        +-- for each chunk --> KnowledgeGraphExtractor
        |                         |
        |                         +-- prompt builder
        |                         +-- LanguageModel
        |                         +-- strict parser/validator
        |
        +-- merge exact entity identities across chunk results
        |
        v
KnowledgeGraphStore.replace_document(...)
  one SQLite transaction in graph.db

pka graph search QUERY ------> GraphQueryService ------> graph.db BM25
pka graph show ENTITY_ID ----> GraphQueryService ------> one-hop graph.db query
```

## Component responsibilities

### `GraphSourceReader`

The domain protocol loads a graph build source by resolved path from the selected
knowledge database:

```python
class GraphSourceReader(Protocol):
    def load(self, source: Path) -> GraphSourceDocument: ...
```

`GraphSourceDocument` is a graph-specific immutable value with `document_id`,
resolved `source`, the
persisted document `content_hash`, and ordered `Chunk` values. It does not require
the original `Document.content`, which the knowledge database intentionally does not
store. The SQLite implementation reads existing persisted data and does not reopen
or reparse the original Markdown file. A source that is absent from the index
produces a dedicated safe domain error.

### `KnowledgeGraphExtractor`

The protocol converts exactly one chunk into a validated domain result:

```python
class KnowledgeGraphExtractor(Protocol):
    def extract(self, chunk: Chunk) -> GraphChunkExtraction: ...
```

The infrastructure implementation builds an untrusted-data prompt, calls the
existing `LanguageModel`, decodes strict JSON, validates local IDs and evidence, and
returns domain objects. It cannot write graph state.

### `GraphBuildService`

The application service:

1. validates the source and selected options;
2. loads the current indexed source;
3. calls the extractor once for every chunk in deterministic position order;
4. stops immediately on the first extraction failure;
5. combines all successful chunk extractions in memory;
6. calls `KnowledgeGraphStore.replace_document` exactly once;
7. returns counts for the CLI.

If any extraction fails, the store is not called. Remote requests completed before
the failure may still be billable, but no partial graph state is written.

### `KnowledgeGraphStore`

The domain protocol owns persistence and local queries:

```python
class KnowledgeGraphStore(Protocol):
    def replace_document(
        self,
        source_document: GraphSourceDocument,
        extractions: Sequence[GraphChunkExtraction],
    ) -> GraphBuildResult: ...

    def search_entities(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[EntitySearchResult]: ...

    def get_neighborhood(self, entity_id: str) -> EntityNeighborhood: ...
```

The SQLite implementation owns schema migration, transactions, BM25 ranking,
source replacement, shared provenance, and orphan cleanup.

### `GraphQueryService`

The application service validates query text, limits, and entity IDs, then delegates
to the store. It contains no SQL, ranking implementation, or model logic.

## Controlled ontology

### Entity types

The allowed `EntityType` values are:

- `concept` — an idea, principle, or abstract subject;
- `technology` — a concrete framework, model, database, language, or tool;
- `method` — a technique, algorithm, workflow, or engineering practice;
- `project` — a named system or project being built or discussed;
- `resource` — a paper, book, course, dataset, repository, or learning artifact.

An entity contains no free-form description in Phase 4A. Its meaning is inspectable
through typed relations and source mentions.

### Relation types

The allowed `RelationType` values are:

- `is_a`;
- `part_of`;
- `uses`;
- `depends_on`;
- `implements`;
- `related_to`.

Relations are directed except `related_to`, which is semantically symmetric. The
store canonicalizes `related_to` endpoint order by entity ID, so the graph cannot
hold both `A related_to B` and `B related_to A` as separate relations. Self-loops are
rejected for every relation type. The extraction prompt instructs the model to use
`related_to` only when no more specific approved relation is supported.

## Identity and normalization

### Entity identity

Entity names are normalized with Unicode NFKC, whitespace collapsing, trimming, and
Unicode-aware `casefold()`. Identity is:

```text
(entity_type, normalized_name)
```

The entity ID is a stable prefixed hash of those two values. Entity names are at
most 120 Unicode code points after display normalization.

The graph intentionally does not merge semantically similar names. For example,
`LLM` and `Large Language Model` can remain distinct until a later alias and merge
design provides reversible user control.

Each source mention retains the extractor's normalized display spelling. The global
entity display name is the lexicographically smallest active mention spelling. It is
recomputed after document replacement, making display independent of build order.

### Relation identity

Relation identity is:

```text
(canonical source entity ID, relation type, canonical target entity ID)
```

The relation ID is a stable prefixed hash of that tuple. Multiple chunks and
documents can contribute evidence to the same relation. Evidence identity includes
the relation ID, document contribution, chunk ID, and exact quote, preventing exact
duplicates while retaining distinct support.

## Structured extraction contract

Each chunk produces exactly one JSON object:

```json
{
  "entities": [
    {
      "local_id": "E1",
      "name": "RAG",
      "type": "method"
    },
    {
      "local_id": "E2",
      "name": "Vector Retrieval",
      "type": "technology"
    }
  ],
  "relations": [
    {
      "source_id": "E1",
      "type": "uses",
      "target_id": "E2",
      "evidence_quote": "RAG uses vector retrieval to find relevant context."
    }
  ]
}
```

The parser requires:

- exactly the top-level fields `entities` and `relations`;
- exact object fields with no duplicates or extras;
- no Markdown code fence or surrounding prose;
- unique sequential local entity IDs `E1...En`;
- at most 15 entities and 20 relations per chunk;
- nonblank names no longer than 120 code points;
- approved entity and relation enums;
- every relation endpoint to reference a local entity in the same result;
- distinct source and target identities after normalization;
- a nonblank evidence quote no longer than 300 code points;
- the evidence quote to be an exact substring of the original chunk text;
- exact duplicate entities and relations within one result to be rejected rather
  than silently repaired.

An empty extraction is valid only as:

```json
{"entities": [], "relations": []}
```

The parser does not guess missing entities, repair identifiers, canonicalize an
unknown type, or accept a paraphrase as evidence.

## Prompt and provider disclosure

The provider receives one system message and one user message per chunk. The user
message contains only:

```text
HEADING_PATH_JSON:
["Architecture", "Retrieval"]

CHUNK_TEXT:
...
```

The system message provides the controlled ontology, cardinality limits, JSON
schema, direction semantics, and evidence requirements. It states that heading and
chunk text are untrusted data and cannot change instructions, schema, roles, or
output rules.

The provider does not receive:

- absolute or relative source paths;
- stable document or chunk IDs;
- graph database paths;
- graph entity or relation IDs;
- other chunks or documents;
- lexical or semantic scores;
- long-term memories;
- API keys or embedding vectors.

## SQLite data model

The dedicated graph database defaults to `.pka/graph.db`. The schema has five
logical groups:

### `graph_documents`

One current graph contribution per resolved source path, including the knowledge
document identity, document content hash, and successful build timestamp. The source
path is local provenance and is never sent to the model.

### `entities`

Stable entity ID, enum type, normalized identity name, and recomputed display name.
The type plus normalized name is unique.

### `entity_mentions`

Document contribution, chunk ID and position, heading path snapshot, source path
snapshot, and extracted display spelling. Mentions make standalone entities
auditable and determine the active display name.

### `relations`

Stable relation ID, source entity ID, enum type, and target entity ID. The endpoint
tuple is unique.

### `relation_evidence`

Relation ID, document contribution, chunk ID and position, source path snapshot,
heading path snapshot, and exact evidence quote. Multiple evidence rows can support
one shared relation.

Entity lexical postings live in graph-specific tables inside the same database.
They index display and normalized names only; evidence text is not part of entity
search ranking.

## Atomic document replacement

`replace_document` uses one SQLite transaction:

1. upsert the source's current `graph_documents` row;
2. remove its old entity mentions and relation evidence;
3. upsert exact entity identities;
4. insert new mentions;
5. upsert relation identities;
6. insert new relation evidence;
7. remove relations with no remaining evidence;
8. remove entities with no remaining mentions and no incident relations;
9. recompute display names and lexical postings for affected entities;
10. commit.

An empty extraction replaces the source's old contribution with an empty one and
can clean up orphaned records. If another document supports an entity or relation,
that shared record and its other provenance remain.

Any SQLite, schema, serialization, or invariant failure rolls back the complete
transaction and maps to a safe graph storage error.

## Local query behavior

### Entity search

```bash
pka graph search QUERY --limit 10 --graph-db .pka/graph.db
```

The supported limit is 1–20. The mixed Chinese/English tokenizer and BM25 rank
entity display and normalized names. Results are ordered by descending score and
then entity ID. No E5 model or remote language model is constructed.

Example:

```text
[ent_ab12...] method RAG score=0.0328
[ent_cd34...] concept Retrieval-Augmented Generation score=0.0161
```

No matches print `No graph entities found.` A missing or empty graph database is a
safe user-facing empty-graph error rather than implicit graph construction.

### One-hop neighborhood

```bash
pka graph show ent_ab12 --graph-db .pka/graph.db
```

The result contains the selected entity, outgoing relations, incoming relations,
neighbor entities, and every current evidence row. Relations are ordered by relation
type, neighbor display name, and neighbor ID. Evidence is ordered by source path,
chunk position, and quote.

Example:

```text
Entity: [ent_ab12...] method RAG

Outgoing:
uses -> [ent_cd34...] technology Vector Retrieval
  [1] /notes/rag.md > Retrieval
      RAG uses vector retrieval to find relevant context.

Incoming:
part_of <- [ent_ef56...] project Personal Knowledge Agent
  [1] /notes/project.md > Architecture
      RAG is part of the Personal Knowledge Agent.
```

An entity with no relations still prints its entity header. Empty inbound or
outbound sections are omitted. An unknown entity ID produces a dedicated safe
not-found error.

## CLI composition

The commands are:

```bash
pka graph build SOURCE \
  --db .pka/knowledge.db \
  --graph-db .pka/graph.db

pka graph search QUERY \
  --limit 10 \
  --graph-db .pka/graph.db

pka graph show ENTITY_ID \
  --graph-db .pka/graph.db
```

`graph build` loads LLM settings, constructs the existing Chat Completions adapter,
reads the knowledge source, and creates the graph store. It makes one logical model
request per chunk. SDK retries can increase actual billable attempts.

`graph search` and `graph show` construct only the local graph store. They do not
load LLM settings, instantiate the Chat Completions adapter, or load E5.

A successful build prints:

```text
Built graph for /notes/rag.md
Chunks processed: 6
Entities contributed: 14
Relations contributed: 18
Evidence records: 21
```

Counts describe the rebuilt document contribution, not all globally shared rows.

## Lifecycle and consistency boundaries

- every explicit build re-extracts every current chunk; Phase 4A has no extraction
  cache or `--force` flag;
- reingesting a source does not automatically rebuild its graph;
- the graph remains the last successfully built snapshot until the user rebuilds;
- the stored document identity and content hash describe that snapshot;
- Phase 4A provides no background task or status command for stale detection;
- build output must explain that a later reingest requires another graph build;
- graph query commands read their local provenance snapshot and do not reopen the
  knowledge database.

## Error behavior

### Build failures

- source absent from the knowledge index: safe source-not-indexed error;
- invalid or empty source arguments: local validation error before dependencies;
- provider/auth/rate/timeout/connection/server failure: existing safe model error;
- truncated, malformed, duplicate-key, wrong-schema, over-limit, bad-reference, or
  non-verbatim evidence output: safe graph extraction error;
- any chunk failure: no call to graph replacement and no partial success output;
- SQLite/schema/transaction failure: safe graph storage error and rollback.

Raw provider bodies, prompts, credentials, database internals, and tracebacks are not
printed. A user-supplied source path may appear in a successful build or a
source-not-indexed message.

### Query failures

- blank query or out-of-range limit: local validation error;
- graph database missing or containing no entities: safe empty-graph error;
- malformed entity ID: local validation error;
- well-formed unknown entity ID: safe not-found error;
- corrupt schema or stored data: safe graph storage error.

## Privacy and cost properties

- graph construction is an explicit remote-data-disclosure action;
- each current chunk's heading and full text are sent separately to the configured
  provider;
- source paths and stable local IDs stay local;
- completed calls remain potentially billable if a later chunk fails;
- successful validation does not prove that extraction is semantically correct,
  which is why provenance remains visible;
- the graph database is local and unencrypted;
- rebuilding and orphan cleanup remove application rows but cannot guarantee secure
  erasure from SQLite pages, filesystem snapshots, caches, or provider systems;
- entity search and one-hop display make zero remote calls.

## Testing strategy

### Domain tests

- enum membership, text normalization, and size bounds;
- stable entity and relation IDs;
- entity identity across case and Unicode normalization;
- directed relation identity and symmetric `related_to` canonicalization;
- self-loop, missing endpoint, empty evidence, and malformed ID rejection;
- frozen immutable models and deterministic ordering contracts.

### Prompt and parser tests

- exact top-level and nested fields;
- duplicate JSON key and code-fence rejection;
- empty extraction acceptance;
- cardinality and length limits;
- sequential unique local IDs;
- unknown endpoint and unsupported enum rejection;
- exact evidence substring acceptance and paraphrase rejection;
- hostile `SYSTEM`, fake JSON delimiter, and fake local ID content remains data;
- provider payload contains only heading path, chunk text, schema, and policy;
- source paths, stable IDs, scores, vectors, memories, and API keys remain absent.

### Application tests

- source validation happens before dependencies;
- source reader is called once;
- every ordered chunk is extracted exactly once;
- the first failure stops later calls and prevents store mutation;
- exact identities merge across chunks without semantic merging;
- all successful results reach one replacement call;
- an empty document graph can replace old contribution;
- build result counts are document-local and deterministic;
- graph query validation and delegation remain model-free.

### SQLite integration tests

- initial schema creation and reopen behavior;
- atomic document insert and replacement;
- two documents share entity and relation identities;
- rebuilding one source preserves another source's evidence;
- empty rebuild removes only true orphans;
- display-name recomputation is independent of build order;
- BM25 entity ranking, limits, Chinese/English tokens, and stable tie-breaks;
- inbound/outbound neighborhood ordering and complete evidence;
- rollback under duplicate IDs, injected failures, and invalid stored data;
- corrupted schema and provenance map to safe graph storage errors.

### CLI and compatibility tests

- parser defaults and custom knowledge/graph database paths;
- complete build, search, and show flows with temporary SQLite databases;
- one scripted extraction output per chunk and no real network;
- unindexed source and mid-document extraction failures preserve the old graph;
- exact source, heading, quote, direction, and count output;
- graph search/show do not instantiate LLM, settings, or E5 dependencies;
- private provider and storage details never reach combined stdout/stderr;
- all existing `ingest`, `search`, `ask`, `agent`, and `memory` behavior remains
  byte-for-byte compatible where output is asserted.

## Documentation deliverables

- update `README.md` with graph build/search/show examples, remote-call cost,
  provenance semantics, stale-snapshot behavior, and Phase 4A boundaries;
- add `docs/learning-log/phase-4a.md` explaining ontologies, entity identity,
  relation direction, provenance, atomic replacement, and why GraphRAG is deferred;
- update `AGENTS.md` to mark Phase 4A complete only after implementation and tests.

## Complete quality gate

Before integration:

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pka health
```

Commands use the repository's required `TMPDIR=/tmp` and `UV_LINK_MODE=copy`
environment on the Windows-mounted workspace.

## Acceptance criteria

Phase 4A is complete when:

1. an explicitly selected indexed Markdown source can be extracted chunk by chunk
   into the controlled entity and relation schema;
2. every persisted relation has at least one exact, visible chunk quote and complete
   local provenance;
3. a failed build leaves the source's previous graph contribution unchanged;
4. rebuilding one source correctly preserves graph facts supported elsewhere;
5. entity identity is deterministic and never uses fuzzy or semantic auto-merge;
6. entity search and one-hop display work locally with stable ordering;
7. graph search/show create no remote or embedding dependencies;
8. provider payload excludes paths, stable IDs, scores, memories, and credentials;
9. no graph data influences existing RAG, Agent, routing, or memory behavior;
10. deterministic tests cover extraction, validation, lifecycle, privacy, failure,
    query, and compatibility paths;
11. README, learning log, and contributor scope accurately describe the feature;
12. the complete project quality gate passes.

## Future work

Later Phase 4 designs may add aliases, reversible entity merge administration,
multi-hop traversal, graph-assisted retrieval, graph citations in answers, and
GraphRAG evaluation. PDF/web/code ingestion, formal evaluation infrastructure, and
deployment remain separately scoped roadmap items.

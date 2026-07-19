# Phase 4A Learning Log: Provenance-First Knowledge Graphs

## What we built

Phase 4A adds explicit, chunk-by-chunk knowledge-graph extraction from
indexed Markdown documents. Each document is processed through a deterministic
pipeline that reads existing chunks from the knowledge database, sends them
one at a time to an LLM for structured entity/relation extraction, strictly
validates the output, and atomically replaces the document's graph
contribution in a dedicated SQLite database.

## Key engineering concepts

### Ontology control versus open information extraction
We defined five entity types (`concept`, `technology`, `method`, `project`,
`resource`) and six relation types (`is_a`, `part_of`, `uses`, `depends_on`,
`implements`, `related_to`). This controlled vocabulary trades coverage for
consistency and explainability — you always know what shape the graph data
takes, and a schema violation is always a bug, not a data point.

### Entity identity versus similarity
Entity identity is exact: `(entity_type, NFKC/casefold/whitespace-normalized
name)`. Two chunks that mention "RAG" and "rag" produce the same global
entity. But "LLM" and "Large Language Model" remain separate. This is a
deliberate design choice: false duplicates are visible (two nodes with
similar names), while false merges are invisible and irreversible.

### Relation direction and symmetric canonicalization
`related_to` is symmetric; the relation ID is computed by sorting the two
endpoint entity IDs before hashing, so reversing the endpoints produces the
same canonical ID. Directed types (`uses`, `depends_on`, `implements`, etc.)
use the supplied order, so `A uses B` and `B uses A` are different relations.

### Exact quote provenance
Every relation carries a validated exact substring of its source chunk text.
This means every graph fact is auditable: you can see exactly what the LLM
saw and quoted. The parser rejects paraphrased quotes, blank quotes, and
quotes exceeding 300 characters.

### Per-chunk isolation and cost
The LLM makes one request per chunk. Each request sees only its own chunk
text and heading path — no document ID, source path, or other document's
data. This bounds the per-chunk context window and makes failures
deterministic. The `pka graph build` command exits on the first extraction
failure, so a bad chunk never writes partial data.

### Cross-document shared entities
Because entity identity is exact, two documents mentioning the same concept
share one global entity in the graph database. When one document is rebuilt,
the store deletes only that document's contributions and orphans, leaving
the shared entity intact.

### BM25 entity search
Entity search is purely local — no LLM or embedding model required. The
tokenizer from Phase 1 handles mixed Chinese/English, and BM25 scores are
computed over entity display and normalized names.

### Future work (not in scope)
- GraphRAG or multi-hop reasoning
- Natural-language graph queries
- Entity merging or fuzzy matching
- Neo4j or external graph services
- Formal evaluation
- Automatic rebuild during ingest

## Why SQLite is sufficient here
One-hop neighborhood queries over a few thousand entities are trivially
served by indexed SQLite. The `graph_edges` table stores each relation
instance with its full provenance, and entity-lookup queries use primary-key
scans. A dedicated graph database like Neo4j would add operational complexity
without benefit at this scale.

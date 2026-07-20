# Phase 4B Learning Log: Explicit Graph-Grounded Answering

## What we built

Phase 4B adds `pka graph ask`, an explicit graph-only answer path. It searches the
whole question against graph entity names with local BM25, selects one to five seeds
(default three), expands their one-hop neighborhoods, and supplies at most twelve
deduplicated relation-plus-provenance records to the configured answer provider.
Validated answers cite that evidence with `[G#]` markers and render a `Graph evidence`
section containing only cited relations and their local provenance.

## GraphRAG versus document RAG

This is intentionally not a replacement for `pka ask`. Document RAG retrieves
Markdown chunks from the knowledge database; graph answering begins with graph
entities and follows only stored relationships. `pka graph ask` does not open the
document `knowledge.db` or construct the document RAG pipeline, E5 embeddings, Agent
router, memory store, LLM query planner, or a fallback to another answer mode. The
boundary keeps its evidence source and its failure behavior inspectable.

## Lexical entity seeds and their limitations

Entity seeds come from mixed Chinese/English BM25 over graph entity names. This is
cheap, local, and deterministic, but it matches wording rather than meaning. An exact
entity-name question can seed the right entity; a paraphrase or an unrecorded alias
may receive no useful seed. We retain the whole question for search rather than
having an LLM invent a query, so a miss is visible and fails closed.

## Relation direction and provenance-level citations

Evidence represents a directed relation together with one exact provenance record.
For an incoming neighborhood edge, the answer evidence preserves the stored source →
target direction rather than reversing the fact around the selected seed. Each `[G#]`
citation therefore identifies a precise relation and quote, while the local `Graph
evidence` section shows its source path, heading path, and exact quote. The provider
does not receive those local paths or stable graph identifiers.

## Deterministic ordering, deduplication, and budgets

Candidates are ordered deterministically by seed rank, relation direction, relation
and neighbor identity, then provenance. Exact duplicate relation/provenance records
are removed before assigning citation IDs. The command limits seeds to 1–5 and
evidence to 12 records, bounding graph traversal and provider context while keeping
the rendered citations reproducible.

## Strict structured output and prompt-injection boundaries

The provider must return a strict JSON answer with an allowed status, supplied
citation IDs, and matching inline `[G#]` markers. Local validation rejects malformed,
unknown, reordered, or unsupported citations; insufficient-evidence output may not
cite anything. Entity names, headings, and quotes are untrusted data in the prompt,
and the system instruction says that embedded instructions cannot change the answer
policy. This is defence in depth, not a complete solution to prompt injection.

The provider receives only the question plus request-local citation IDs, source and
target entity types and names, relation types, heading paths, and evidence quotes.
It does not receive source paths, graph/entity/relation/chunk IDs, chunk positions,
database paths, scores, credentials, or other local storage metadata. Quotes may be
private, so graph answering still requires a trusted configured provider when local
evidence exists.

## Lazy model construction and failure-closed refusal

The command searches and expands graph evidence before constructing model settings or
a provider client. If no relationship evidence is found, it returns a local
insufficient-evidence refusal with zero model calls and no API configuration needed.
This failure-closed path does not fall back to document RAG, the Agent, or memory.

## Experiments to run

- Compare an exact entity-name question with its expected local BM25 seed.
- Try a paraphrase that BM25 cannot link to an entity and confirm the local refusal.
- Build one relation with multiple provenance records and inspect distinct `[G#]`
  citations and quotes.
- Ask through an incoming relation and confirm the rendered edge keeps its original
  direction.
- Ask a no-match question and confirm it makes no model call and needs no API
  configuration.

## Deferred work

- Evaluation
- Hybrid document/graph context
- Agent routing
- LLM query planning
- Multi-hop paths
- Alias management

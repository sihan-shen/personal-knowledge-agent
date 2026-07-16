# Personal Knowledge Agent TODO

This file tracks product milestones. Detailed design decisions and implementation
steps belong in `docs/superpowers/`.

## In progress: Phase 1B — local vector and hybrid retrieval

- [x] Approve the local, CPU-friendly `intfloat/multilingual-e5-small` model.
- [x] Approve and commit the Phase 1B architecture and implementation plan.
- [x] Define `Embedding`, `SearchMode`, `EmbeddingModel`, and related tests.
- [x] Add the lazy E5 adapter with `passage:` and `query:` prefixes.
- [x] Add deterministic fake-embedder tests without model downloads.
- [x] Migrate SQLite from schema v1 to v2 and persist float32 embeddings.
- [x] Guarantee re-ingestion atomically replaces postings and vectors together.
- [x] Implement exact cosine semantic retrieval.
- [x] Implement RRF fusion and `lexical`, `semantic`, `hybrid` search modes.
- [ ] Expose search mode through `pka search --mode`.
- [ ] Add semantic-synonym, persistence, migration, CLI, and rollback tests.
- [ ] Run a manual real-model smoke test and record findings in the learning log.

## Phase 1 completion criteria

- [ ] Support clear comparison of lexical, semantic, and hybrid retrieval.
- [ ] Keep the local knowledge database persistent and reproducible.
- [ ] Document retrieval trade-offs and known limitations.

## Phase 2 — Agent capability

- [ ] Define a question-routing policy: direct answer, retrieval, or tool use.
- [ ] Add an LLM provider boundary and configuration validation.
- [ ] Implement a minimal retrieval tool with observable inputs and outputs.
- [ ] Add prompt templates and tool-selection tests.
- [ ] Capture agent traces for debugging.

## Phase 3 — Long-term memory

- [ ] Separate user/profile memory from retrieved knowledge documents.
- [ ] Define memory write, retrieval, expiry, and deletion rules.
- [ ] Add transparent memory inspection and reset commands.

## Phase 4 — Knowledge graph

- [ ] Define entities, relations, provenance, and confidence records.
- [ ] Extract graph candidates from ingested content.
- [ ] Combine graph traversal with RAG retrieval without hiding provenance.

## Phase 5 — Evaluation

- [ ] Create a versioned evaluation dataset with expected evidence.
- [ ] Measure retrieval recall and ranking quality.
- [ ] Evaluate answer faithfulness, relevance, and citation quality.
- [ ] Add regression evaluation to the quality workflow.

## Phase 6 — Deployment and engineering

- [ ] Add configuration profiles and secure secret handling.
- [ ] Package a usable service or UI.
- [ ] Add structured logging, health checks, and error observability.
- [ ] Profile retrieval latency and introduce ANN/vector infrastructure only when
  measured corpus size or latency warrants it.
- [ ] Prepare architecture, trade-off, and evaluation material for the project
  portfolio and internship résumé.

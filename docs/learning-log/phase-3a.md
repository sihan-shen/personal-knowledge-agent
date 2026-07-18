# Phase 3A Learning Log

## What this phase built

Phase 3A adds a complete, confirmation-gated long-term-memory lifecycle without yet
allowing memory to influence an answer. A user can create an active memory directly,
or `pka agent` can propose a pending candidate after a successful response. The user
then approves, replaces, keeps both, rejects, searches, lists, or forgets records
through explicit commands.

```text
successful pka agent answer
  -> local first-person prefilter
     -> miss: stop locally
     -> hit: raw-message safety screen
        -> safe: structured extraction request
           -> strict parse and proposal safety screen
              -> one atomic pending write
                 -> user approves, replaces, keeps both, or rejects
```

The deliberate Phase 3A boundary is that none of these records enters routing,
retrieval for answering, or generation prompts. That separation lets the project
test write authority, privacy, lifecycle transitions, and deletion before memory
becomes another source of model context.

## Long-term memory is not chat history or document knowledge

These three forms of context answer different questions:

- Chat history preserves the sequence of one conversation. For example, “that
  project” can refer to a project named two messages earlier. This project still has
  no multi-turn session history.
- Document knowledge stores source material that can support an answer, such as a
  Markdown note explaining RRF. It is retrieved as evidence and cited with `S1`,
  `S2`, and so on.
- Long-term memory stores a durable user fact, such as “I prefer concise
  explanations” or “My goal is to finish an AI Engineer portfolio.” It has user
  ownership and lifecycle semantics rather than document provenance.

Treating a preference as a synthetic document would blur user intent, pending
approval, replacement, and forgetting with evidence ingestion. Phase 3A therefore
uses a dedicated memory protocol and database rather than the knowledge index.

## The LLM proposes; the user authorizes

Fluent extraction is not write authority. A model can misunderstand a temporary
statement or confidently rewrite it into a false durable fact. Phase 3A therefore
allows the extractor to return only proposals. For example:

```text
User: My goal is to complete a portfolio-ready knowledge agent.
Model proposal: goal / goal.personal_knowledge_agent / Complete a portfolio-ready knowledge agent.
Stored state: pending
```

The candidate becomes active only after `pka memory approve <candidate-id>`. The
extractor cannot call the store directly, approve itself, or silently replace an old
value. By contrast, `pka memory remember --type ... --key ... CONTENT` is already an
explicit user action, so it may create an active record without an LLM call.

This split turns a model suggestion into an auditable side effect: proposal and
authorization are separate events, and every automatic candidate remains visible
until the user chooses its outcome.

## Lifecycle transitions and transactional replacement

The persistent statuses are `pending`, `active`, and `superseded`. Rejection is not
a fourth memory status because rejected content is deleted.

```text
automatic proposal -> pending -> approve -> active
                                  replace -> active (new)
                                             + superseded (old)
                                  keep both -> active (new and old)
                                  reject -> content deleted
```

Suppose `mem_old` is the active value for `goal.portfolio` and `mem_new` is a pending
candidate for the same slot. Approval with `--replace mem_old` performs two changes
in one SQLite transaction: `mem_old` becomes superseded and `mem_new` becomes active
with a replacement link to the old record. Either both changes commit or neither
does, so a crash cannot leave the slot halfway replaced.

`--keep-both` is different from replacement. It records an explicit decision that
two values may coexist in the same slot. A bare approval or remember fails when an
active conflict exists, preventing “explicit write” from becoming silent overwrite.

## Stable keys separate conflict policy from semantic search

Conflict detection asks whether two facts occupy the same conceptual slot. Search
asks which records are relevant to a query. Those are different decisions.

Phase 3A detects a conflict by exact local `(memory_type, memory_key)` equality. For
example, both of these values use `preference.response_style`:

```text
I prefer concise answers.
I prefer detailed explanations with examples.
```

They conflict even if an embedding model assigns an unexpectedly low similarity.
Conversely, two semantically close project descriptions with different stable keys
do not trigger an opaque similarity-threshold replacement. Exact slots favor
explainable false negatives over accidental false-positive overwrites. The CLI shows
keys so the user can establish a canonical slot with explicit `remember`.

Semantic similarity still matters for retrieval. It simply does not receive write
or replacement authority.

## Local cost guard and two-stage safety screening

Automatic extraction begins with a deterministic first-person durable-signal
prefilter. “What is RAG?” misses and adds no extraction request. “My goal is to learn
RAG” may hit and proceed. The prefilter is intentionally conservative: a miss does
not mean the fact is unimportant, because the user can always use explicit
`memory remember`.

Safety screening then runs at two points:

1. Before extraction, the raw current message is checked locally. A likely secret,
   identity document, health record, financial record, or privileged legal detail
   stops the additional provider request.
2. After strict JSON parsing, every proposed key and content value is checked again
   before the batch is persisted. This catches unsafe model reformulation. An
   explicit remember passes through the same persisted-content policy.

The complete candidate batch is atomic. One unsafe remaining proposal prevents all
of it from being written. False positives are accepted as the safer trade-off, and
there is no safety override in this phase. Regular expressions and term lists are
defence in depth, not proof that all sensitive data will be recognized.

## A dedicated SQLite lifecycle and hybrid retrieval

The default `.pka/memory.db` database is separate from `.pka/knowledge.db`. Its
schema stores memory state and replacement links, BM25 postings, local E5 vectors,
schema metadata, and one-way rejection fingerprints. Foreign keys and transactions
keep records, postings, and embeddings consistent across restarts and failures.

`pka memory search` considers active records only. It combines mixed-language BM25
lexical ranking with exact cosine ranking over E5 embeddings, then fuses the ordered
IDs with Reciprocal Rank Fusion. For a query such as “What am I learning?”, BM25 can
reward the exact word “learning” while embeddings can retrieve a Chinese or
paraphrased learning-progress record. RRF combines the rankings without pretending
their raw scores share one scale.

Pending candidates are deliberately absent from search because the user has not
authorized them. Superseded records remain available to lifecycle listing and audit
until forgotten, but they are not current facts and therefore are also absent from
search.

## Rejection and forgetting mean different things

Rejecting answers “this automatic proposal was wrong or unwanted.” In one
transaction the store records a one-way type-key-content fingerprint, deletes the
pending content, and cascades deletion to its postings and vector. The fingerprint
prevents the exact rejected proposal from reappearing without retaining its text.

Forgetting answers “remove this memory and its replacement history.” Starting from
any record, the store follows replacement links in both directions and hard-deletes
the complete connected chain, its postings, embeddings, conflict references, and
matching rejection fingerprints. It never reactivates an older superseded value,
and Phase 3A has no undo command.

The distinction matters: rejection remembers only that one proposal should be
suppressed, while forgetting removes the application's retained lifecycle data.

## Provider disclosure, local storage, and erasure limits

Automatic extraction is considered only after a successful Agent answer on a
non-fallback route. If a router provider or parsing error triggers fail-closed
retrieval, extraction is skipped even when the subsequent grounded answer succeeds.
On an eligible non-fallback route with an allowed extraction hit, the configured
provider receives only the current user message plus the extraction instructions. It
does not receive existing memories, knowledge evidence, database paths, timestamps,
hashes, embeddings, or API keys. This additional request does not reduce what the
earlier router or grounded-answer calls already received.

The cost model is explicit:

- On an eligible non-fallback route, a prefilter miss adds zero model calls.
- On that route, an allowed hit adds one logical extraction request after the
  successful answer, although provider SDK retries may create more than one billable
  attempt.
- A fail-closed fallback route adds no extraction request, even if its grounded answer
  succeeds.
- Every explicit memory command makes zero remote model calls. Local writes and
  searches may load E5.

The SQLite memory database is local but not encrypted at rest. Rejection and
forgetting hard-delete application records, but neither can guarantee physical
secure erasure from reusable SQLite pages, filesystem snapshots, operating-system
caches, or provider systems. Host-filesystem protection and a trusted provider are
therefore part of the real security boundary.

## What Phase 3B will add

Phase 3B will retrieve relevant active memories for an Agent request, assign visible
request-local labels such as `M1...Mn`, and inject a separated memory context into
the generation prompt. The answer contract will need to make memory use visible and
keep it distinct from document evidence such as `S1...Sn`.

That future flow introduces questions Phase 3A intentionally avoids: how many
memories to recall, when memory should affect routing or tone, how to prevent a
stored memory from acting as an instruction, and how to validate that a displayed
`M1` reference was actually supplied. No active memory influences an answer until
that design is implemented and tested.

## Commands I can explain

```bash
uv run pka agent "My goal is to build an AI Engineer portfolio."
uv run pka memory list --status pending
uv run pka memory approve mem_candidate
uv run pka memory remember --type preference --key preference.response_style "Concise answers"
uv run pka memory search "How should answers be written?"
uv run pka memory reject mem_candidate
uv run pka memory forget mem_active
```

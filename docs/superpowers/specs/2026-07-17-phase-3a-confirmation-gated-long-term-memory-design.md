# Phase 3A: Confirmation-Gated Long-Term Memory Design

## Context and goal

Phase 2B added a conservative `pka agent` command that visibly chooses between a
safe local direct response and the existing grounded RAG path. The agent still has
no persistent model of the user. Every invocation starts without knowledge of the
user's preferences, goals, projects, or learning progress.

Phase 3 is split into two independently testable increments:

- Phase 3A implements the complete memory lifecycle: explicit creation, automatic
  candidate extraction, confirmation, rejection, replacement, search, listing,
  forgetting, persistence, and audit metadata.
- Phase 3B will retrieve active memories for `pka agent`, inject only relevant
  memory context into generation, and display which memories affected an answer.

This document specifies Phase 3A only. A pending or active memory does not affect an
answer in this phase. That boundary keeps storage and write authority testable before
memory becomes another source of model context.

## Learning objectives

By completing Phase 3A, a contributor should be able to explain:

- why long-term memory is different from conversation history and a document
  knowledge base;
- why extraction, confirmation, persistence, retrieval, and answer personalization
  are separate responsibilities;
- why an LLM may propose a memory but must not receive direct write authority;
- how explicit state transitions make an agent's side effects auditable;
- why a stable memory key is safer for conflict detection than an uncalibrated
  similarity threshold;
- how lexical and vector retrieval can coexist in a dedicated memory store;
- why rejection and forgetting require different retention semantics;
- what personal data is sent to the configured provider and what stays local.

## Approved product decisions

| Concern | Decision | Reason |
| --- | --- | --- |
| Phase boundary | Build lifecycle in 3A and answer integration in 3B | Avoid changing storage, extraction, prompts, citations, and answer parsing at once. |
| Write authority | Automatic extraction creates `pending` candidates only | The user must confirm every automatically proposed memory. |
| Explicit write | `memory remember` creates an `active` memory directly | The command itself is an explicit user decision. |
| Allowed types | `preference`, `goal`, `project`, `learning_progress` | Small learning-focused allowlist with clear future value. |
| Excluded content | Secrets, credentials, identity documents, health, legal, and financial information | These categories are too sensitive for this phase. |
| Extraction trigger | Local first-person prefilter before an LLM call | Avoid an extra remote request for ordinary questions. |
| Candidate workflow | Persist candidates, then approve or reject by ID | Works with the existing one-shot CLI and non-interactive automation. |
| Conflict detection | Local exact `(type, memory_key)` slot matching | Deterministic and does not send existing memories to the extractor. |
| Replacement | User confirms replacement; old record becomes `superseded` | Preserves an audit trail without silent overwrites. |
| Retrieval | Dedicated hybrid memory search over active records | Memory is searchable without pretending it is a document. |
| Storage | Separate `.pka/memory.db` SQLite database | Keeps memory schema and lifecycle independent of the knowledge index. |
| Rejection | Delete candidate content; retain only a one-way fingerprint and time | Avoid storing a rejected false or sensitive statement. |
| Forgetting | Hard-delete the entire connected replacement chain | “Forget” must remove content, vectors, conflicts, and fingerprints rather than hide them. |
| Provider boundary | Reuse `LanguageModel`; extractor sees only the current user message | Keeps the adapter provider-neutral and minimizes disclosure. |
| Answer behavior | Extraction runs only after a successful `pka agent` response | A failed answer must not create a long-term side effect. |

## Alternatives considered

### Store memories as synthetic knowledge documents

The existing knowledge index could store sources such as `memory://<id>`. This
would reuse ingestion and search, but it would force pending approval, replacement,
rejection, and forgetting into a document model that has none of those semantics.
Document citations and user-memory provenance would also become difficult to
distinguish. Phase 3A uses a dedicated boundary instead.

### Store one JSON user profile

A `.pka/profile.json` file would be easy to inspect and edit. It would require
whole-file rewrites, provide weak transactional guarantees, make replacement audit
awkward, and eventually require loading the complete profile for every search.
SQLite provides explicit state, indexes, atomic transitions, and restart-safe tests.

### Automatically write high-confidence memories

Confidence scores from an extractor are not calibrated probabilities. A fluent model
can confidently turn a temporary thought into a false stable fact. Phase 3A treats
every automatic result as a proposal regardless of model confidence.

### Send existing memories to the extractor for conflict detection

The model could choose which old memory a candidate replaces. Existing memories are
not required to extract the new fact, and sending them expands remote disclosure.
Phase 3A instead gives each fact a normalized local slot key and resolves conflicts
inside the application and SQLite boundaries.

## Domain vocabulary

### Enums

```python
class MemoryType(StrEnum):
    PREFERENCE = "preference"
    GOAL = "goal"
    PROJECT = "project"
    LEARNING_PROGRESS = "learning_progress"


class MemoryStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class MemoryActivationAction(StrEnum):
    ACTIVATE = "activate"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"
```

`rejected` is not a persistent memory status. Rejection removes candidate content
and stores a fingerprint in a separate table.

### Stable memory slots

Every memory has a `memory_key` describing the stable subject of the fact. Examples:

- `preference.response_style`
- `goal.ai_engineer_internship`
- `project.personal_knowledge_agent`
- `learning_progress.rag_retrieval`

Keys are lower-case ASCII and must match `[a-z][a-z0-9_.-]{0,63}`. Automatic
extraction supplies a key. Explicit creation requires `--key`; no LLM call is made
to invent one.

Conflicts are exact active records with the same `(memory_type, memory_key)`. This
avoids an arbitrary vector-similarity threshold. Search ranking remains independent
of conflict semantics. An extractor can still choose inconsistent keys for the same
real-world subject, so this design prefers explainable false-negative conflicts over
opaque false-positive replacements. The CLI always displays keys, and users can
choose a canonical key with explicit `remember`.

### Records

```python
@dataclass(frozen=True, slots=True)
class MemoryProposal:
    memory_type: MemoryType
    memory_key: str
    content: str


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    memory_type: MemoryType
    memory_key: str
    content: str
    content_hash: str
    status: MemoryStatus
    supersedes_id: str | None
    source_hash: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    memory: MemoryRecord
    score: float


@dataclass(frozen=True, slots=True)
class MemoryExtractionReport:
    candidates: tuple[MemoryRecord, ...]
    skipped_reason: MemoryExtractionSkipReason | None
```

All timestamps are timezone-aware UTC values. IDs use a project-owned `mem_` prefix
plus a random UUID hex value. Tests inject clocks and ID factories at application or
adapter boundaries rather than depending on wall-clock time.

### Normalization and hashes

Stored content preserves user-visible letter case but applies Unicode NFKC,
collapses whitespace, and strips surrounding whitespace. The normalized comparison
value is additionally case-folded before SHA-256 hashing. A memory key is validated
but not inferred locally.

`content_hash` supports duplicate detection. `source_hash` identifies the user
message that produced an automatic proposal without retaining the original message.
Explicit memories use `source_hash=None`.

A rejected-proposal fingerprint hashes the normalized memory type, key, and
comparison content with unambiguous separators. It therefore suppresses only the
same proposal, rather than suppressing equal text used for a different memory slot.

Content is limited to 500 Unicode code points. The automatic extractor accepts at
most 4,000 Unicode code points from the current message; a longer message produces a
local skip rather than truncation or a provider call. An extraction may contain at
most three candidates. Empty or over-limit candidates invalidate the complete
extractor response; the parser does not partially repair it.

## Protocol boundaries

The domain layer defines replaceable behavior without importing SQLite, an SDK, or
an embedding runtime.

```python
class MemoryExtractor(Protocol):
    def extract(self, message: str) -> tuple[MemoryProposal, ...]: ...


class MemoryStore(Protocol):
    def create_pending(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        source_hash: str,
    ) -> tuple[MemoryRecord, ...]: ...

    def remember(
        self,
        memory_type: MemoryType,
        memory_key: str,
        content: str,
        *,
        action: MemoryActivationAction = MemoryActivationAction.ACTIVATE,
        replace_id: str | None = None,
    ) -> MemoryRecord: ...

    def approve(
        self,
        candidate_id: str,
        *,
        action: MemoryActivationAction,
        replace_id: str | None = None,
    ) -> MemoryRecord: ...

    def reject(self, candidate_id: str) -> None: ...
    def forget(self, memory_id: str) -> int: ...

    def list_memories(
        self,
        *,
        status: MemoryStatus | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]: ...

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
    ) -> list[MemorySearchResult]: ...
```

The `approve` operation is an atomic storage capability because activating a
candidate and superseding an existing record must commit together.

## Components and responsibilities

### First-person memory prefilter

`should_extract_memory(message: str) -> bool` is a local, conservative heuristic.
It detects first-person durable-information signals such as:

- `my goal`, `I prefer`, `I am working on`, `I have completed`;
- `我的目标`, `我更喜欢`, `我正在做`, `我已经学完`.

It does not decide what to store. A miss means no automatic proposal, and the user
can still use `memory remember`. A hit permits the structured extractor call.

### Structured memory extractor

The infrastructure extractor builds one system and one user message. The system
message defines:

- the four allowed types;
- the stable-key format;
- the distinction between durable information and temporary conversation;
- the prohibited sensitive categories;
- a maximum of three candidates;
- exact JSON without code fences.

The user message contains only the current user message. Existing memories,
database paths, source hashes, embeddings, API keys, and answer evidence are absent.

The exact provider contract is:

```json
{
  "candidates": [
    {
      "type": "goal",
      "key": "goal.ai_engineer_internship",
      "content": "The user aims to complete an AI Engineer internship project."
    }
  ]
}
```

An empty candidate list is valid. The local parser rejects invalid JSON, code
fences, duplicate keys, non-object roots, missing or extra fields, wrong Python
types, unknown types, invalid keys, blank or over-limit content, and more than three
candidates. It never repairs output.

### Local memory safety policy

The safety policy runs locally in two places: on the raw current message before an
automatic extractor call, and on every parsed proposal's key and content before
persistence. It also protects explicit `remember` calls. It conservatively rejects
common secret formats and English/Chinese indicators for:

- passwords, API keys, access tokens, and private keys;
- identity documents and government identifiers;
- medical diagnoses, treatment, and health records;
- bank accounts, payment cards, income, debt, and other financial records;
- legal cases and privileged legal details.

The pre-extraction check prevents the memory feature from making an additional
provider call for a message that appears sensitive. The post-extraction check
protects against unsafe reformulation or an unsafe key. The policy is defense in
depth, not a perfect personal-data classifier. False positives are safer than
silently persisting a prohibited category. Phase 3A does not provide an override
flag.

### Memory proposal service

`MemoryProposalService.propose(message)` coordinates:

1. Reject a blank input, or locally skip an input over 4,000 code points, before
   dependencies.
2. Run the local prefilter.
3. If it misses, return a report with no candidates and no provider call.
4. Run the local safety policy on the current message; if it detects a prohibited
   category, return a safe skip with no extraction call.
5. Call the extractor on the current message only.
6. Apply the local safety policy to the complete parsed proposal batch.
7. If every proposal passes, store the complete batch as pending in one transaction.
8. Return created records for visible CLI output.

Duplicate active or pending `(type, key, content_hash)` values are skipped. A
fingerprint in the rejection table also suppresses the exact same rejected proposal.
Duplicate or previously rejected proposals may be filtered before the transaction.
If every candidate is filtered, no pending row is created. If any remaining proposal
is unsafe, the complete remaining batch is rejected and no pending row is created;
the unsafe content is not added to the rejection table.

### Memory administration service

`MemoryAdminService` owns use-case validation and delegates atomic operations to the
store:

- `remember` validates type, key, content, safety, and conflict action, then creates
  an active memory without an LLM call;
- `list` returns ordered records, newest first;
- `search` returns active records only, unless a later phase adds an explicit audit
  search command;
- `approve` resolves a pending candidate;
- `reject` removes pending content and records its fingerprint;
- `forget` hard-deletes a complete replacement chain.

Approval rules are explicit:

- `ACTIVATE` is valid only when no active record occupies the same `(type, key)`;
- `REPLACE` requires `replace_id`, and that record must be active with the same type
  and key;
- `KEEP_BOTH` requires an existing same-slot conflict and preserves both active
  values because the user explicitly chose ambiguity;
- a non-pending candidate or stale replacement target is an error.

Explicit `remember` uses the same activation rules. A bare write is valid only when
the slot has no active record. `--replace` performs the same atomic supersession as
candidate approval, while `--keep-both` records an explicit choice to preserve an
existing same-slot value. This prevents “explicit” from becoming “silent overwrite.”

### SQLite memory store

`SQLiteMemoryStore` owns `.pka/memory.db` and reuses the existing local E5 embedding
adapter, mixed-language tokenizer, exact cosine approach, and RRF helper without
implementing the `KnowledgeIndex` protocol.

Schema version 1 contains:

```text
schema_metadata(version)

memories(
  memory_id primary key,
  memory_type,
  memory_key,
  content,
  content_hash,
  status,
  supersedes_id nullable,
  source_hash nullable,
  token_count,
  created_at,
  updated_at
)

memory_postings(token, memory_id, frequency)
memory_embeddings(memory_id, model_id, dimensions, vector)
memory_rejections(proposal_hash primary key, rejected_at)
```

Indexes cover status/type/key, content hash, postings token, and embedding model ID.
SQLite foreign keys and cascade deletes protect postings and embeddings.

Search filters to active records, computes BM25 lexical results and exact cosine
semantic results, then fuses their ranked lists with RRF. `memory search` does not
expose raw vectors or provider data.

## CLI contract

All memory commands default to `.pka/memory.db` and accept `--memory-db` for an
explicit path.

```bash
pka memory remember \
  --type goal \
  --key goal.ai_engineer_internship \
  "Complete a portfolio-ready Personal Knowledge Agent"

pka memory remember \
  --type goal \
  --key goal.ai_engineer_internship \
  --replace <active-id> \
  "Complete a production-ready Personal Knowledge Agent"

pka memory remember \
  --type goal \
  --key goal.ai_engineer_internship \
  --keep-both \
  "Publish the Personal Knowledge Agent as an open-source project"

pka memory list
pka memory list --status pending
pka memory list --type project
pka memory search "What am I learning?" --limit 5
pka memory approve <candidate-id>
pka memory approve <candidate-id> --replace <active-id>
pka memory approve <candidate-id> --keep-both
pka memory reject <candidate-id>
pka memory forget <memory-id>
```

For both `remember` and `approve`, `--replace` and `--keep-both` are mutually
exclusive. A bare activation reports the exact active conflict IDs when a slot is
occupied.

After a successful `pka agent` answer, a prefilter hit may add output such as:

```text
Memory candidates:
[mem_a1b2] goal goal.ai_engineer_internship
  Complete a portfolio-ready Personal Knowledge Agent.
  Conflict: [mem_old1] Become an AI Engineer intern.
Approve with:
  pka memory approve mem_a1b2 --replace mem_old1
  pka memory approve mem_a1b2 --keep-both
```

Phase 3A adds `--memory-db` to `pka agent`. It does not add memory extraction to
`pka ask`, because `ask` remains the explicit fixed RAG comparison path.

## Agent integration data flow

```text
pka agent question
  -> existing route and response pipeline
  -> successful response is printed
  -> local first-person prefilter
     -> miss: stop, zero extraction calls
     -> hit: local raw-message safety screen
        -> sensitive: stop, zero extraction calls
        -> allowed: structured extraction call
           -> strict parse and safety policy
           -> atomic pending write
           -> local same-slot active lookup
           -> display candidate and conflict commands
```

Extraction happens after the answer. Pending candidates do not affect that answer or
future answers in Phase 3A. Only Phase 3B will retrieve active memories for response
generation.

## Error handling

- Blank messages, invalid types or keys, invalid limits, and invalid activation option
  combinations fail before model or storage calls. An automatic extraction input over
  4,000 code points is skipped locally without affecting the already printed answer.
- A raw-message safety-policy hit skips extraction without a provider or storage call.
- A provider failure during extraction leaves the successful answer intact and
  prints `Memory extraction skipped: provider error`; expected extraction skips keep
  the successful agent command's zero exit status.
- Invalid structured output leaves the answer intact and prints
  `Memory extraction skipped: invalid extractor output`.
- No partial candidate batch is persisted after a parser, safety-policy, embedding,
  or SQLite failure.
- SQLite and embedding failures leave the answer intact but print a concise memory
  error category. Explicit `memory` commands return a non-zero CLI result.
- Unknown IDs, stale candidate status, wrong replacement slots, and conflicting bare
  activation produce safe, actionable errors.
- Rejection is valid only for a pending candidate.
- Forgetting an unknown ID is an error rather than a false success.
- Unexpected programming exceptions are not converted into normal memory skips.

Provider bodies, complete prompts, raw user messages, API keys, embeddings, and
stack traces are not included in user-facing errors.

## Rejection and forgetting semantics

Rejecting a candidate performs one transaction:

1. insert or update its type-key-content `proposal_hash` and rejection time;
2. delete the pending memory row;
3. cascade-delete its postings and embedding.

Forgetting any memory follows both directions of the `supersedes_id` relationship to
find the connected replacement chain. One transaction deletes every record in the
chain, their postings, embeddings, conflict references, and matching rejection
fingerprints. It never reactivates a superseded record.

The CLI reports the number of records permanently removed. There is no undo command
in Phase 3A.

## Privacy and security boundary

The memory database is local but not encrypted at rest. Users must protect the host
filesystem. Encryption, multiple user profiles, synchronization, and remote backup
are out of scope.

On an automatic extraction hit that passes the local safety screen, the configured
provider receives the current user message. It does not receive existing memories,
knowledge-base evidence, local paths, timestamps, content hashes, or embeddings.
This boundary describes the additional memory-extraction call; it does not alter
provider disclosure in the existing routing and answer pipeline. A user who does
not want the message sent for extraction can avoid `pka agent` for that message and
use explicit local `memory remember`, which makes no LLM request.

Rejected and forgotten content is hard-deleted from the application database, but
the project cannot guarantee physical secure erasure from SQLite pages, filesystem
snapshots, operating-system caches, or provider systems. Documentation must state
this limitation honestly.

## Testing strategy

Automated tests stay offline and deterministic.

### Unit tests

- enum values, frozen records, UTC timestamps, ID/key/content invariants;
- Unicode normalization and stable content/source fingerprints;
- first-person prefilter hits and ordinary-question misses in Chinese and English;
- extraction prompt allowlist, exclusions, exact schema, and data minimization;
- parser acceptance of zero to three valid candidates;
- parser rejection of malformed JSON, fences, duplicates, extra fields, wrong
  types, unknown types, invalid keys, blank content, long content, and four or more
  candidates;
- pre- and post-extraction safety-policy coverage for secrets, identity, health,
  legal, and financial text, including zero provider calls on a raw-message hit;
- proposal-service prefilter short-circuit, batch atomicity, duplicate suppression,
  extraction failures, and safe reports;
- administration validation for activate, replace, keep-both, reject, and forget,
  including explicit-write conflicts;

### SQLite integration tests

- schema creation and version validation;
- active, pending, and superseded persistence across adapter restarts;
- posting and vector persistence with the configured embedding model ID;
- lexical, semantic, and hybrid active-only search;
- exact slot conflict lookup;
- atomic approval and replacement;
- rejection content removal and fingerprint persistence;
- complete replacement-chain deletion with cascades;
- rollback after injected storage failures;
- safe mapping of corrupt rows and vector blobs.

### CLI tests

- parser defaults, type/key/limit boundaries, and mutually exclusive activation
  flags on both `remember` and `approve`;
- explicit remember/list/search without a language-model call;
- automatic prefilter miss with zero extraction calls;
- valid pending-candidate display and conflict commands;
- provider, parser, safety, embedding, and storage failures after a successful answer;
- approval, replacement, keep-both, rejection, and hard forgetting across processes;
- no traceback, API key, raw provider body, or embedding leakage;
- unchanged `pka ask` and existing `pka agent` route/answer output before the optional
  memory section.

The full quality gate remains:

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
```

## Cost model

- Ordinary `pka agent` request without a first-person signal: unchanged Phase 2B
  model-call count.
- Prefilter hit: one additional extraction call after a successful answer.
- Explicit `memory remember/list/search/approve/reject/forget`: zero remote model
  calls; local E5 may load for writes or search.
- Provider SDK retries can multiply an extraction request, so the CLI documents that
  the prefilter is a cost guard rather than a billing guarantee.

## Out of scope

Phase 3A does not implement:

- conversation history or multi-turn sessions;
- automatic activation without confirmation;
- memory-aware answer generation or memory citations;
- sending active memories to the route model or answer model;
- expiration, decay, importance scoring, consolidation, or summarization;
- background jobs or asynchronous extraction;
- encrypted storage, secure physical erasure, multi-user isolation, or cloud sync;
- PDF, web, or code ingestion;
- knowledge graphs;
- routing, extraction, retrieval, or answer-quality evaluation datasets.

Phase 3B will design memory-aware generation separately, including request-local
`M1...Mn` labels, prompt separation from document evidence, visible memory use, and
strict validation. Evaluation remains a later project phase.

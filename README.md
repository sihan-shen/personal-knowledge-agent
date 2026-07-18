# Personal Knowledge Agent

A personal knowledge-management system built incrementally with RAG and Agent patterns.

## Current scope

Phase 3A provides persistent Markdown ingestion, Chinese/English lexical, semantic,
and hybrid retrieval, evidence-grounded answers through an OpenAI-compatible Chat
Completions endpoint, a conservative Agent router, and confirmation-gated long-term
memory. Embeddings run locally on CPU and are stored in SQLite alongside each BM25
index. `pka ask` remains a fixed RAG pipeline that always retrieves; `pka agent`
visibly chooses a safe local direct response or the same grounded retrieval path,
then may propose a memory after a successful response.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --all-groups
```

The dependency lock selects the CPU-only PyTorch build. The first ingest command
downloads `intfloat/multilingual-e5-small` from Hugging Face; later commands reuse
the default Hugging Face cache (normally `~/.cache/huggingface/hub`). Model
inference is explicitly configured for CPU, so a GPU is not required. Keep the
machine online for the first model download.

### WSL environment

This repository's current WSL location is:

```bash
/mnt/c/Users/19850/Projects/RAG-Agent
```

Run project commands from that directory (not the previous native-Linux path
under `/home/sihan/文档/...`). On this Windows-mounted filesystem, keep Python
temporary files in WSL's Linux filesystem and ask uv to copy cached packages:

```bash
export TMPDIR=/tmp
export UV_LINK_MODE=copy
```

These exports are already configured in `~/.bashrc`. Open a new shell or run
`source ~/.bashrc` after changing that file.

## Health check

```bash
uv run pka health
```

## Configure answer generation

`pka ask` and `pka agent` require an API key in the shell. The defaults target
DeepSeek's OpenAI-compatible endpoint and the `deepseek-v4-flash` model:

```bash
export PKA_LLM_API_KEY="your-key"
export PKA_LLM_BASE_URL="https://api.deepseek.com"  # optional
export PKA_LLM_MODEL="deepseek-v4-flash"            # optional
```

`.env.example` documents these names, but the CLI does not automatically load a
`.env` file. Never commit the real key. The endpoint and model are configurable so
the application service remains independent of one provider.

## Build a local knowledge base

By default, the CLI stores its SQLite index in `.pka/knowledge.db` relative to the
current directory:

```bash
uv run pka ingest notes/rag.md
uv run pka search "RAG 如何检索 knowledge"
```

Search defaults to hybrid retrieval. Select a mode explicitly when comparing
retrieval behavior:

```bash
uv run pka search "exact keyword" --mode lexical
uv run pka search "a related idea using different words" --mode semantic
uv run pka search "RAG 如何减少幻觉" --mode hybrid
```

- `lexical` ranks exact Chinese/English token matches with BM25 and does not run
  the embedding model for the query.
- `semantic` embeds the query and exact-scans the stored vectors by cosine
  similarity.
- `hybrid` combines the BM25 and semantic rankings with Reciprocal Rank Fusion
  (RRF). It is the default because either retriever may contribute a useful
  candidate.

E5 uses different inputs for the two sides of retrieval: indexed chunk text is
encoded as `passage: <heading and body>`, while user text is encoded as
`query: <query>`. The adapter adds these prefixes automatically.

Use `--db` to select an explicit knowledge base and `--limit` to return between
1 and 20 chunks:

```bash
uv run pka ingest notes/rag.md --db data/knowledge.db
uv run pka search "retrieval context" --limit 3 --db data/knowledge.db
```

Ingesting the same resolved source path replaces its previous chunks atomically.
The same transaction replaces postings and float32 embedding vectors. A Phase 1A
database is migrated automatically, but its existing documents must be reingested
before semantic retrieval can search them. Search output labels the mode and
contains retrieval scores, source paths, heading paths, and text snippets. It
returns evidence chunks rather than an LLM-written answer.

## Ask an evidence-grounded question

After ingesting a document, ask a question against the same database:

```bash
uv run pka ask \
  "RAG 为什么能减少模型脱离资料的猜测？" \
  --mode hybrid \
  --limit 5 \
  --db data/knowledge.db
```

`ask` defaults to hybrid retrieval and five evidence chunks. Its evidence limit is
1 to 10, a smaller boundary than `search` because every selected chunk consumes
remote-model context and can add API cost. An accepted answer contains inline
citations such as `[S1]` and a `Sources` section mapping only those cited labels
back to local paths and heading paths. If retrieval returns nothing, the CLI refuses
locally without calling the model. The model can also declare that the retrieved
evidence is insufficient; this produces no `Sources` section.

The provider receives the question plus each selected chunk's request-local citation
ID, heading path, and text. Absolute source paths, stable chunk IDs, retrieval scores,
the API key, and the embedding vectors remain local. Retrieved text can still be
private, so configure only a provider you trust and understand its data policy before
using `ask`.

## Let the Agent choose whether to retrieve

Use `agent` when you want the routing decision to be part of the workflow:

```bash
uv run pka agent "你好" --db data/knowledge.db
uv run pka agent "RAG 为什么需要检索？" --db data/knowledge.db
```

The route is always printed before the response:

```text
Route: direct (greeting, rule)
你好。我可以在需要时检索你的本地知识库，并用引用回答问题。
```

```text
Route: retrieve (llm)
Answer:
...
```

Direct responses are deliberately limited to whole-message greetings, capability
questions, and thanks. Factual, learning, personal-knowledge, mixed, ambiguous, and
uncertain requests retrieve. For example, `你好，请解释 RAG` retrieves because the
greeting is combined with a knowledge request. Phase 2B does not allow uncited
direct factual answers.

The hybrid router uses exact local rules first, then asks the configured model for
one strict two-field JSON decision. A rule match makes no model request. An LLM
direct decision makes one routing request and renders a local template. A normal
retrieval answer usually makes two model calls—one to route and one to generate the
grounded answer—although empty retrieval returns a local refusal after the routing
call. Provider or invalid-JSON routing failures are displayed as safe fallback
categories and fail closed to retrieval; raw provider details are not printed.

Only the question is sent during LLM routing. Retrieved chunks are sent later only
when the chosen route is `retrieve`. `pka ask` remains available when you explicitly
want retrieval without a routing call.

The adapter requests one non-streaming JSON object through Chat Completions. JSON
mode only helps the provider produce syntactically valid JSON; the project still
validates the exact fields, answer status, citation IDs, body citation order, and
membership in the supplied evidence. It does not repair malformed responses.
Retrieved documents are serialized as untrusted evidence data and the system prompt
tells the model not to follow instructions inside them. That boundary, data
minimization, and local citation validation are defence in depth, not a mathematical
guarantee against prompt injection.

The SDK may retry transient failures up to two times, so one CLI command can make
more than one billable request. Authentication, permission, bad-request, rate-limit,
timeout, connection, server, truncation, schema, and citation failures become concise
CLI errors without echoing provider payloads. Automated tests use deterministic
fakes and never call DeepSeek or incur API cost.

## Manage confirmation-gated long-term memory

Phase 3A stores long-term memory in a dedicated SQLite database. The default is
`.pka/memory.db`, separate from the knowledge index. Every memory has one of four
types: `preference`, `goal`, `project`, or `learning_progress`. Use `--memory-db` on
`pka agent` or any explicit memory command to select a different database:

```bash
uv run pka agent \
  "My goal is to finish an AI Engineer portfolio project." \
  --db data/knowledge.db \
  --memory-db data/memory.db
```

Keys are explicit lower-case ASCII slots matching
`[a-z][a-z0-9_.-]{0,63}`, and stored content is limited to 500 Unicode code points.
Automatic extraction accepts at most 4,000 code points from the current message and
at most three candidates; an over-limit message is skipped locally instead of being
truncated or sent for extraction.

After a successful Agent response, a local first-person prefilter may allow one
structured extraction request. The model can only propose candidates; it cannot
activate them. Proposed records remain `pending` and do not appear in memory search
until the user explicitly approves them:

```bash
uv run pka memory list --status pending
uv run pka memory approve mem_candidate
```

An explicit `remember` command is itself the user's authorization, so it creates an
`active` memory directly. The stable key names the fact's local slot:

```bash
uv run pka memory remember \
  --type goal \
  --key goal.ai_engineer_internship \
  "Complete a portfolio-ready Personal Knowledge Agent"
```

If that type-and-key slot already has an active record, a bare approval or remember
fails and prints the conflicting IDs. Choose the conflict outcome explicitly:

```bash
# Atomically supersede one existing active value.
uv run pka memory remember \
  --type goal \
  --key goal.ai_engineer_internship \
  --replace mem_old \
  "Complete a production-ready Personal Knowledge Agent"

# Deliberately preserve both active values in the same slot.
uv run pka memory remember \
  --type goal \
  --key goal.ai_engineer_internship \
  --keep-both \
  "Publish the Personal Knowledge Agent as an open-source project"

# The same two choices apply when approving an automatic candidate.
uv run pka memory approve mem_candidate --replace mem_old
uv run pka memory approve mem_candidate --keep-both
```

List records newest first, optionally filtering their lifecycle status or type.
Search uses local BM25 and E5 cosine rankings fused with RRF, returns `active`
records only, accepts a limit from 1 to 20, and can filter by type:

```bash
uv run pka memory list
uv run pka memory list --status superseded
uv run pka memory list --type project
uv run pka memory search "What am I learning?" --limit 5
uv run pka memory search "current projects" --type project
```

Rejecting removes a pending candidate's content and keeps only a one-way fingerprint
so the same proposal is not recreated. Forgetting any record hard-deletes its entire
connected replacement chain, including postings, vectors, and matching rejection
fingerprints; there is no undo command:

```bash
uv run pka memory reject mem_candidate
uv run pka memory forget mem_active
```

### Phase boundary, provider disclosure, and cost

Phase 3A implements the complete memory lifecycle but does **not** retrieve active
memories for routing or answer generation. Pending, active, and superseded records
do not influence `pka agent` or `pka ask` answers yet. Phase 3B will design active
memory recall, prompt injection, and visible memory use separately.

Automatic extraction is considered only after a successful `pka agent` answer on a
non-fallback route. If a router provider or parsing error triggers fail-closed
retrieval, extraction is skipped even when the subsequent grounded answer succeeds.
On an eligible non-fallback route, a local prefilter miss adds zero model calls. If
the prefilter hits and the raw message passes the local safety screen, the configured
provider receives the current user message in one additional logical extraction
request. Existing memories, knowledge evidence, local paths, hashes, timestamps,
embeddings, and credentials are not included in that request. Provider SDK retries
can turn one logical extraction request into multiple billable attempts, so the
prefilter is a cost guard rather than a billing guarantee.

Every explicit `pka memory remember/list/search/approve/reject/forget` command makes
zero remote model calls. Local writes and searches may load the E5 embedding model.
The local safety policy screens the raw message before automatic extraction and
screens each proposed key and content again before persistence; explicit remembers
receive the same persisted-content screen. This is conservative defence in depth,
not a complete personal-data classifier, and Phase 3A has no override flag.

The memory database is local but unencrypted. Protect the host filesystem and do not
store secrets, identity documents, health, financial, or privileged legal data.
Reject and forget hard-delete application records, but the project cannot guarantee
physical secure erasure from SQLite pages, filesystem snapshots, operating-system
caches, or provider systems.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

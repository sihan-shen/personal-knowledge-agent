# Personal Knowledge Agent

A personal knowledge-management system built incrementally with RAG and Agent patterns.

## Current scope

Phase 4B provides persistent Markdown ingestion, Chinese/English lexical, semantic,
and hybrid retrieval, evidence-grounded answers through an OpenAI-compatible Chat
Completions endpoint, a conservative Agent router, and confirmation-gated long-term
memory with conservative request-local recall. Embeddings run locally on CPU and
are stored in SQLite alongside each BM25 index. `pka ask` remains a fixed
document-only RAG pipeline; `pka agent` visibly chooses a safe local direct response
or a retrieval path that may combine document evidence with confirmed active
memories, then may propose a new memory after a successful response.

Phase 4B also provides a separate, inspectable knowledge graph and an explicit
graph-grounded answer command. The graph is built only on request from chunks already
stored in the Markdown knowledge index. `pka graph ask` is graph-only: it does not
open `.pka/knowledge.db` or use normal document RAG, the Agent, long-term memory, E5
embeddings, LLM query planning, or a fallback to another answer path. The existing
`ask`, `agent`, routing, and long-term-memory workflows remain separate.

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

## Evaluate document RAG

Run the packaged bilingual evaluation suite with the same answer-provider
configuration required by `pka ask` (including `PKA_LLM_API_KEY`):

```bash
uv run pka evaluate \
  --mode hybrid \
  --limit 5 \
  --output .pka/reports/xquad-mini-v1.json
```

The command rebuilds its fixture index at
`.pka/evaluations/xquad-mini-v1/knowledge.db` and writes the requested JSON report
to `.pka/reports/xquad-mini-v1.json`. It deliberately has no `--db` option: it never
reads or modifies the private `.pka/knowledge.db` knowledge base. The packaged
24-case English/Chinese `xquad-mini v1` fixture is a CC BY-SA 4.0 derivative of
[XQuAD](src/personal_knowledge_agent/evaluation_suites/xquad_mini/ATTRIBUTION.md);
the attribution records the pinned upstream revision and integrity checks.

The CLI prints macro averages for four metrics: answer exact match, answer token F1,
citation precision, and citation recall. It also prints English and Chinese token F1
to make a language imbalance visible. Exact match and extractive token F1 compare a
generated answer with reference answer strings after deterministic normalization.
They are useful regression signals, but do not measure semantic equivalence,
helpfulness, completeness, style, or whether an otherwise correct paraphrase was
preferred. Citation precision and recall measure whether the answer cited the case's
gold fixture document; they do not establish that a citation fully supports every
claim. Live generation can vary by model/provider even though the local fixture,
ingestion, scoring, and report formatting are deterministic.

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
grounded answer. Local memory recall adds no remote request. If both document and
memory context are empty, the Agent refuses locally after the routing call. Provider
or invalid-JSON routing failures are displayed as safe fallback categories and fail
closed to retrieval; raw provider details are not printed.

Only the question is sent during LLM routing. Retrieved chunks are sent later only
when the chosen route is `retrieve`. `pka ask` remains available when you explicitly
want retrieval without a routing call.

### Use active memory in Agent answers

On a normal non-fallback `retrieve` route, the Agent searches the selected
`--memory-db` for confirmed `active` memories. Direct routes, fail-closed fallback
routes, and `pka ask` do not construct or query the memory store for answering.
Pending and superseded records are never eligible.

Recall is deliberately stricter than ranking. It inspects at most 12 active
candidates, then accepts a slot only when it has meaningful Chinese/English token
overlap with the query or cosine similarity of at least `0.84`. At most three memory
items enter one request. All active values under the same `(type, key)` slot are
handled atomically: a keep-both slot is either included completely or omitted. A
slot with more than three active values is omitted with a safe ambiguity warning;
one unsafe peer omits the complete slot.

A memory-only answer can look like:

```text
Route: retrieve (llm)
Answer:
You prefer concise answers. [M1]

Memories used:
[M1] preference.answer_style — Prefer concise answers.
```

A mixed answer keeps document and memory citations separate:

```text
Route: retrieve (llm)
Answer:
Use retrieval evidence for your portfolio project. [S1] [M1]

Sources:
[S1] /local/notes/rag.md > RAG

Memories used:
[M1] goal.rag_project — Build a portfolio-ready RAG project.
```

Only cited context is displayed, and empty `Sources` or `Memories used` sections are
omitted. Recall failures degrade to document-only RAG with fixed labels such as
`embedding error`, `storage error`, `unsafe stored memory`, or
`ambiguous memory slot`;
private exception details are not printed. With neither documents nor accepted
memories, the Agent returns a local insufficient-evidence response without an answer
model call.

Document evidence and personal memory have different authority. `S` context may
support external factual claims; `M` context may support only user-specific
preferences, goals, project state, and learning progress. Both arrays are serialized
as untrusted data. The provider-visible memory fields are exactly the request-local
`citation_id` and normalized `content`; stable memory IDs, type/key, hashes,
timestamps, scores, vectors, database paths, and credentials remain local. The full
type/key/content line under `Memories used` is rendered locally after strict citation
validation.

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

## Build and inspect a knowledge graph

Build one already-indexed Markdown source explicitly. This sends the current heading
path and full text of each indexed chunk to the configured LLM once; provider retries
can make more billable attempts. The source path, stable chunk/document IDs, graph
database path, scores, memories, vectors, and credentials remain local.

```bash
uv run pka graph build notes/rag.md
uv run pka graph build notes/rag.md \
  --db data/knowledge.db \
  --graph-db data/graph.db
```

The graph uses a deliberately small ontology: entity types are `concept`,
`technology`, `method`, `project`, and `resource`; relation types are `is_a`,
`part_of`, `uses`, `depends_on`, `implements`, and `related_to`. Entity identity is
exact `(type, NFKC/casefold/whitespace-normalized name)`, not semantic similarity.
Every retained relation has an exact quote from one chunk plus its local source,
heading, and chunk provenance.

Build output reports only the selected document's contribution:

```text
Built graph for /absolute/path/notes/rag.md
Chunks processed: 6
Entities contributed: 14
Relations contributed: 18
Evidence records: 21
```

Search and inspect the graph locally; these commands do not load E5, LLM settings,
or a provider client:

```bash
uv run pka graph search "vector retrieval" --limit 10
uv run pka graph show ent_ab12... --graph-db .pka/graph.db
```

`graph search` uses local mixed Chinese/English BM25 over entity names. `graph show`
prints one hop of incoming and outgoing relations and every evidence quote with its
source and heading. A missing or empty graph is reported safely and is not created by
a query command.

### Ask from graph relationship evidence

Use the explicit graph-only command when you want an answer grounded in graph
relations rather than document chunks:

```bash
uv run pka graph ask "RAG 使用什么技术？"
uv run pka graph ask "RAG 使用什么技术？" --limit 3
uv run pka graph ask "RAG 使用什么技术？" --graph-db data/graph.db
```

The whole question is used as a local mixed Chinese/English BM25 entity search. It
selects between 1 and 5 entity seeds (default 3), expands only each seed's one-hop
incoming and outgoing relationships, and sends at most 12 deduplicated
relation-plus-provenance evidence records to the provider. Each usable fact has a
`[G#]` citation, and the `Graph evidence` section renders exactly the cited directed
relation plus its local source, heading, and exact provenance quote. This makes an
incoming relation visible in its original direction rather than reversing its claim.

If no seed yields relationship evidence, `pka graph ask` refuses locally with zero
model calls; it does not require API configuration in that case. The language model
is constructed lazily only after local evidence exists. When there is evidence, the
configured provider receives the question and only each record's request-local
citation ID, source/target entity types and names, relation type, heading path, and
evidence quote. Local source paths, entity and relation IDs, chunk IDs and positions,
graph-database path, search scores, credentials, and other storage metadata are not
sent. Evidence quotes can still be private, so use a provider whose data policy you
trust.

Graph answers use strict structured output validation: an answered response must use
only supplied `[G#]` citations, and an insufficient-evidence response cannot cite
anything. This command has no document-RAG, knowledge-database, Agent, memory, or
alternate-answer fallback.

The default graph database is `.pka/graph.db`, separate from `.pka/knowledge.db` and
`.pka/memory.db`. Reingesting a document does not rebuild its graph automatically:
the graph remains its last successful snapshot until you explicitly run `graph build`
again. SQLite storage is local and unencrypted. Replacement deletes application rows,
but cannot guarantee secure erasure from SQLite pages, filesystem snapshots, caches,
or provider systems.

Phase 4B intentionally excludes hybrid document-and-graph context, Agent routing to
the graph, LLM query planning, multi-hop traversal, aliases or fuzzy entity merging,
graph editing, PDF/web/code ingestion, visualization, and external graph services.

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

### Provider disclosure, cost, and storage limits

Phase 3B adds active-memory recall only to eligible Agent retrieval answers. It does
not add chat history, memory-influenced routing, automatic approval, or memory use in
`pka ask`. Retrieval is request-local and does not rewrite stored records.

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
Eligible Agent recall also makes zero remote calls, but it may initialize SQLite and
load E5 before the existing answer request. An allowed Phase 3A extraction remains
one additional logical provider request after the validated answer; SDK retries can
still increase the billable request count.

The local safety policy screens the raw message before automatic extraction and
screens each proposed key and content again before persistence; explicit remembers
receive the same persisted-content screen. This is conservative defence in depth,
not a complete personal-data classifier, and Phase 3A has no override flag.

The memory database is local but unencrypted. Protect the host filesystem and do not
store secrets, identity documents, health, financial, or privileged legal data.
Reject and forget hard-delete application records, but the project cannot guarantee
physical secure erasure from SQLite pages, filesystem snapshots, operating-system
caches, or provider systems. A recalled memory is transmitted to the configured
external answer provider, whose retention and training policies are outside this
project's control.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

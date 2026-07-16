# Phase 2A Grounded Answer Generation Implementation Plan

**Goal:** Add `pka ask`, which retrieves local evidence, sends only the minimum
necessary evidence to a DeepSeek-compatible Chat Completions endpoint, validates a
structured JSON answer locally, and displays citations mapped back to local sources.

**Architecture:** Domain records and a `LanguageModel` protocol isolate application
logic from provider SDK types. `AnswerService` coordinates the existing
`SearchService`, pure evidence and prompt builders, a provider-neutral model call,
and a strict structured-answer parser. The first real adapter uses the OpenAI Python
SDK against a configurable OpenAI-compatible Chat Completions endpoint; deterministic
fakes keep all automated tests offline.

**Tech stack:** Python 3.12+, dataclasses, standard-library JSON and regular
expressions, OpenAI Python SDK, DeepSeek Chat Completions, pytest, Ruff, strict mypy,
SQLite, BM25, multilingual E5, and RRF.

**Approved design:**
`docs/superpowers/specs/2026-07-16-phase-2a-grounded-answer-generation-design.md`

> This WSL repository is mounted under `/mnt/c`. Use `TMPDIR=/tmp`,
> `UV_CACHE_DIR=/tmp/uv-cache`, and `UV_LINK_MODE=copy`. A Linux-native project
> environment such as `UV_PROJECT_ENVIRONMENT=/tmp/pka-phase2a-venv` avoids the
> severe small-file cost of installing PyTorch into the Windows-mounted worktree.

## Development discipline

For every task:

1. Review the named learning checkpoint before editing code.
2. Write the smallest behavioral test that expresses the next requirement.
3. Run it and confirm it fails for the intended reason, not for an import typo.
4. Implement only enough production code to make that behavior pass.
5. Run targeted tests, Ruff, and strict mypy before committing.
6. Keep provider objects, secrets, generated databases, and model artifacts out of
   Git.

Do not begin Phase 2B routing, Phase 2C tools, memory, streaming, or new ingestion
formats while executing this plan.

## File map

- Modify `pyproject.toml`: add the OpenAI Python SDK only when the adapter boundary
  has failing tests.
- Modify `uv.lock`: lock the new SDK and its transitive dependencies.
- Modify `.env.example`: document `PKA_LLM_API_KEY`, `PKA_LLM_BASE_URL`, and
  `PKA_LLM_MODEL` without placing a real secret in the file.
- Modify `src/personal_knowledge_agent/domain/errors.py`: add configuration,
  provider, truncation, structured-schema, and citation failures.
- Modify `src/personal_knowledge_agent/domain/models.py`: add messages, evidence,
  provider output, answer status, and grounded-answer records.
- Modify `src/personal_knowledge_agent/domain/protocols.py`: add `LanguageModel`.
- Create `src/personal_knowledge_agent/application/answer.py`: orchestrate retrieval
  and generation without SDK or prompt details.
- Create `src/personal_knowledge_agent/infrastructure/evidence_builder.py`: assign
  local `S1...Sn` labels in retrieval order.
- Create `src/personal_knowledge_agent/infrastructure/grounded_prompt_builder.py`:
  serialize the question, safe rules, examples, and minimum evidence JSON.
- Create
  `src/personal_knowledge_agent/infrastructure/structured_answer_parser.py`: parse
  the exact JSON contract and validate body citations.
- Create
  `src/personal_knowledge_agent/infrastructure/openai_compatible_chat_model.py`:
  implement non-streaming Chat Completions with JSON output and safe error mapping.
- Create `src/personal_knowledge_agent/infrastructure/llm_settings.py`: load an
  explicit environment mapping into validated settings.
- Modify `src/personal_knowledge_agent/cli.py`: add `pka ask`, compose dependencies,
  and print answers and cited local sources.
- Modify `tests/helpers.py`: add typed fakes for the model and, where useful, index
  behavior.
- Modify `tests/unit/test_models.py`: cover new domain invariants.
- Create `tests/unit/test_evidence_builder.py`.
- Create `tests/unit/test_grounded_prompt_builder.py`.
- Create `tests/unit/test_structured_answer_parser.py`.
- Create `tests/unit/test_answer_service.py`.
- Create `tests/unit/test_openai_compatible_chat_model.py`.
- Create `tests/unit/test_llm_settings.py`.
- Modify `tests/test_cli.py`: cover `ask` without network access.
- Modify `README.md`: explain configuration, privacy boundary, and `ask` usage.
- Create `docs/learning-log/phase-2a.md`: explain grounded generation and the
  boundary to Agent routing.

## Task 1: Define grounded-answer domain contracts and a deterministic fake

**Learning checkpoint**

Understand why provider SDK response objects cannot be domain models. The application
needs stable concepts—message, evidence, model output, and grounded answer—even when
the provider, endpoint, or SDK changes. Decide which invariants belong in record
constructors and which require the parser's access to the current evidence set.

**Files:**

- Modify `src/personal_knowledge_agent/domain/errors.py`
- Modify `src/personal_knowledge_agent/domain/models.py`
- Modify `src/personal_knowledge_agent/domain/protocols.py`
- Modify `tests/helpers.py`
- Modify `tests/unit/test_models.py`
- Modify `tests/unit/test_helpers.py`

### RED: express the domain vocabulary

1. Add tests for stable enum values:
   - `MessageRole`: `system`, `user`, `assistant`;
   - `AnswerStatus`: `answered`, `insufficient_evidence`.
2. Add tests that `ChatMessage` is frozen and rejects blank content.
3. Add tests that `Evidence`:
   - preserves local `chunk_id`, `source`, heading path, text, and score;
   - rejects a blank citation ID, chunk ID, source, or text;
   - rejects non-finite retrieval scores.
4. Add tests that `ModelOutput` rejects blank model ID, content, or finish reason.
5. Add tests that `GroundedAnswer` is frozen, preserves all supplied evidence, and
   rejects a blank answer. Keep evidence-relative citation checks out of this
   constructor; they belong to the parser.
6. Extend `tests/helpers.py` with `FakeLanguageModel` that:
   - is initialized with a `ModelOutput` or configured exception;
   - records each immutable message sequence;
   - returns or raises deterministically;
   - never imports an SDK or reads environment variables.
7. Prove the fake satisfies the structural `LanguageModel` protocol through a typed
   consumer or mypy rather than a runtime `isinstance` check.

Run and confirm the intended failures:

```bash
uv run pytest tests/unit/test_models.py tests/unit/test_helpers.py -v
```

### GREEN: add the minimum contracts

1. Add under `KnowledgeError`:
   - `ConfigurationError`;
   - `LanguageModelError`;
   - `ModelOutputTruncatedError(LanguageModelError)`;
   - `StructuredOutputError`;
   - `InvalidAnswerSchemaError(StructuredOutputError)`;
   - `InvalidCitationError(StructuredOutputError)`.
2. Add the two enums and four dataclasses exactly as approved by the design.
3. Reuse `_require_text`; use `math.isfinite` for the retrieval score.
4. Add `LanguageModel.generate(messages: Sequence[ChatMessage]) -> ModelOutput` to
   `domain/protocols.py`.
5. Implement the fake and keep all new types explicit under strict mypy.

Verify:

```bash
uv run pytest tests/unit/test_models.py tests/unit/test_helpers.py -v
uv run ruff check src tests
uv run mypy src
```

Commit:

```text
feat: define grounded answer contracts
```

## Task 2: Build ranked evidence and a prompt-injection-aware message pair

**Learning checkpoint**

Review prompt injection as an untrusted-data problem. A system instruction alone is
not a complete defence: retrieved documents must remain serialized data, unnecessary
metadata must stay local, and returned citations must be checked later. Be able to
explain why `json.dumps` preserves boundaries better than custom text delimiters but
does not make the model perfectly secure.

**Files:**

- Create `src/personal_knowledge_agent/infrastructure/evidence_builder.py`
- Create `src/personal_knowledge_agent/infrastructure/grounded_prompt_builder.py`
- Create `tests/unit/test_evidence_builder.py`
- Create `tests/unit/test_grounded_prompt_builder.py`

### RED: specify evidence labelling

1. Construct ranked `SearchResult` fixtures with distinct chunk IDs, sources,
   headings, text, and scores.
2. Assert `build_evidence`:
   - returns a tuple;
   - assigns `S1`, `S2`, and `S3` in input rank order;
   - preserves all local metadata without reranking or filtering;
   - returns an empty tuple for no results.
3. Run the evidence test and confirm the missing module/function failure.

### GREEN: implement the pure evidence builder

1. Enumerate results from 1.
2. Construct validated `Evidence` records.
3. Do not deduplicate, cap, or normalize scores.

### RED: specify prompt structure and data minimization

1. Assert `build_grounded_messages` returns exactly one system and one user message.
2. Assert the system message contains:
   - the evidence-only rule;
   - the untrusted-data rule;
   - the exact three JSON fields;
   - one answered example and one refusal example;
   - an explicit instruction to output JSON without a code fence.
3. Put quotes, newlines, `EVIDENCE_JSON:`, fake role declarations, and “ignore prior
   rules” inside evidence text. Extract the JSON region and prove `json.loads`
   reconstructs the original text exactly.
4. Use unique sentinel values for local source, chunk ID, and score; assert they do
   not appear in either outgoing message.
5. Assert outgoing evidence contains only `citation_id`, `heading_path`, and `text`.
6. Assert blank questions and empty evidence are rejected before message creation.

### GREEN: construct the approved messages

1. Keep the stable system policy in a named module constant.
2. Serialize provider-visible evidence with `json.dumps(..., ensure_ascii=False)`.
3. Build the labelled question and evidence regions without logging them.
4. Return immutable project `ChatMessage` records.

Verify:

```bash
uv run pytest \
  tests/unit/test_evidence_builder.py \
  tests/unit/test_grounded_prompt_builder.py -v
uv run ruff check src tests
uv run mypy src
```

Commit:

```text
feat: build grounded evidence prompts
```

## Task 3: Parse the exact answer schema and validate citation semantics

**Learning checkpoint**

Separate four questions: Is the response valid JSON? Is it the expected object
shape? Is its status internally consistent? Does it refer only to evidence supplied
for this request? Provider JSON mode answers only the first question.

**Files:**

- Create
  `src/personal_knowledge_agent/infrastructure/structured_answer_parser.py`
- Create `tests/unit/test_structured_answer_parser.py`

### RED: cover valid outputs

1. Parse an `answered` JSON object with `[S1]` and `[S2]` in the body and the same
   ordered IDs in `citation_ids`.
2. Assert the `GroundedAnswer` contains:
   - `AnswerStatus.ANSWERED`;
   - the exact answer text;
   - an immutable ordered citation tuple;
   - every supplied evidence record, not only cited evidence;
   - the provider-reported model ID.
3. Parse `insufficient_evidence` with no citation markers and an empty ID list.

### RED: reject schema defects

Use focused parametrized cases for:

- invalid JSON and Markdown code fences;
- a JSON list instead of an object;
- missing or additional fields;
- non-string status or answer;
- unknown status;
- non-list citation IDs;
- a non-string item in the list;
- blank answer text.

Each must raise `InvalidAnswerSchemaError` with a stable, safe explanation. Do not
include the complete provider response in an exception.

### RED: reject citation defects

Cover:

- an answered response with no citations;
- an unknown ID such as `S99`;
- duplicate declared IDs;
- repeated body use of the same valid citation, which is allowed and collapses to
  its first occurrence for the ordered comparison;
- a body citation missing from `citation_ids`;
- a declared citation absent from the body;
- the same valid IDs in a different order;
- an insufficient-evidence response containing either a body marker or declared ID.

These must raise `InvalidCitationError`.

### GREEN: implement strict parsing without repair

1. Decode with `json.loads` and translate `JSONDecodeError`.
2. Require a plain object and the exact key set.
3. Validate raw Python types before constructing enums or tuples. Reject booleans
   where a string/list is expected rather than relying on coercion.
4. Extract body markers with one explicit `[S<number>]` regular expression.
5. Deduplicate body markers in first-occurrence order and compare that sequence to
   `citation_ids` exactly.
6. Validate all IDs against the supplied evidence mapping.
7. Construct `GroundedAnswer`; never strip code fences or guess missing fields.

Verify:

```bash
uv run pytest tests/unit/test_structured_answer_parser.py -v
uv run ruff check src tests
uv run mypy src
```

Commit:

```text
feat: validate grounded answer citations
```

## Task 4: Orchestrate retrieval and grounded generation

**Learning checkpoint**

Distinguish an application service from infrastructure. `AnswerService` owns use-case
order and early exit; it must not know the system prompt string, JSON decoder, SDK
request shape, or retry policy.

**Files:**

- Create `src/personal_knowledge_agent/application/answer.py`
- Create `tests/unit/test_answer_service.py`
- Modify `tests/helpers.py` only if another small typed fake is genuinely shared

### RED: specify application behavior

1. Build a `SearchService` around a deterministic fake index and a
   `FakeLanguageModel`.
2. Assert default `answer(question)` forwards limit 5 and hybrid mode.
3. Assert explicit lexical/semantic modes and limits are forwarded unchanged.
4. Assert blank questions and limits outside 1..10 fail before index/model calls.
5. For an empty retrieval list, assert:
   - the model receives no calls;
   - status is `INSUFFICIENT_EVIDENCE`;
   - citations and evidence are empty;
   - `model_id` is `None`;
   - the local refusal text is stable and non-blank.
6. For non-empty results, configure the fake model with valid JSON and assert the
   returned answer contains mapped evidence and provider model ID.
7. Inspect recorded messages to prove the service used ranked evidence and the
   approved prompt path.
8. Assert `LanguageModelError`, `InvalidAnswerSchemaError`, and
   `InvalidCitationError` are not converted into false insufficient-evidence results.

### GREEN: add the thin service

1. Validate question and the ask-specific 1..10 limit.
2. Call `SearchService.search` with the requested mode.
3. Return a local refusal immediately for no matches.
4. Otherwise call `build_evidence`, `build_grounded_messages`, model `generate`, and
   `parse_grounded_answer` in that order.
5. Do not catch project errors merely to reword them.

Verify:

```bash
uv run pytest tests/unit/test_answer_service.py -v
uv run ruff check src tests
uv run mypy src
```

Commit:

```text
feat: orchestrate grounded answer generation
```

## Task 5: Add the OpenAI-compatible Chat Completions adapter

**Learning checkpoint**

Review the anti-corruption layer: SDK request and response objects may exist inside
one infrastructure module only. Understand why SDK retries cover transient transport
failures while schema, citation, authentication, bad-request, and truncation failures
must remain visible.

**Files:**

- Modify `pyproject.toml`
- Modify `uv.lock`
- Create
  `src/personal_knowledge_agent/infrastructure/openai_compatible_chat_model.py`
- Create `tests/unit/test_openai_compatible_chat_model.py`

### RED: establish the adapter boundary before adding the SDK

1. Create small typed fake client/response objects that expose only
   `chat.completions.create` and the response fields the adapter consumes.
2. Assert construction is lazy when no client is injected.
3. Assert `generate` sends:
   - configured model ID;
   - converted role/content messages;
   - `response_format={"type": "json_object"}`;
   - `stream=False`;
   - configured `max_tokens`, defaulting to 1024.
4. Assert the adapter maps provider model, content, and `stop` into `ModelOutput`.
5. Assert it rejects:
   - no choices;
   - blank or missing content;
   - `length` as `ModelOutputTruncatedError`;
   - any other non-`stop` finish reason as `LanguageModelError`.
6. Add provider exception cases for authentication, permission, bad request, rate
   limit, timeout, connection, server status, and an unknown SDK error. Build real
   SDK exception objects with minimal `httpx.Request`/`Response` fixtures after the
   dependency is added.
7. For every error, assert the message identifies a safe category and never contains
   the configured API key or full response body.

Run RED before dependency installation; the missing module or adapter should be the
expected failure.

### GREEN: add and lock the SDK

1. Add a current lower-bound dependency for `openai` to `pyproject.toml`.
2. Run `uv lock` and `uv sync --all-groups` in the Linux-native project environment.
3. Confirm the lock changes contain the SDK and necessary transport packages only;
   preserve the CPU-only PyTorch source configuration.

### GREEN: implement one isolated SDK adapter

1. Validate non-blank API key, base URL, model ID, and positive max tokens without
   ever echoing the key.
2. Define private structural protocols for the injected fake surface and cast the
   real SDK client at the adapter boundary; do not export SDK types.
3. Lazily build `OpenAI(api_key=..., base_url=..., max_retries=2)`.
4. Convert each `ChatMessage` to the SDK-compatible role/content dictionary.
5. Call non-streaming Chat Completions with JSON object output.
6. Map only the first choice and validate content/finish reason.
7. Catch SDK error classes from most specific to general and raise safe project
   errors with exception chaining.
8. Ignore `reasoning_content`; do not log the request or response.

Verify:

```bash
uv run pytest tests/unit/test_openai_compatible_chat_model.py -v
uv run ruff check src tests
uv run mypy src
```

Commit:

```text
feat: add OpenAI-compatible chat adapter
```

## Task 6: Load LLM settings and expose `pka ask`

**Learning checkpoint**

Review the composition root. Environment variables and concrete adapters belong at
the program edge. `AnswerService` should receive already-constructed dependencies,
and configuration tests should not mutate global process state unnecessarily.

**Files:**

- Create `src/personal_knowledge_agent/infrastructure/llm_settings.py`
- Create `tests/unit/test_llm_settings.py`
- Modify `.env.example`
- Modify `src/personal_knowledge_agent/cli.py`
- Modify `tests/test_cli.py`

### RED: specify configuration

1. Test `load_llm_settings` with an explicit mapping.
2. Assert `PKA_LLM_API_KEY` is required and whitespace-only values fail with a
   `ConfigurationError` naming the variable.
3. Assert defaults:
   - base URL `https://api.deepseek.com`;
   - model `deepseek-v4-flash`.
4. Assert non-blank overrides are preserved and surrounding whitespace is handled
   consistently.
5. Assert no error representation or dataclass `repr` exposes the API key. Mark the
   dataclass field with `repr=False` if the record may be inspected.

### GREEN: implement focused settings

1. Add frozen, slotted `LLMSettings` in `llm_settings.py`.
2. Accept `Mapping[str, str]`; do not read `os.environ` inside the loader.
3. Update `.env.example` to the three `PKA_LLM_*` names with an empty key and public
   defaults. Do not add `python-dotenv`; the file documents shell configuration.

### RED: specify the `ask` command

1. Extend the parser tests for:
   - default limit 5 and hybrid mode;
   - explicit lexical, semantic, and hybrid modes;
   - ask limit 1..10;
   - search still accepts its existing 1..20 range;
   - invalid modes receive an argparse choice error.
2. Patch `E5Embedder` and `OpenAICompatibleChatModel` so every CLI test stays
   offline. Provide an explicit environment mapping/key in the test harness.
3. Test a successful answer with two retrieved evidence records but only `S1` cited;
   assert output shows the answer and only the `S1` local path/heading.
4. Test local and model-declared insufficient evidence output without a Sources
   section.
5. Test missing API key, provider failure, invalid JSON, and invalid citation as
   concise `error:` messages without tracebacks or secrets.
6. Preserve every existing `health`, `ingest`, and `search` test.

### GREEN: compose the command explicitly

1. Add an ask-specific argparse limit validator for 1..10.
2. Add `ask` with query, mode, limit, and database options.
3. In the command path, call `load_llm_settings(os.environ)`, construct
   `E5Embedder`, `SQLiteKeywordIndex`, `SearchService`,
   `OpenAICompatibleChatModel`, and `AnswerService` explicitly.
4. Format answered output and map each cited ID back to its local `Evidence`.
5. Reuse heading formatting behavior where it remains clear; do not introduce a CLI
   framework or factory abstraction.
6. Let the existing `KnowledgeError` boundary convert failures to exit code 1.

Verify:

```bash
uv run pytest tests/unit/test_llm_settings.py tests/test_cli.py -v
uv run ruff check src tests
uv run mypy src
```

Commit:

```text
feat: expose grounded answers in CLI
```

## Task 7: Document, validate, and smoke-test the real DeepSeek workflow

**Learning checkpoint**

Be able to explain the completed distinction:

```text
Retrieval chooses evidence.
Grounded generation answers from that evidence.
Structured validation checks the answer contract.
Agent routing, still absent, would decide whether to retrieve or call tools.
```

**Files:**

- Modify `README.md`
- Create `docs/learning-log/phase-2a.md`
- Modify `AGENTS.md` only if the durable architecture summary or commands have
  materially changed

### Document the user workflow

1. Explain how to export `PKA_LLM_API_KEY`, and document the base URL/model overrides
   without showing a real key.
2. Document `pka ask`, its default hybrid mode, result limit, answer format, and
   evidence-refusal behavior.
3. State clearly that retrieved chunk text is sent to the configured remote provider
   while local paths, chunk IDs, and scores remain local.
4. Explain JSON mode versus local schema/citation validation.
5. Explain the prompt-injection boundary and why it is defence in depth rather than
   a mathematical guarantee.
6. Explain retry categories, API cost, and the rule that automated tests never call
   DeepSeek.

### Write the learning log

Cover:

- LLM next-token generation and hallucination risk;
- the retrieval/generation/agent-routing distinction;
- ports and adapters for provider independence;
- Chat Completions JSON object mode;
- local schema and citation semantics;
- data minimization and prompt injection;
- known limitations: no routing, threshold evaluation, token budgeter, streaming,
  memory, or answer-quality metrics.

### Run the deterministic quality gate

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pka health
```

All automated tests must pass with no API key and no network call.

### Obtain approval before the billable smoke test

Do not infer approval from the decision to use DeepSeek. Ask explicitly before using
the user's API key, sending fixture text, or incurring API cost.

After approval:

1. Use a temporary database outside the repository.
2. Ingest only `tests/fixtures/knowledge/rag_notes.md`.
3. Run:

   ```bash
   uv run pka ask \
     "RAG 为什么能减少模型脱离资料的猜测？" \
     --mode hybrid \
     --limit 5 \
     --db /tmp/pka-phase2a-deepseek.db
   ```

4. Confirm:
   - the provider returns valid JSON through Chat Completions;
   - finish reason is `stop`;
   - local parser accepts the answer;
   - at least one `[Sx]` citation maps to the public fixture;
   - no API key, full prompt, reasoning content, or private source is recorded.
5. Record the provider-reported model ID, citation outcome, latency observation, and
   limitations in the learning log. Do not commit the temporary database.

### Re-run the final quality gate

After documentation reflects the observed smoke test, run the complete deterministic
gate again and inspect the entire feature diff.

Commit:

```text
docs: explain grounded answer generation
```

## Final review and handoff

Before proposing a merge:

1. Confirm the feature branch contains only Phase 2A contracts, implementation,
   tests, dependencies, configuration documentation, and learning material.
2. Confirm `.env`, API keys, prompts containing private data, `.db` files, provider
   responses, and reasoning content are absent from Git.
3. Run:

   ```bash
   git diff --check main...HEAD
   git status --short --branch
   uv run pytest
   uv run ruff format --check .
   uv run ruff check .
   uv run mypy src
   uv run pka health
   ```

4. Summarize what the learner can now explain and identify Phase 2B question routing
   as the next design cycle.
5. Merge into `main` only after the user explicitly approves the completed feature.

# Phase 2B: Conservative Agent Routing Design

## Context and goal

Phase 2A added `pka ask`: every user question is answered through retrieval,
grounded generation, and strict citation validation. That is correct for knowledge
questions, but it is not yet agentic. The system cannot decide when retrieval is
unnecessary.

Phase 2B adds the smallest useful agent behavior: a conservative router that decides
whether a request should be answered directly or routed through the existing RAG
answer path. The new command is `pka agent`; `pka ask` remains unchanged and always
retrieves.

This phase deliberately avoids a general multi-tool loop. The action space is only
`direct` or `retrieve`, and direct responses are limited to greetings, capability
questions, and thanks. Any factual, learning, mixed, ambiguous, or unsupported
request retrieves.

## Learning objectives

By completing this phase, a contributor should be able to explain:

- why agent routing is a policy decision, not the same thing as answer generation;
- why conservative routing is safer than broad direct answering in a RAG assistant;
- how a small action space makes agent behavior testable;
- why rules can be better than an LLM for exact fast paths;
- why structured LLM output still needs local parsing and validation;
- how fail-closed behavior differs from hiding errors;
- what data is sent to the router versus the answer generator.

## Decisions

| Concern | Decision | Reason |
| --- | --- | --- |
| User-facing entry point | Add `pka agent` | Keeps the Phase 2A `pka ask` contract stable for comparison. |
| Routing strategy | Rule fast path plus LLM structured routing | Exact greetings are cheap and deterministic; the LLM handles phrasing the narrow rules do not match. |
| Default behavior | Retrieve when uncertain | The project is a personal knowledge agent, so unsupported direct answers would undermine trust. |
| Direct scope | Only `greeting`, `capabilities`, and `thanks` | These intents do not require knowledge-base evidence and are safe to template locally. |
| Factual direct answers | Not allowed | Direct factual answers would bypass citations and belong after evaluation data exists. |
| Rule matching | Whole-message matches only | `你好，请解释 RAG` must retrieve; a greeting embedded in a real question is not a direct greeting. |
| Router output | Exact JSON object with `action` and `direct_intent` | Small enough to validate manually and easy to inspect in tests. |
| Router failures | Display safe fallback category and retrieve | Users see that routing fell back without exposing provider traces or secrets. |
| CLI observability | Always print the route decision | Agent behavior should be visible while the project is still learning-first. |
| Provider boundary | Reuse the existing `LanguageModel` protocol | Routing is another structured model call, not a new provider abstraction. |
| Direct rendering | Local templates, language-aware | Direct responses cost no model call after a rule match and stay deterministic. |
| Privacy | Router receives only the user question | Retrieved evidence is sent only if the route is `retrieve`. |

## Alternatives considered

### Pure rule router

Pure rules are fast, offline, and easy to test. They are too brittle for natural
phrasing beyond a few exact cases, and adding many patterns would turn the router
into a hidden pile of special cases. Phase 2B keeps the exact cases but uses the LLM
for phrasing that still fits the same tiny intent set.

### Pure LLM router

A pure LLM router is more flexible, but it charges latency and money for obvious
messages such as `hi` or `谢谢`. It also makes simple behavior depend on a provider.
The selected hybrid keeps the easy path local and makes the LLM justify only one
small structured decision.

### Broad direct answering

The router could answer general questions directly when it thinks retrieval is not
needed. That is tempting, but it would create a second answer path without citations,
evaluation, or trust boundaries. Phase 2B rejects this. If the user asks "What is
RAG?", the route is `retrieve`, even though a base model probably knows the answer.

## Domain model

All routing records live in the domain layer and remain provider-independent.

```python
class RouteAction(StrEnum):
    DIRECT = "direct"
    RETRIEVE = "retrieve"


class DirectIntent(StrEnum):
    GREETING = "greeting"
    CAPABILITIES = "capabilities"
    THANKS = "thanks"


class RouteSource(StrEnum):
    RULE = "rule"
    LLM = "llm"
    FALLBACK = "fallback"


class RouterFallbackReason(StrEnum):
    PROVIDER_ERROR = "provider_error"
    INVALID_OUTPUT = "invalid_output"
```

`RouteDecision` is immutable:

```python
@dataclass(frozen=True, slots=True)
class RouteDecision:
    action: RouteAction
    direct_intent: DirectIntent | None
    source: RouteSource
    fallback_reason: RouterFallbackReason | None = None
```

The record enforces these invariants:

- `direct` requires a `direct_intent`;
- `retrieve` requires `direct_intent is None`;
- `fallback` decisions must retrieve and include a fallback reason;
- non-fallback decisions must not include a fallback reason.

`AgentResponse` wraps either a direct response or the existing grounded answer:

```python
@dataclass(frozen=True, slots=True)
class AgentResponse:
    route: RouteDecision
    direct_answer: str | None
    grounded_answer: GroundedAnswer | None
```

Exactly one of `direct_answer` and `grounded_answer` is present.

## Protocol boundary

The application layer receives a router through a small protocol:

```python
class QuestionRouter(Protocol):
    def route(self, question: str) -> RouteDecision: ...
```

`HybridQuestionRouter` implements this protocol outside the domain boundary. It uses
the existing `LanguageModel` protocol for the LLM branch and catches only known
router-safe failures:

- provider failures become `RouteSource.FALLBACK` with
  `RouterFallbackReason.PROVIDER_ERROR`;
- invalid route JSON or schema violations become `RouteSource.FALLBACK` with
  `RouterFallbackReason.INVALID_OUTPUT`.

Fallback is not a direct answer. It always retrieves.

## Components and responsibilities

### Exact direct intent matcher

`match_direct_intent(question: str) -> DirectIntent | None` is a pure helper. It
normalizes whitespace, case, and trailing lightweight punctuation, then matches only
the whole message. Examples that may match:

- `hi`
- `hello`
- `你好`
- `谢谢`
- `thanks`
- `what can you do`
- `你能做什么`

Examples that must not match:

- `你好，请解释 RAG`
- `thanks, now search my notes`
- `what can you do with RAG?`

### Router prompt builder

`build_route_messages(question: str) -> tuple[ChatMessage, ChatMessage]` creates one
system message and one user message. The system message states:

- output must be one JSON object with exactly `action` and `direct_intent`;
- `action` is either `direct` or `retrieve`;
- `direct_intent` is `greeting`, `capabilities`, `thanks`, or `null`;
- direct is allowed only for greetings, capability questions, and thanks;
- factual, learning, personal-knowledge, mixed, ambiguous, or uncertain requests
  must retrieve.

The user message contains only the question. No retrieved evidence is sent to the
router.

### Route parser

`parse_route_decision(output: ModelOutput) -> RouteDecision` accepts exactly:

```json
{"action":"retrieve","direct_intent":null}
```

or:

```json
{"action":"direct","direct_intent":"greeting"}
```

with `greeting`, `capabilities`, or `thanks` as the only direct intents. Extra
fields, arrays, code fences, blank values, unknown actions, unknown intents, direct
without intent, and retrieve with intent are invalid.

### Hybrid question router

`HybridQuestionRouter` runs the exact matcher first. If a direct intent is found, it
returns a rule-sourced direct decision and does not call the model. Otherwise it
builds route messages, calls the model, parses the result, and returns the LLM
decision. Known router failures become retrieve fallback decisions.

### Agent service

`AgentService` coordinates routing and response production:

1. reject a blank question before dependencies are called;
2. ask the router for a `RouteDecision`;
3. if the decision is direct, render a local template;
4. otherwise call the existing `AnswerService.answer(...)`;
5. return an `AgentResponse`.

The service does not know provider SDK details, embedding details, SQLite details,
or JSON parsing internals.

### CLI composition

`pka agent` accepts the same `query`, `--limit`, `--mode`, and `--db` options as
`pka ask`. It composes:

- `SQLiteKeywordIndex`;
- `E5Embedder`;
- `SearchService`;
- one `OpenAICompatibleChatModel`;
- `AnswerService`;
- `HybridQuestionRouter`;
- `AgentService`.

For a factual question, the first model call routes and the second model call
generates the grounded answer. For an exact rule direct message, the router does not
call the model, but the CLI still requires LLM configuration because the command is
composed for both possible paths.

## CLI output

The route decision is always printed first.

Direct example:

```text
Route: direct (greeting, rule)
Hi. I can search your local knowledge base and answer with citations when needed.
```

Retrieve example:

```text
Route: retrieve (llm)
Answer:
...
```

Fallback example:

```text
Route: retrieve (fallback: invalid router output)
Answer:
...
```

Provider error fallbacks use `fallback: provider error`. The CLI does not display raw
exceptions, stack traces, API keys, or provider response bodies.

## Error handling

The router catches only `LanguageModelError` and route structured-output errors.
After fallback, retrieval and grounded answer generation follow the existing Phase
2A behavior. If the answer-generation provider call fails, the command exits with
the existing user-facing error instead of hiding it behind another fallback.

Blank questions are rejected before routing. Invalid CLI limits or modes remain
argparse errors. Empty retrieval still returns the Phase 2A local insufficient
evidence response.

## Testing strategy

Automated tests remain offline and deterministic.

Unit tests cover:

- route domain-model invariants;
- exact direct rule matches and mixed-message non-matches;
- route prompt contents and data minimization;
- parser acceptance of the minimal JSON contract;
- parser rejection of malformed JSON, code fences, extra fields, wrong types,
  unknown enum values, direct without intent, and retrieve with intent;
- hybrid router rule short-circuit with zero model calls;
- hybrid router LLM decisions;
- hybrid router provider-error and invalid-output fallbacks;
- `AgentService` coordination for direct, retrieve, fallback retrieve, and invalid
  input.

CLI tests cover:

- `pka agent` parser defaults and boundaries;
- direct output with visible route decision;
- retrieve output reusing the existing grounded-answer display;
- fallback output with a safe fallback category;
- no traceback or secret leakage.

The full quality gate remains:

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
```

## Out of scope

Phase 2B does not implement:

- general direct factual answers;
- tool calling beyond choosing retrieval;
- multi-step planning or loops;
- conversation history;
- memory;
- streaming;
- PDF, web, or code ingestion;
- knowledge graphs;
- score thresholds or answer-quality evaluation.

Those are later phases. This phase makes one agent decision observable and reliable
before expanding the agent's action space.

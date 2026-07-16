# Phase 2B Conservative Agent Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pka agent`, which conservatively chooses a deterministic direct response for safe conversational intents or routes the request through the existing grounded RAG answer path.

**Architecture:** Provider-neutral routing records and a `QuestionRouter` protocol define the application boundary. A hybrid infrastructure adapter checks exact whole-message rules before asking the existing `LanguageModel` for strict JSON; known router failures fail closed to retrieval. `AgentService` coordinates the decision and either renders a local language-aware template or delegates to `AnswerService`.

**Tech Stack:** Python 3.12+, frozen dataclasses, `StrEnum`, standard-library JSON and regular expressions, OpenAI-compatible Chat Completions, pytest, Ruff, strict mypy, SQLite, multilingual E5, BM25, and RRF.

**Approved design:** `docs/superpowers/specs/2026-07-16-phase-2b-conservative-agent-routing-design.md`

## Global Constraints

- Work only in the isolated `codex/phase-2b-agent-router` worktree.
- Preserve `pka ask`: it always retrieves and its output contract does not change.
- `direct` is limited to `greeting`, `capabilities`, and `thanks`; factual, mixed, ambiguous, unsupported, and uncertain requests retrieve.
- Exact rules match the whole normalized message only.
- Router output has exactly `action` and `direct_intent`; local code validates it without repair.
- Provider and route-schema failures display a safe category and retrieve.
- The router receives only the user question; evidence is sent only after `retrieve`.
- Automated tests stay offline and deterministic; no secrets or provider response bodies appear in output.
- Generated databases stay under temporary test paths or `.pka/` and are never committed.
- On `/mnt/c`, run uv with `TMPDIR=/tmp`, `UV_PROJECT_ENVIRONMENT=/tmp/pka-phase2b-venv`, `UV_CACHE_DIR=/tmp/uv-cache`, and `UV_LINK_MODE=copy`.

---

## File Map

- Modify `src/personal_knowledge_agent/domain/errors.py`: define the route-schema error.
- Modify `src/personal_knowledge_agent/domain/models.py`: define routing enums, `RouteDecision`, and `AgentResponse` invariants.
- Modify `src/personal_knowledge_agent/domain/protocols.py`: define the replaceable `QuestionRouter` boundary.
- Create `src/personal_knowledge_agent/infrastructure/direct_intent_rules.py`: normalize and match safe whole-message intents.
- Create `src/personal_knowledge_agent/infrastructure/route_prompt_builder.py`: build the two routing messages without evidence.
- Create `src/personal_knowledge_agent/infrastructure/structured_route_parser.py`: validate the exact route JSON contract.
- Create `src/personal_knowledge_agent/infrastructure/hybrid_question_router.py`: combine rule, model, parser, and safe fallback behavior.
- Create `src/personal_knowledge_agent/application/agent.py`: coordinate routing, local direct rendering, and grounded answering.
- Modify `src/personal_knowledge_agent/cli.py`: add `pka agent`, compose dependencies, and display the route.
- Modify `tests/helpers.py`: add a sequential model fake for the router call followed by the answer call.
- Modify `tests/unit/test_models.py`: cover routing record invariants.
- Create `tests/unit/test_direct_intent_rules.py`.
- Create `tests/unit/test_route_prompt_builder.py`.
- Create `tests/unit/test_structured_route_parser.py`.
- Create `tests/unit/test_hybrid_question_router.py`.
- Create `tests/unit/test_agent_service.py`.
- Modify `tests/unit/test_helpers.py`: cover the sequential fake.
- Modify `tests/test_cli.py`: cover parser, direct, retrieve, and fallback behavior.
- Modify `README.md`: document `pka agent` and its conservative policy.
- Create `docs/learning-log/phase-2b.md`: explain routing policy, hybrid decisions, and fail-closed behavior.

### Task 1: Define routing domain contracts

**Files:**
- Modify: `src/personal_knowledge_agent/domain/errors.py`
- Modify: `src/personal_knowledge_agent/domain/models.py`
- Modify: `src/personal_knowledge_agent/domain/protocols.py`
- Modify: `tests/unit/test_models.py`

**Interfaces:**
- Consumes: existing `GroundedAnswer` and `InvalidQueryError`.
- Produces: `RouteAction`, `DirectIntent`, `RouteSource`, `RouterFallbackReason`, `RouteDecision`, `AgentResponse`, `QuestionRouter.route(question: str) -> RouteDecision`, and `InvalidRouteSchemaError`.

- [x] **Step 1: Write failing routing-model tests**

Add focused tests equivalent to:

```python
def test_direct_route_requires_intent() -> None:
    with pytest.raises(ValueError, match="direct route requires"):
        RouteDecision(RouteAction.DIRECT, None, RouteSource.RULE)


def test_retrieve_route_rejects_direct_intent() -> None:
    with pytest.raises(ValueError, match="retrieve route"):
        RouteDecision(
            RouteAction.RETRIEVE,
            DirectIntent.GREETING,
            RouteSource.LLM,
        )


def test_fallback_route_requires_reason_and_retrieval() -> None:
    with pytest.raises(ValueError, match="fallback"):
        RouteDecision(RouteAction.RETRIEVE, None, RouteSource.FALLBACK)


def test_agent_response_contains_exactly_one_answer() -> None:
    route = RouteDecision(
        RouteAction.DIRECT,
        DirectIntent.GREETING,
        RouteSource.RULE,
    )
    with pytest.raises(ValueError, match="exactly one"):
        AgentResponse(route, None, None)
```

- [x] **Step 2: Verify RED**

Run:

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase2b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_models.py -v
```

Expected: collection fails because the routing types do not exist.

- [x] **Step 3: Add the minimum contracts**

Implement the approved enums and frozen records. `RouteDecision.__post_init__` must enforce:

```python
if self.action is RouteAction.DIRECT and self.direct_intent is None:
    raise ValueError("direct route requires a direct intent")
if self.action is RouteAction.RETRIEVE and self.direct_intent is not None:
    raise ValueError("retrieve route must not have a direct intent")
if self.source is RouteSource.FALLBACK:
    if self.action is not RouteAction.RETRIEVE or self.fallback_reason is None:
        raise ValueError("fallback route must retrieve and include a reason")
elif self.fallback_reason is not None:
    raise ValueError("non-fallback route must not include a fallback reason")
```

`AgentResponse.__post_init__` must use `(self.direct_answer is None) == (self.grounded_answer is None)` to reject zero or two answer values, and reject a direct answer on a retrieve route or a grounded answer on a direct route. Add:

```python
class QuestionRouter(Protocol):
    def route(self, question: str) -> RouteDecision:
        """Choose the safe path for one user question."""
        ...
```

- [x] **Step 4: Verify GREEN**

Run the targeted model tests, Ruff, and mypy. Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/domain tests/unit/test_models.py
git commit -m "feat: define agent routing contracts"
```

### Task 2: Add deterministic direct-intent rules and templates

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/direct_intent_rules.py`
- Create: `src/personal_knowledge_agent/application/agent.py`
- Create: `tests/unit/test_direct_intent_rules.py`
- Create: `tests/unit/test_agent_service.py`

**Interfaces:**
- Consumes: `DirectIntent`, `RouteDecision`, `QuestionRouter`, `AnswerService`.
- Produces: `match_direct_intent(question: str) -> DirectIntent | None`, `render_direct_answer(intent: DirectIntent, question: str) -> str`, and `AgentService.respond(...) -> AgentResponse`.

- [x] **Step 1: Write failing exact-rule tests**

Parametrize safe inputs (`hi`, `Hello!`, `你好。`, `谢谢`, `what can you do?`, `你能做什么？`) and assert their intents. Parametrize mixed or factual inputs (`你好，请解释 RAG`, `thanks, now search my notes`, `what can you do with RAG?`, `What is RAG?`, blank text) and assert `None`.

- [x] **Step 2: Verify exact-rule RED**

Run `uv run pytest tests/unit/test_direct_intent_rules.py -v`; expect missing-module failure.

- [x] **Step 3: Implement exact whole-message matching**

Normalize with collapsed whitespace, `casefold()`, and trailing `" .!?。！？,，;；~～"` removal, then use immutable exact-value maps. Do not use substring or prefix matching.

- [x] **Step 4: Verify exact-rule GREEN**

Run the targeted rule tests; expect all pass.

- [x] **Step 5: Write failing AgentService tests**

Use small typed fakes to prove:

```python
response = AgentService(router, answer_service).respond("你好")
assert response.route.action is RouteAction.DIRECT
assert response.direct_answer is not None
assert answer_service.calls == []
```

Also prove English/Chinese templates, retrieval delegation with unchanged `limit` and `mode`, fallback retrieval, blank rejection before router calls, and limit range 1..10.

- [x] **Step 6: Verify AgentService RED**

Run `uv run pytest tests/unit/test_agent_service.py -v`; expect missing service failure.

- [x] **Step 7: Implement the thin service and templates**

Use these stable response families:

```python
_DIRECT_ANSWERS = {
    (DirectIntent.GREETING, False): "Hi. I can search your local knowledge base and answer with citations when needed.",
    (DirectIntent.GREETING, True): "你好。我可以在需要时检索你的本地知识库，并用引用回答问题。",
    (DirectIntent.CAPABILITIES, False): "I can search your local knowledge base, answer from retrieved evidence, and show cited sources.",
    (DirectIntent.CAPABILITIES, True): "我可以检索你的本地知识库、基于证据回答问题，并显示引用来源。",
    (DirectIntent.THANKS, False): "You're welcome.",
    (DirectIntent.THANKS, True): "不客气。",
}
```

Detect Chinese with any character in `\u4e00` through `\u9fff`. Validate the question and limit before calling dependencies. Direct returns a local answer; retrieve delegates to `AnswerService.answer`.

- [x] **Step 8: Verify AgentService GREEN and commit**

Run both targeted files, Ruff, and mypy; then commit as `feat: add deterministic direct responses`.

### Task 3: Build and validate the structured route call

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/route_prompt_builder.py`
- Create: `src/personal_knowledge_agent/infrastructure/structured_route_parser.py`
- Create: `tests/unit/test_route_prompt_builder.py`
- Create: `tests/unit/test_structured_route_parser.py`

**Interfaces:**
- Produces: `build_route_messages(question: str) -> tuple[ChatMessage, ChatMessage]` and `parse_route_decision(output: ModelOutput) -> RouteDecision`.

- [x] **Step 1: Write failing prompt tests**

Assert one system and one user message, the exact two-field JSON contract, the three direct intents, the conservative retrieve categories, and that only the original question—not evidence, database paths, or model settings—appears in the user message. Assert blank input raises `InvalidQueryError`.

- [x] **Step 2: Verify prompt RED, then implement minimal prompt builder**

The user message format is exactly `QUESTION:\n{question}`. The system policy must explicitly say: direct only for whole-request greeting/capabilities/thanks; otherwise retrieve; output one JSON object without Markdown fences.

- [x] **Step 3: Verify prompt GREEN**

Run `uv run pytest tests/unit/test_route_prompt_builder.py -v`; expect all pass.

- [x] **Step 4: Write failing parser tests**

Accept only:

```json
{"action":"retrieve","direct_intent":null}
{"action":"direct","direct_intent":"greeting"}
```

with all approved direct intents. Reject invalid JSON, code fences, arrays, missing/extra fields, non-string actions, unknown action/intent, direct with null, retrieve with an intent, and blank values. Every rejection raises `InvalidRouteSchemaError` without echoing the full model response.

- [x] **Step 5: Verify parser RED, then implement strict parsing**

Use `json.loads`, require `type(payload) is dict`, require exactly `{"action", "direct_intent"}`, validate raw Python types, construct enums inside safe `try/except ValueError`, then construct an LLM-sourced `RouteDecision`. Do not strip fences or repair fields.

- [x] **Step 6: Verify parser GREEN and commit**

Run both targeted files, Ruff, and mypy; then commit as `feat: add structured route contract`.

### Task 4: Implement the fail-closed hybrid router

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/hybrid_question_router.py`
- Create: `tests/unit/test_hybrid_question_router.py`

**Interfaces:**
- Consumes: `LanguageModel.generate`, `match_direct_intent`, `build_route_messages`, `parse_route_decision`.
- Produces: `HybridQuestionRouter.route(question: str) -> RouteDecision`.

- [x] **Step 1: Write failing behavioral tests**

Prove exact rules return a rule-sourced direct route with zero model calls; unmatched questions use the model; valid model direct/retrieve decisions are preserved; `LanguageModelError` yields retrieve/provider-error fallback; `InvalidRouteSchemaError` yields retrieve/invalid-output fallback; blank questions fail before model calls; unexpected exceptions propagate.

- [x] **Step 2: Verify RED**

Run `uv run pytest tests/unit/test_hybrid_question_router.py -v`; expect missing router failure.

- [x] **Step 3: Implement minimal hybrid flow**

```python
intent = match_direct_intent(question)
if intent is not None:
    return RouteDecision(RouteAction.DIRECT, intent, RouteSource.RULE)
try:
    output = self._language_model.generate(build_route_messages(question))
    return parse_route_decision(output)
except LanguageModelError:
    return RouteDecision(RouteAction.RETRIEVE, None, RouteSource.FALLBACK, RouterFallbackReason.PROVIDER_ERROR)
except InvalidRouteSchemaError:
    return RouteDecision(RouteAction.RETRIEVE, None, RouteSource.FALLBACK, RouterFallbackReason.INVALID_OUTPUT)
```

- [x] **Step 4: Verify GREEN and commit**

Run the targeted tests, Ruff, and mypy; then commit as `feat: route agent requests conservatively`.

### Task 5: Compose and display `pka agent`

**Files:**
- Modify: `tests/helpers.py`
- Modify: `tests/unit/test_helpers.py`
- Modify: `src/personal_knowledge_agent/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `ScriptedLanguageModel(results: Sequence[ModelOutput | Exception])`, CLI `_agent(...)`, `_format_route(...)`, and `_print_agent_response(...)`.

- [x] **Step 1: Write failing sequential-fake tests**

Assert the fake returns configured results in order, records immutable message tuples, raises configured exceptions, and raises a clear `AssertionError` when exhausted.

- [x] **Step 2: Verify RED, implement the fake, verify GREEN**

Keep the fake in tests only. Do not add test-only hooks to production code.

- [x] **Step 3: Write failing CLI parser and output tests**

Cover defaults (`limit=5`, `mode=hybrid`), modes, limits 1 and 10, rejected limits 0 and 11, direct rule output, LLM retrieve plus grounded answer output, invalid-router fallback output plus answer, provider-router fallback output plus answer, missing API key, and absence of traceback/secret/raw provider output.

- [x] **Step 4: Verify CLI RED**

Run `uv run pytest tests/test_cli.py -v`; expect `agent` parser/dispatch failures.

- [x] **Step 5: Implement CLI composition and display**

Add an `agent` parser mirroring `ask`. Compose one model instance shared by `HybridQuestionRouter` and `AnswerService`. Format routes exactly:

```text
Route: direct (greeting, rule)
Route: retrieve (llm)
Route: retrieve (fallback: provider error)
Route: retrieve (fallback: invalid router output)
```

Then print either the direct text or reuse `_print_answer` unchanged.

- [x] **Step 6: Verify CLI GREEN and commit**

Run CLI tests plus all new unit tests, Ruff, and mypy; commit as `feat: add agent command`.

### Task 6: Document, verify, and finish the branch

**Files:**
- Modify: `README.md`
- Create: `docs/learning-log/phase-2b.md`

**Interfaces:**
- Produces: user-facing command documentation and the Phase 2B learning record.

- [x] **Step 1: Update documentation**

Document that `pka agent` exposes its route, exact greetings can avoid retrieval/model calls, factual and mixed messages retrieve, and router failures fail closed. Explain the two model calls on an LLM-routed factual request and show direct/retrieve examples without secrets.

- [x] **Step 2: Run targeted regression tests**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase2b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_models.py tests/unit/test_direct_intent_rules.py tests/unit/test_route_prompt_builder.py tests/unit/test_structured_route_parser.py tests/unit/test_hybrid_question_router.py tests/unit/test_agent_service.py tests/test_cli.py -v
```

Expected: all pass.

- [x] **Step 3: Run the complete quality gate**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase2b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase2b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff format --check .
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase2b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff check .
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase2b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run mypy src
```

Expected: all tests pass, Ruff reports no changes/errors, and mypy reports success.

- [x] **Step 4: Commit documentation**

```bash
git add README.md docs/learning-log/phase-2b.md
git commit -m "docs: explain conservative agent routing"
```

- [x] **Step 5: Review the complete diff**

Run `git status --short`, `git diff --check`, and `git log --oneline --decorate -6`. Confirm no `.env`, `.pka/`, provider output, `papers/`, `references/`, or `research/` files are included.

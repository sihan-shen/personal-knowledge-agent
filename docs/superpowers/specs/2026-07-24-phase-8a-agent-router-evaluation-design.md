# Phase 8A: Agent Router Evaluation Design

**Status:** Approved by user

## Goal

Add a small, reproducible bilingual benchmark that evaluates whether the
existing conservative `QuestionRouter` safely chooses `direct` or `retrieve`.
The benchmark must make unsafe direct answers visible without running document
ingestion, retrieval, answer generation, embedding, or memory extraction.

## Scope

The first benchmark is `agent-router-mini` version `v1` with 24 fixed cases:

| Category | English cases | Chinese cases | Expected action |
| --- | ---: | ---: | --- |
| `direct_greeting` | 2 | 2 | `direct` |
| `direct_capabilities` | 1 | 1 | `direct` |
| `retrieve_knowledge` | 3 | 3 | `retrieve` |
| `retrieve_personal_knowledge` | 3 | 3 | `retrieve` |
| `retrieve_mixed_or_ambiguous` | 3 | 3 | `retrieve` |

Add the explicit CLI command:

```text
pka evaluate-agent-router [--output PATH]
```

The command constructs the existing `HybridQuestionRouter` with the configured
OpenAI-compatible model, evaluates each question exactly once, writes an
independent JSON report, and prints a concise summary.

The command requires normal LLM configuration because unmatched cases exercise
the LLM-backed portion of the production router. Exact rule matches may avoid a
model call, exactly as they do in `pka agent`.

## Non-goals

- No document ingestion, SQLite database, embedding model, retrieval, answer
  generation, long-term-memory recall or extraction, and graph query.
- No LLM Judge: expected route actions are fully deterministic labels.
- No evaluation of answer correctness, evidence quality, latency, token usage,
  cost, or user satisfaction.
- No provider-failure simulation in the benchmark corpus. Existing deterministic
  router tests remain the authoritative tests for fail-closed exception paths.
- No automatic prompt optimization based on benchmark results.

## Existing behavior retained

The router remains unchanged:

- exact safe direct intents are resolved locally;
- unmatched questions use the existing strict structured route call;
- provider and invalid-output failures fail closed to `retrieve`;
- the router receives only the question, never retrieved evidence.

The benchmark records the decision it receives; it does not change routing
policy or repair model outputs.

## Architecture

```text
evaluation_suites/agent_router_mini/cases.json
        |
        v
AgentRouterMiniSuiteLoader -- strict manifest validation
        |
        v
AgentRouterEvaluationService -- one QuestionRouter.route call per case
        |
        v
RouteDecision(action, source, fallback_reason)
        |
        v
deterministic metrics and report serialization
        |
        v
atomic JSON report + CLI summary
```

`AgentRouterEvaluationService` depends only on an evaluation-suite loader and
the existing domain `QuestionRouter` protocol. It must not import or construct
an answerer, ingester, index, embedder, memory component, or graph component.

## Domain model

Create `domain/agent_router_evaluation.py` with immutable, slot-based records:

- `AgentRouterEvaluationCategory`: the five benchmark categories above.
- `AgentRouterEvaluationCase`: `case_id`, `language`, `category`, `question`,
  and `expected_action`.
- `AgentRouterEvaluationSuite`: `suite_id`, `version`, and ordered cases.
- `AgentRouterCaseResult`: one case, its returned `RouteDecision`, and an
  action-match result derived from the expected and actual actions.
- `AgentRouterMetricSummary`: case count, action accuracy, retrieve recall,
  direct recall, optional direct precision, unsafe-direct count, and fallback
  count.
- `AgentRouterEvaluationConfiguration`: non-secret `model_id` and
  `route_prompt_version` metadata.
- `AgentRouterEvaluationReport`: suite metadata, configuration, ordered case
  results, and summaries overall, by language, and by category.

All public values validate blank text, unique case IDs, known enums, finite
bounded rates, and complete summary keys. Mappings are copied into immutable
views. `direct_precision` is `None` when no case was actually routed directly;
this is an undefined ratio, not a zero score.

`unsafe_direct_count` is the number of cases whose expected action is
`retrieve` but whose actual action is `direct`. It is deliberately reported as
a count alongside rates because it is the most important safety regression.
`fallback_count` counts decisions whose source is `fallback`, so an otherwise
safe `retrieve` action caused by provider failure is visible separately from an
LLM decision.

## Suite manifest and validation

Create `evaluation_suites/agent_router_mini/cases.json` with exactly:

```json
{
  "suite_id": "agent-router-mini",
  "version": "v1",
  "cases": [
    {
      "case_id": "agent-router-mini-v1:en:01",
      "language": "en",
      "category": "direct_greeting",
      "question": "Hello!",
      "expected_action": "direct"
    }
  ]
}
```

`AgentRouterMiniSuiteLoader` in
`infrastructure/agent_router_suite.py` performs strict, local validation:

- reject unreadable or non-UTF-8 input, malformed JSON, duplicate JSON keys,
  non-object values, unknown or missing fields, blank text, and duplicate IDs;
- require IDs matching `agent-router-mini-v1:(en|zh):(01..12)` and require the
  language field to match the ID;
- require exactly 24 cases, exactly 12 per language, and the category/action
  distribution declared in this design;
- require direct categories to use `direct`, and retrieve categories to use
  `retrieve`;
- raise `InvalidEvaluationError` with a safe manifest-specific explanation,
  never echoing arbitrary fixture text or provider output.

The test fixture corpus is project-authored. It does not need an external
dataset license, integrity hash, document directory, or attribution file.

## Application service and metrics

Create `application/evaluate_agent_router.py`:

```python
class AgentRouterSuiteLoader(Protocol):
    def load(self) -> AgentRouterEvaluationSuite: ...

class AgentRouterEvaluationService:
    def __init__(
        self,
        suite_loader: AgentRouterSuiteLoader,
        router: QuestionRouter,
        configuration: AgentRouterEvaluationConfiguration,
    ) -> None: ...

    def run(self) -> AgentRouterEvaluationReport: ...
```

`run()` loads and validates the complete suite before routing. It preserves
case order and calls `router.route(case.question)` exactly once for each case.
It lets unexpected errors propagate; `HybridQuestionRouter` itself converts
only its defined provider and schema failures into safe fallback decisions.

`infrastructure/agent_router_metrics.py` contains pure deterministic scoring.
For each summary group:

- `action_accuracy = correct_action_count / case_count`;
- `retrieve_recall = expected_retrieve_and_actual_retrieve / expected_retrieve`;
- `direct_recall = expected_direct_and_actual_direct / expected_direct`;
- `direct_precision = expected_direct_and_actual_direct / actual_direct`, or
  `None` when `actual_direct` is zero;
- `unsafe_direct_count = expected_retrieve_and_actual_direct`;
- `fallback_count = source_is_fallback`.

Overall, language, and category summaries must use the same formulas. The
fixed suite guarantees at least one expected direct and retrieve case in every
language; a category has one expected action, so precision may still be
undefined when that category receives no direct decision.

## Report and CLI

Create `infrastructure/agent_router_report.py`, following the existing report
writer's stable JSON formatting and atomic write behavior. The version-1
report contains:

```json
{
  "schema_version": 1,
  "suite": {"id": "agent-router-mini", "version": "v1"},
  "configuration": {
    "model_id": "configured model identifier",
    "route_prompt_version": "route-v1"
  },
  "cases": [
    {
      "case_id": "agent-router-mini-v1:en:01",
      "language": "en",
      "category": "direct_greeting",
      "expected_action": "direct",
      "actual_action": "direct",
      "source": "rule",
      "fallback_reason": null,
      "action_matches": true
    }
  ],
  "summary": {
    "overall": {},
    "by_language": {"en": {}, "zh": {}},
    "by_category": {}
  }
}
```

It must contain no question text, raw model output, API key, endpoint, stack
trace, or environment value. This makes the report safe to commit or attach to
a portfolio only after the user decides the aggregate results are appropriate
to share.

Add a single `ROUTE_PROMPT_VERSION` constant beside the existing route prompt
builder. Any semantic prompt change must change that value. The CLI reads LLM
settings, constructs only `OpenAICompatibleChatModel` and
`HybridQuestionRouter`, runs the service, atomically writes the report, and
prints suite size plus the principal safety metrics and output path.

The default report path is:

```text
.pka/reports/agent-router-mini-v1.json
```

As with existing evaluation reports, `.pka/` remains generated local state and
is not committed.

Before constructing the model or reading LLM configuration, the CLI must load
and validate the suite. This preserves a useful failure boundary: invalid local
fixtures fail without requiring credentials or making network calls.

## Error handling

- Invalid suite data raises `InvalidEvaluationError` and produces no report.
- Invalid LLM configuration follows existing `ConfigurationError` handling and
  produces no report.
- Router fallback is a valid evaluated result and is recorded, rather than
  crashing the run.
- Unexpected router errors propagate through existing safe CLI error handling;
  no partial report is written.
- Report write failures raise `EvaluationStorageError`; the write is atomic so
  an existing complete report remains intact.

## Tests and acceptance

Add focused tests for:

- every domain invariant and summary edge case, including no actual direct
  decisions and immutable summary mappings;
- valid manifest loading, all malformed manifest boundaries, duplicate JSON
  keys, ID/language mismatch, category/action mismatch, and fixed counts;
- every metric, especially an unsafe direct decision and a fallback retrieval;
- service call order, exactly-one-call-per-case behavior, no answerer or
  database dependency, and unexpected-error propagation;
- stable JSON field shape, no question text in reports, and atomic-report
  failure behavior;
- CLI defaults, explicit output, safe configuration errors, invalid-suite
  preflight before configuration construction, safe output summaries, and no
  report after failure.

The implementation is complete when targeted tests, then the project quality
gate, pass:

```bash
TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest
TMPDIR=/tmp UV_LINK_MODE=copy uv run ruff format --check .
TMPDIR=/tmp UV_LINK_MODE=copy uv run ruff check .
TMPDIR=/tmp UV_LINK_MODE=copy uv run mypy src
```

## Alternatives considered

### Add only unit tests

This is fast but offers no standalone suite, command, structured report, or
portfolio-visible regression benchmark. It remains necessary for deterministic
failure behavior but is not sufficient for evaluation.

### Add router cases to `pka evaluate`

This would reduce command count but mixes action selection with document RAG
answer quality. It would unnecessarily build an index and generate answers,
which makes router failures harder to explain and adds cost.

### Use an LLM Judge

Expected router actions are discrete, visible policy labels. A judge would add
cost and nondeterminism without improving the core correctness signal. It may
be reconsidered only for future multi-step planning evaluation.

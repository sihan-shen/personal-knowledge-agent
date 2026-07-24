# Phase 8A Agent Router Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible bilingual `agent-router-mini` benchmark and `pka evaluate-agent-router` command that evaluates the existing conservative Router without running RAG answer generation or storage.

**Architecture:** A strict local JSON suite loader produces immutable domain cases. A focused application service calls the existing `QuestionRouter` once per case and passes results to pure metrics and an atomic report writer. The CLI constructs only the existing OpenAI-compatible model and `HybridQuestionRouter`; it pre-validates the suite before reading provider configuration.

**Tech Stack:** Python 3.12+, dataclasses, `StrEnum`, `MappingProxyType`, typing `Protocol`, standard-library JSON, existing OpenAI-compatible Chat Completions adapter, pytest, Ruff, mypy, uv.

## Global Constraints

- Keep `domain/` independent of OpenAI, JSON, SQLite, filesystem paths, embeddings, and answer-generation classes.
- Reuse `QuestionRouter`, `RouteAction`, `RouteDecision`, `RouteSource`, `RouterFallbackReason`, and `EvaluationLanguage`; do not change production routing policy.
- Add no runtime dependency; all type annotations must pass strict mypy.
- The evaluation service must not construct or import an ingester, answerer, index, embedder, memory component, or graph component.
- The packaged suite has exactly 24 cases: 12 English, 12 Chinese, and the approved category/action distribution.
- Use deterministic fakes in tests; no test makes a live provider or network call.
- Generated reports remain under `.pka/` and must never be committed.
- Router provider/schema failures remain valid fallback results; unexpected errors produce no report.
- Use `TMPDIR=/tmp UV_LINK_MODE=copy` with every `uv` command in this Windows-mounted workspace.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/personal_knowledge_agent/domain/agent_router_evaluation.py` | Immutable suite, case, result, configuration, metric summary, and report values. |
| `src/personal_knowledge_agent/application/evaluate_agent_router.py` | Loader protocol and Router-only evaluation orchestration. |
| `src/personal_knowledge_agent/infrastructure/agent_router_suite.py` | Strict JSON loading and fixed suite-distribution validation. |
| `src/personal_knowledge_agent/infrastructure/agent_router_metrics.py` | Pure action, recall, precision, safety, and fallback aggregation. |
| `src/personal_knowledge_agent/infrastructure/agent_router_report.py` | Stable privacy-safe JSON serialization and atomic report persistence. |
| `src/personal_knowledge_agent/infrastructure/route_prompt_builder.py` | Public route-prompt version constant. |
| `src/personal_knowledge_agent/cli.py` | `evaluate-agent-router` parser, composition root, output defaults, and summary. |
| `src/personal_knowledge_agent/evaluation_suites/agent_router_mini/cases.json` | Project-authored versioned 24-case bilingual benchmark. |
| `tests/unit/test_agent_router_evaluation_models.py` | Domain-value invariants. |
| `tests/unit/test_agent_router_suite.py` | Manifest validation and packaged-suite assertions. |
| `tests/unit/test_agent_router_metrics.py` | Deterministic metric calculations. |
| `tests/unit/test_evaluate_agent_router.py` | Service call boundary and propagation behavior. |
| `tests/unit/test_agent_router_report.py` | Stable serialization and atomic persistence. |
| `tests/test_cli.py` | Parser, preflight, default output, summary, and safe failures. |
| `README.md` | User-facing command, behavior, and scope boundary. |
| `docs/learning-log/phase-8a.md` | Learning-oriented explanation of routing evaluation and safety metrics. |

## Task 1: Define immutable Router-evaluation contracts

**Files:**
- Create: `src/personal_knowledge_agent/domain/agent_router_evaluation.py`
- Create: `tests/unit/test_agent_router_evaluation_models.py`

**Interfaces:**
- Consumes: `EvaluationLanguage`, `RouteAction`, and `RouteDecision` from existing domain modules.
- Produces: `AgentRouterEvaluationCategory`, `AgentRouterEvaluationCase`, `AgentRouterEvaluationSuite`, `AgentRouterCaseResult`, `AgentRouterMetricSummary`, `AgentRouterEvaluationConfiguration`, and `AgentRouterEvaluationReport`.

- [ ] **Step 1: Write failing domain-value tests.**

```python
def test_case_result_derives_action_match_from_route_decision() -> None:
    result = AgentRouterCaseResult(
        case(expected_action=RouteAction.RETRIEVE),
        RouteDecision(RouteAction.DIRECT, DirectIntent.GREETING, RouteSource.RULE),
    )
    assert result.action_matches is False


def test_metric_summary_allows_undefined_direct_precision_only_without_directs() -> None:
    AgentRouterMetricSummary(2, 1.0, 1.0, 1.0, None, 0, 0)
    with pytest.raises(ValueError, match="direct_precision"):
        AgentRouterMetricSummary(2, 1.0, 1.0, 1.0, 1.1, 0, 0)
```

Also cover blank identifiers, duplicate case IDs, category/action mismatch, non-finite rates, invalid count ranges, incomplete language/category summary keys, frozen instances, and immutable copied mappings.

- [ ] **Step 2: Run the focused tests and confirm failure.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/unit/test_agent_router_evaluation_models.py -v`

Expected: FAIL because `agent_router_evaluation` does not exist.

- [ ] **Step 3: Implement the smallest complete domain model.**

```python
class AgentRouterEvaluationCategory(StrEnum):
    DIRECT_GREETING = "direct_greeting"
    DIRECT_CAPABILITIES = "direct_capabilities"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    RETRIEVE_PERSONAL_KNOWLEDGE = "retrieve_personal_knowledge"
    RETRIEVE_MIXED_OR_AMBIGUOUS = "retrieve_mixed_or_ambiguous"


@dataclass(frozen=True, slots=True)
class AgentRouterCaseResult:
    case: AgentRouterEvaluationCase
    decision: RouteDecision

    @property
    def action_matches(self) -> bool:
        return self.case.expected_action is self.decision.action
```

Use the established `_require_text` and bounded-float validation pattern from `domain/evaluation.py`. `AgentRouterEvaluationCase.__post_init__` must reject direct categories paired with `retrieve` and retrieval categories paired with `direct`. `AgentRouterEvaluationReport` must require exactly `en` and `zh` language summaries and all five category summaries, then wrap copied mappings in `MappingProxyType`.

- [ ] **Step 4: Re-run focused tests and static types.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/unit/test_agent_router_evaluation_models.py -v`

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run mypy src/personal_knowledge_agent/domain`

Expected: both PASS.

- [ ] **Step 5: Commit the contracts.**

```bash
git add src/personal_knowledge_agent/domain/agent_router_evaluation.py
git add tests/unit/test_agent_router_evaluation_models.py
git commit -m "feat: define agent router evaluation contracts"
```

## Task 2: Package and validate the 24-case bilingual suite

**Files:**
- Create: `src/personal_knowledge_agent/evaluation_suites/agent_router_mini/cases.json`
- Create: `src/personal_knowledge_agent/infrastructure/agent_router_suite.py`
- Create: `tests/unit/test_agent_router_suite.py`

**Interfaces:**
- Consumes: domain suite/case/category values and `InvalidEvaluationError`.
- Produces: `AgentRouterMiniSuiteLoader(suite_directory: Path = _SUITE_DIRECTORY)` and `load() -> AgentRouterEvaluationSuite`.

- [ ] **Step 1: Write failing suite-loader tests.**

```python
def test_packaged_suite_has_the_approved_bilingual_distribution() -> None:
    suite = AgentRouterMiniSuiteLoader().load()
    assert len(suite.cases) == 24
    assert Counter(case.language for case in suite.cases) == {
        EvaluationLanguage.EN: 12,
        EvaluationLanguage.ZH: 12,
    }


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    (tmp_path / "cases.json").write_text(
        '{"suite_id":"agent-router-mini","suite_id":"other"}', encoding="utf-8"
    )
    with pytest.raises(InvalidEvaluationError, match="duplicate JSON key"):
        AgentRouterMiniSuiteLoader(tmp_path).load()
```

Add tests for unknown/missing fields, invalid UTF-8, malformed JSON, bad ID format, ID/language mismatch, duplicated IDs, category/action mismatch, missing category count, and a non-file suite directory.

- [ ] **Step 2: Run the focused tests and confirm failure.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/unit/test_agent_router_suite.py -v`

Expected: FAIL because the loader and packaged suite do not exist.

- [ ] **Step 3: Add the complete fixed manifest.**

Create exactly these cases; IDs `01` through `12` must be ordered within each language.

| ID suffix | Language | Category | Question | Expected action |
| --- | --- | --- | --- | --- |
| `en:01` | en | `direct_greeting` | `Hello!` | direct |
| `en:02` | en | `direct_greeting` | `Good morning!` | direct |
| `en:03` | en | `direct_capabilities` | `What can you do?` | direct |
| `en:04` | en | `retrieve_knowledge` | `What is retrieval-augmented generation?` | retrieve |
| `en:05` | en | `retrieve_knowledge` | `How do vector databases support semantic retrieval?` | retrieve |
| `en:06` | en | `retrieve_knowledge` | `How does BM25 rank documents?` | retrieve |
| `en:07` | en | `retrieve_personal_knowledge` | `Based on my notes, what are my project goals?` | retrieve |
| `en:08` | en | `retrieve_personal_knowledge` | `What did I write about Phase 2?` | retrieve |
| `en:09` | en | `retrieve_personal_knowledge` | `Summarize the document I uploaded yesterday.` | retrieve |
| `en:10` | en | `retrieve_mixed_or_ambiguous` | `Hi, can you explain hybrid retrieval?` | retrieve |
| `en:11` | en | `retrieve_mixed_or_ambiguous` | `Can you tell me what you know from my knowledge base?` | retrieve |
| `en:12` | en | `retrieve_mixed_or_ambiguous` | `I need help with my study notes. What should I review?` | retrieve |
| `zh:01` | zh | `direct_greeting` | `你好！` | direct |
| `zh:02` | zh | `direct_greeting` | `早上好！` | direct |
| `zh:03` | zh | `direct_capabilities` | `你能做什么？` | direct |
| `zh:04` | zh | `retrieve_knowledge` | `什么是检索增强生成？` | retrieve |
| `zh:05` | zh | `retrieve_knowledge` | `向量数据库如何支持语义检索？` | retrieve |
| `zh:06` | zh | `retrieve_knowledge` | `BM25 如何给文档排序？` | retrieve |
| `zh:07` | zh | `retrieve_personal_knowledge` | `根据我的笔记，我的项目目标是什么？` | retrieve |
| `zh:08` | zh | `retrieve_personal_knowledge` | `我在 Phase 2 写了什么？` | retrieve |
| `zh:09` | zh | `retrieve_personal_knowledge` | `总结我昨天上传的文档。` | retrieve |
| `zh:10` | zh | `retrieve_mixed_or_ambiguous` | `你好，请解释混合检索。` | retrieve |
| `zh:11` | zh | `retrieve_mixed_or_ambiguous` | `你能根据我的知识库告诉我你知道什么吗？` | retrieve |
| `zh:12` | zh | `retrieve_mixed_or_ambiguous` | `我应该复习学习笔记中的什么内容？` | retrieve |

The JSON root contains only `suite_id`, `version`, and `cases`; each case object contains only the five fields defined in the design.

- [ ] **Step 4: Implement strict local parsing.**

```python
def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _invalid(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


class AgentRouterMiniSuiteLoader:
    def __init__(self, suite_directory: Path = _SUITE_DIRECTORY) -> None:
        self._suite_directory = suite_directory

    def load(self) -> AgentRouterEvaluationSuite:
        manifest = _read_manifest(self._suite_directory / "cases.json")
        return _parse_and_validate_suite(manifest)
```

Follow `infrastructure/xquad_suite.py` for safe text/array/object validation, but do not copy its document, attribution, hash, or network-dataset logic. Convert parser and dataclass `ValueError` into `InvalidEvaluationError` that identifies the field category without echoing question text.

- [ ] **Step 5: Re-run focused tests and commit.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/unit/test_agent_router_suite.py -v`

Expected: PASS.

```bash
git add src/personal_knowledge_agent/evaluation_suites/agent_router_mini/cases.json
git add src/personal_knowledge_agent/infrastructure/agent_router_suite.py
git add tests/unit/test_agent_router_suite.py
git commit -m "feat: add bilingual agent router evaluation suite"
```

## Task 3: Add pure metrics and Router-only orchestration

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/agent_router_metrics.py`
- Create: `src/personal_knowledge_agent/application/evaluate_agent_router.py`
- Create: `tests/unit/test_agent_router_metrics.py`
- Create: `tests/unit/test_evaluate_agent_router.py`

**Interfaces:**
- Consumes: the Task 1 values and existing `QuestionRouter` protocol.
- Produces: `summarize_agent_router_results(results: Sequence[AgentRouterCaseResult]) -> AgentRouterMetricSummary`, `AgentRouterSuiteLoader`, and `AgentRouterEvaluationService.run() -> AgentRouterEvaluationReport`.

- [ ] **Step 1: Write failing metric and service tests.**

```python
def test_summary_exposes_unsafe_direct_and_fallback_separately() -> None:
    summary = summarize_agent_router_results(
        (
            result("retrieve_knowledge", RouteAction.DIRECT, RouteSource.LLM),
            result("retrieve_knowledge", RouteAction.RETRIEVE, RouteSource.FALLBACK),
            result("direct_greeting", RouteAction.DIRECT, RouteSource.RULE),
        )
    )
    assert summary.action_accuracy == pytest.approx(2 / 3)
    assert summary.unsafe_direct_count == 1
    assert summary.fallback_count == 1


def test_service_calls_router_once_per_case_and_never_constructs_answerer() -> None:
    router = RecordingRouter(decisions=(retrieve_decision(), direct_decision()))
    report = AgentRouterEvaluationService(loader_with_two_cases(), router, config()).run()
    assert router.questions == ["first question", "second question"]
    assert len(report.cases) == 2
```

Add direct-precision-zero-denominator coverage, per-language/per-category summaries, order preservation, suite-load-before-route behavior, and propagation of an unexpected router exception.

- [ ] **Step 2: Run the focused tests and confirm failure.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/unit/test_agent_router_metrics.py tests/unit/test_evaluate_agent_router.py -v`

Expected: FAIL because metric and application modules do not exist.

- [ ] **Step 3: Implement deterministic aggregation.**

```python
def summarize_agent_router_results(
    results: Sequence[AgentRouterCaseResult],
) -> AgentRouterMetricSummary:
    if not results:
        raise InvalidEvaluationError("router evaluation requires at least one case")
    expected_retrieve = [item for item in results if item.case.expected_action is RouteAction.RETRIEVE]
    expected_direct = [item for item in results if item.case.expected_action is RouteAction.DIRECT]
    actual_direct = [item for item in results if item.decision.action is RouteAction.DIRECT]
    true_direct = [item for item in actual_direct if item.case.expected_action is RouteAction.DIRECT]
    return AgentRouterMetricSummary(
        case_count=len(results),
        action_accuracy=sum(item.action_matches for item in results) / len(results),
        retrieve_recall=sum(item.decision.action is RouteAction.RETRIEVE for item in expected_retrieve) / len(expected_retrieve),
        direct_recall=len(true_direct) / len(expected_direct),
        direct_precision=(len(true_direct) / len(actual_direct) if actual_direct else None),
        unsafe_direct_count=sum(item.decision.action is RouteAction.DIRECT for item in expected_retrieve),
        fallback_count=sum(item.decision.source is RouteSource.FALLBACK for item in results),
    )
```

Use `sum(bool_values)` only for counts and pass `float` ratios into the validated domain value.

- [ ] **Step 4: Implement service grouping without runtime infrastructure.**

```python
class AgentRouterSuiteLoader(Protocol):
    def load(self) -> AgentRouterEvaluationSuite: ...


def run(self) -> AgentRouterEvaluationReport:
    suite = self._suite_loader.load()
    results = tuple(
        AgentRouterCaseResult(case, self._router.route(case.question))
        for case in suite.cases
    )
    return AgentRouterEvaluationReport(
        suite.suite_id,
        suite.version,
        self._configuration,
        results,
        summarize_agent_router_results(results),
        {language: summarize_agent_router_results(_by_language(results, language)) for language in EvaluationLanguage},
        {category: summarize_agent_router_results(_by_category(results, category)) for category in AgentRouterEvaluationCategory},
    )
```

`_by_language` and `_by_category` return tuples. Do not catch `Exception`; `HybridQuestionRouter` is the sole owner of expected provider/schema fallback conversion.

- [ ] **Step 5: Re-run tests, static types, and commit.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/unit/test_agent_router_metrics.py tests/unit/test_evaluate_agent_router.py -v`

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run mypy src`

Expected: both PASS.

```bash
git add src/personal_knowledge_agent/infrastructure/agent_router_metrics.py
git add src/personal_knowledge_agent/application/evaluate_agent_router.py
git add tests/unit/test_agent_router_metrics.py
git add tests/unit/test_evaluate_agent_router.py
git commit -m "feat: evaluate agent router decisions"
```

## Task 4: Serialize privacy-safe reports atomically

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/agent_router_report.py`
- Create: `tests/unit/test_agent_router_report.py`

**Interfaces:**
- Consumes: `AgentRouterEvaluationReport` and `EvaluationStorageError`.
- Produces: `report_as_json(report: AgentRouterEvaluationReport) -> str` and `write_report(report: AgentRouterEvaluationReport, output: Path) -> None`.

- [ ] **Step 1: Write failing serialization and write-failure tests.**

```python
def test_report_omits_questions_and_raw_model_text() -> None:
    payload = json.loads(report_as_json(report_with_question("private question")))
    rendered = json.dumps(payload, ensure_ascii=False)
    assert "private question" not in rendered
    assert payload["cases"][0]["fallback_reason"] == "invalid_output"


def test_write_report_preserves_existing_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"
    output.write_text("old", encoding="utf-8")
    monkeypatch.setattr(Path, "replace", raise_os_error)
    with pytest.raises(EvaluationStorageError):
        write_report(report(), output)
    assert output.read_text(encoding="utf-8") == "old"
```

Assert the exact schema-version-1 top-level keys, sorted/indented stable bytes, `null` direct precision, and `action_matches` as a JSON Boolean.

- [ ] **Step 2: Run the focused tests and confirm failure.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/unit/test_agent_router_report.py -v`

Expected: FAIL because the report module does not exist.

- [ ] **Step 3: Implement stable serialization and atomic persistence.**

```python
def _case_as_dict(result: AgentRouterCaseResult) -> dict[str, object]:
    return {
        "case_id": result.case.case_id,
        "language": result.case.language.value,
        "category": result.case.category.value,
        "expected_action": result.case.expected_action.value,
        "actual_action": result.decision.action.value,
        "source": result.decision.source.value,
        "fallback_reason": (
            result.decision.fallback_reason.value
            if result.decision.fallback_reason is not None
            else None
        ),
        "action_matches": result.action_matches,
    }
```

Use `json.dumps(..., ensure_ascii=False, sort_keys=True, indent=2) + "\n"`. Copy the established `NamedTemporaryFile`, `fsync`, `Path.replace`, cleanup, and `EvaluationStorageError` pattern from `infrastructure/evaluation_report.py`; do not write question text to the payload.

- [ ] **Step 4: Re-run focused tests and commit.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/unit/test_agent_router_report.py -v`

Expected: PASS.

```bash
git add src/personal_knowledge_agent/infrastructure/agent_router_report.py
git add tests/unit/test_agent_router_report.py
git commit -m "feat: write agent router evaluation reports"
```

## Task 5: Compose the CLI and version the route prompt

**Files:**
- Modify: `src/personal_knowledge_agent/infrastructure/route_prompt_builder.py`
- Modify: `src/personal_knowledge_agent/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `ROUTE_PROMPT_VERSION: Final[str] = "route-v1"`, `_DEFAULT_AGENT_ROUTER_EVALUATION_REPORT`, and `_evaluate_agent_router(output: Path) -> None`.
- Adds: `pka evaluate-agent-router [--output PATH]`.

- [ ] **Step 1: Write failing parser and composition tests.**

```python
def test_evaluate_agent_router_parser_uses_private_default_report() -> None:
    parsed = build_parser().parse_args(["evaluate-agent-router"])
    assert parsed.output == Path(".pka/reports/agent-router-mini-v1.json")


def test_evaluate_agent_router_validates_suite_before_llm_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("personal_knowledge_agent.cli.AgentRouterMiniSuiteLoader.load", raise_invalid_suite)
    result = run_cli("evaluate-agent-router")
    assert result.returncode == 1
    assert "invalid agent-router-mini manifest" in result.stderr
    assert "PKA_LLM_API_KEY" not in result.stderr


def test_evaluate_agent_router_prints_safety_summary_and_report_path(...) -> None:
    result = run_cli("evaluate-agent-router", language_model=scripted_router_model())
    assert "Unsafe direct routes: 0" in result.stdout
    assert "Report: " in result.stdout
```

Also cover explicit output paths, missing LLM configuration after valid fixture preflight, provider-error fallback count, invalid-output fallback count, no report on unexpected failure, and absence of traceback, secret, question text, or raw model output.

- [ ] **Step 2: Run the focused CLI tests and confirm failure.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/test_cli.py -k evaluate_agent_router -v`

Expected: FAIL because the parser subcommand is absent.

- [ ] **Step 3: Add metadata and compose only the Router dependencies.**

```python
# route_prompt_builder.py
ROUTE_PROMPT_VERSION: Final[str] = "route-v1"


# cli.py
_DEFAULT_AGENT_ROUTER_EVALUATION_REPORT = Path(
    ".pka/reports/agent-router-mini-v1.json"
)

def _evaluate_agent_router(output: Path) -> None:
    suite = AgentRouterMiniSuiteLoader().load()
    settings = load_llm_settings(os.environ)
    language_model = OpenAICompatibleChatModel(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model_id=settings.model_id,
    )
    report = AgentRouterEvaluationService(
        _LoadedAgentRouterSuiteLoader(suite),
        HybridQuestionRouter(language_model),
        AgentRouterEvaluationConfiguration(settings.model_id, ROUTE_PROMPT_VERSION),
    ).run()
    write_agent_router_report(report, output)
```

Create `_LoadedAgentRouterSuiteLoader` beside the existing loaded XQuAD wrapper so validation completes before model construction:

```python
class _LoadedAgentRouterSuiteLoader:
    def __init__(self, suite: AgentRouterEvaluationSuite) -> None:
        self._suite = suite

    def load(self) -> AgentRouterEvaluationSuite:
        return self._suite
```

Add the parser and main dispatch branch exactly once; do not add database, search-mode, limit, or judge options:

```python
agent_router_evaluation_parser = subparsers.add_parser(
    "evaluate-agent-router",
    help="Evaluate safe direct-versus-retrieve routing decisions.",
)
agent_router_evaluation_parser.add_argument(
    "--output",
    type=Path,
    default=_DEFAULT_AGENT_ROUTER_EVALUATION_REPORT,
)

# In main(), immediately after the existing `evaluate` branch:
elif parsed.command == "evaluate-agent-router":
    _evaluate_agent_router(parsed.output)
```

Print suite ID/version/case count, action accuracy, retrieve recall, direct recall, optional direct precision, unsafe-direct count, fallback count, and report path. Format an undefined direct precision as `n/a`, not `0.0%`.

- [ ] **Step 4: Re-run focused CLI tests and commit.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/test_cli.py -k evaluate_agent_router -v`

Expected: PASS.

```bash
git add src/personal_knowledge_agent/infrastructure/route_prompt_builder.py
git add src/personal_knowledge_agent/cli.py
git add tests/test_cli.py
git commit -m "feat: add agent router evaluation command"
```

## Task 6: Document the benchmark and validate the full quality gate

**Files:**
- Modify: `README.md`
- Create: `docs/learning-log/phase-8a.md`

**Interfaces:**
- Documents `pka evaluate-agent-router`, its LLM configuration requirement, report location, metrics, and explicit non-goals.

- [ ] **Step 1: Write failing documentation-presence tests.**

```python
def test_readme_documents_agent_router_evaluation_command() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "pka evaluate-agent-router" in readme
    assert "unsafe direct" in readme.lower()
```

Place this assertion in the existing documentation-oriented test module if one already owns README command coverage; otherwise add it to `tests/test_cli.py` near evaluation command tests.

- [ ] **Step 2: Run the documentation test and confirm failure.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/test_cli.py -k agent_router_evaluation_command -v`

Expected: FAIL because README has no Router-evaluation section.

- [ ] **Step 3: Document usage, interpretation, and limits.**

Add a README section near the existing evaluation documentation:

```bash
uv run pka evaluate-agent-router
```

Explain that it reads the fixed bilingual local suite, performs Router decisions only, writes `.pka/reports/agent-router-mini-v1.json`, requires normal `PKA_LLM_*` configuration for LLM-routed cases, and records no question text in the report. Define `unsafe_direct_count` as “expected retrieve, actual direct,” explain why it is safety-critical, and state that the benchmark does not evaluate final answers, evidence, memory, graph, latency, or cost.

Create `docs/learning-log/phase-8a.md` explaining: policy vs generation, why a false direct is asymmetric risk, why fallback count must be separated from action accuracy, how bilingual fixtures detect routing regressions, and why a deterministic labelled suite is preferable to an LLM Judge for this discrete decision.

- [ ] **Step 4: Run targeted and full checks.**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/unit/test_agent_router_evaluation_models.py tests/unit/test_agent_router_suite.py tests/unit/test_agent_router_metrics.py tests/unit/test_evaluate_agent_router.py tests/unit/test_agent_router_report.py tests/test_cli.py -k 'agent_router or documentation' -v`

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest`

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run ruff format --check .`

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run ruff check .`

Run: `TMPDIR=/tmp UV_LINK_MODE=copy uv run mypy src`

Expected: every command PASS. Do not run the live `pka evaluate-agent-router` command during automated verification because it requires configured provider credentials and intentionally exercises a live model.

- [ ] **Step 5: Commit documentation.**

```bash
git add README.md
git add docs/learning-log/phase-8a.md
git add tests/test_cli.py
git commit -m "docs: explain agent router evaluation"
```

## Final Verification and Handoff

- [ ] Confirm `git status --short` contains only user-owned unrelated paths, if any.
- [ ] Confirm no `.pka/` report, database, credential, or raw model response is staged.
- [ ] Summarize the benchmark scope, report location, the exact quality-gate results, and any intentionally unexecuted live-provider check.

# Phase 5B LLM Judge Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Add opt-in, input-isolated LLM Judge scores for relevance, faithfulness, and completeness to the existing Phase 5A document-RAG evaluation command.

**Architecture:** Keep the deterministic Phase 5A flow intact. With \`--judge\`, \`EvaluationService\` calls a Faithfulness Judge with evidence only and a Reference Quality Judge with gold answers only for every \`ANSWERED\` result. Domain values express requests and scores; infrastructure owns prompts, strict JSON parsing, and the Chat Completions adapter; the CLI writes schema v2 only after the complete run succeeds.

**Tech Stack:** Python 3.12+, dataclasses, typing \`Protocol\`, existing OpenAI-compatible Chat Completions adapter, standard-library JSON, pytest, Ruff, mypy, uv.

## Global Constraints

- Add no runtime dependency and keep complete Python type annotations.
- Keep \`domain/\` independent of OpenAI, JSON, SQLite, and filesystem paths.
- Tests use deterministic fakes and no live provider call.
- Without \`--judge\`, the command makes no Judge call and keeps the existing schema-v1 report byte-for-byte.
- Judge only \`AnswerStatus.ANSWERED\` results; refusal quality is outside Phase 5B.
- Any Judge transport or schema error propagates to the safe CLI path and writes no report.
- Faithfulness input has no reference answers; reference-quality input has no retrieved evidence.
- Generated reports and indexes stay under \`.pka/\` and are never committed.

---

## File Structure

| File | Responsibility |
| --- | --- |
| \`domain/evaluation.py\` | Immutable Judge request, result, configuration, summary, and report values. |
| \`domain/protocols.py\` | Replaceable \`FaithfulnessJudge\` and \`ReferenceQualityJudge\` boundaries. |
| \`domain/errors.py\` | Safe invalid-Judge-schema error. |
| \`infrastructure/answer_judge_prompt_builder.py\` | Two labelled data-only prompt sequences. |
| \`infrastructure/structured_answer_judge_parser.py\` | Strict JSON validation for both contracts. |
| \`infrastructure/openai_compatible_answer_judge.py\` | Adapter from existing \`LanguageModel\` to Judge protocols. |
| \`infrastructure/llm_settings.py\` | Optional \`PKA_JUDGE_MODEL\` fallback. |
| \`application/evaluate.py\` | Ordered Judge orchestration and summary creation. |
| \`infrastructure/evaluation_report.py\` | Schema-v1/v2 stable JSON serialization. |
| \`cli.py\` | \`--judge\` composition and terminal summary. |

## Task 1: Define Judge values, protocols, and errors

**Files:**
- Modify: \`src/personal_knowledge_agent/domain/evaluation.py\`
- Modify: \`src/personal_knowledge_agent/domain/protocols.py\`
- Modify: \`src/personal_knowledge_agent/domain/errors.py\`
- Modify: \`tests/unit/test_evaluation_models.py\`

**Interfaces:**
- Produces \`JudgeMetric\`, \`JudgeMetricScore\`, \`FaithfulnessJudgeRequest\`, \`ReferenceQualityJudgeRequest\`, \`ReferenceQualityScores\`, \`JudgeEvaluation\`, \`JudgeConfiguration\`, \`JudgeSummary\`, \`FaithfulnessJudge\`, and \`ReferenceQualityJudge\`.
- \`CaseEvaluation.judge: JudgeEvaluation | None\`; Judge data is valid only when status is \`ANSWERED\`.

- [ ] **Step 1: Write failing validation tests.**

\`\`\`python
def test_judge_metric_score_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="score"):
        JudgeMetricScore(JudgeMetric.FAITHFULNESS, 3, "supported")


def test_faithfulness_request_requires_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        FaithfulnessJudgeRequest("question", "answer", ())


def test_case_evaluation_rejects_judge_for_insufficient_evidence() -> None:
    with pytest.raises(ValueError, match="judge"):
        CaseEvaluation(case(), AnswerStatus.INSUFFICIENT_EVIDENCE, None, (), metrics(), judge())
\`\`\`

- [ ] **Step 2: Verify the tests fail.**

Run: \`uv run pytest tests/unit/test_evaluation_models.py -v\`

Expected: FAIL because \`JudgeMetric\` and related values do not exist.

- [ ] **Step 3: Implement immutable values and protocols.**

\`\`\`python
class JudgeMetric(StrEnum):
    RELEVANCE = "relevance"
    FAITHFULNESS = "faithfulness"
    COMPLETENESS = "completeness"


@dataclass(frozen=True, slots=True)
class JudgeMetricScore:
    metric: JudgeMetric
    score: int
    reason: str

    def __post_init__(self) -> None:
        if self.score not in (0, 1, 2):
            raise ValueError("judge score must be 0, 1, or 2")
        _require_text(self.reason, "judge reason")
        if len(self.reason) > 200:
            raise ValueError("judge reason must contain at most 200 code points")


class FaithfulnessJudge(Protocol):
    def judge_faithfulness(
        self, request: FaithfulnessJudgeRequest
    ) -> JudgeMetricScore: ...


class ReferenceQualityJudge(Protocol):
    def judge_reference_quality(
        self, request: ReferenceQualityJudgeRequest
    ) -> ReferenceQualityScores: ...
\`\`\`

Add \`InvalidJudgeSchemaError(EvaluationFeatureError)\`. Add optional Judge fields
after existing required dataclass fields, and validate that report configuration,
overall Judge summary, and language Judge summaries are either all present or all
absent.

- [ ] **Step 4: Verify focused tests and types.**

Run: \`uv run pytest tests/unit/test_evaluation_models.py -v && uv run mypy src/personal_knowledge_agent/domain\`

Expected: PASS.

- [ ] **Step 5: Commit.**

\`\`\`bash
git add src/personal_knowledge_agent/domain/evaluation.py src/personal_knowledge_agent/domain/protocols.py src/personal_knowledge_agent/domain/errors.py tests/unit/test_evaluation_models.py
git commit -m "feat: define llm judge evaluation contracts"
\`\`\`

## Task 2: Build isolated prompts and strict parsers

**Files:**
- Create: \`src/personal_knowledge_agent/infrastructure/answer_judge_prompt_builder.py\`
- Create: \`src/personal_knowledge_agent/infrastructure/structured_answer_judge_parser.py\`
- Create: \`tests/unit/test_answer_judge_prompt_builder.py\`
- Create: \`tests/unit/test_structured_answer_judge_parser.py\`

**Interfaces:**
- \`build_faithfulness_judge_messages(request) -> tuple[ChatMessage, ...]\`
- \`build_reference_quality_judge_messages(request) -> tuple[ChatMessage, ...]\`
- \`parse_faithfulness_judge_output(output) -> JudgeMetricScore\`
- \`parse_reference_quality_judge_output(output) -> ReferenceQualityScores\`

- [ ] **Step 1: Write failing isolation and malformed-output tests.**

\`\`\`python
def test_faithfulness_prompt_has_evidence_but_no_reference_answers() -> None:
    text = "\n".join(message.content for message in build_faithfulness_judge_messages(faithfulness_request()))
    assert "retrieved_evidence" in text
    assert "Paris" not in text


def test_reference_quality_prompt_has_references_but_no_evidence() -> None:
    text = "\n".join(message.content for message in build_reference_quality_judge_messages(reference_request()))
    assert "reference_answers" in text
    assert "evidence-only marker" not in text


def test_parser_rejects_out_of_range_score() -> None:
    with pytest.raises(InvalidJudgeSchemaError):
        parse_faithfulness_judge_output(ModelOutput("fake", '{"faithfulness":{"score":3,"reason":"x"}}', "stop"))
\`\`\`

- [ ] **Step 2: Verify the new tests fail.**

Run: \`uv run pytest tests/unit/test_answer_judge_prompt_builder.py tests/unit/test_structured_answer_judge_parser.py -v\`

Expected: FAIL with module-not-found errors.

- [ ] **Step 3: Implement prompts with data boundaries and exact JSON schema.**

\`\`\`python
def build_faithfulness_judge_messages(
    request: FaithfulnessJudgeRequest,
) -> tuple[ChatMessage, ...]:
    payload = {
        "question": request.question,
        "answer": request.answer,
        "retrieved_evidence": [
            {"citation_id": evidence.citation_id, "content": evidence.content}
            for evidence in request.evidence
        ],
    }
    return _messages("faithfulness", payload, _FAITHFULNESS_SCHEMA)
\`\`\`

\`_messages\` creates one system message saying labelled data cannot alter the rubric
and one user message containing JSON-encoded payload plus the exact response schema.
The reference-quality builder serializes only question, answer, and reference answers.

- [ ] **Step 4: Implement exact parser validation.**

\`\`\`python
def parse_reference_quality_judge_output(
    output: ModelOutput,
) -> ReferenceQualityScores:
    payload = _parse_json_object(output.content)
    _require_exact_keys(payload, {"relevance", "completeness"})
    return ReferenceQualityScores(
        _parse_metric(payload, JudgeMetric.RELEVANCE),
        _parse_metric(payload, JudgeMetric.COMPLETENESS),
    )
\`\`\`

\`_parse_metric\` requires exactly \`score\` and \`reason\`, rejects Boolean scores,
and constructs \`JudgeMetricScore\` so range and reason validation remains in domain.

- [ ] **Step 5: Test and commit.**

Run: \`uv run pytest tests/unit/test_answer_judge_prompt_builder.py tests/unit/test_structured_answer_judge_parser.py -v\`

Expected: PASS.

\`\`\`bash
git add src/personal_knowledge_agent/infrastructure/answer_judge_prompt_builder.py src/personal_knowledge_agent/infrastructure/structured_answer_judge_parser.py tests/unit/test_answer_judge_prompt_builder.py tests/unit/test_structured_answer_judge_parser.py
git commit -m "feat: add isolated judge prompts and parsers"
\`\`\`

## Task 3: Add the Judge adapter and model-setting override

**Files:**
- Create: \`src/personal_knowledge_agent/infrastructure/openai_compatible_answer_judge.py\`
- Modify: \`src/personal_knowledge_agent/infrastructure/llm_settings.py\`
- Modify: \`tests/unit/test_llm_settings.py\`
- Create: \`tests/unit/test_openai_compatible_answer_judge.py\`

**Interfaces:**
- \`load_judge_llm_settings(environment: Mapping[str, str]) -> LLMSettings\`
- \`OpenAICompatibleAnswerJudge(language_model: LanguageModel)\` implements both Judge protocols.

- [ ] **Step 1: Write failing settings and adapter tests.**

\`\`\`python
def test_judge_settings_use_optional_model_override() -> None:
    settings = load_judge_llm_settings(
        {"PKA_LLM_API_KEY": "key", "PKA_LLM_MODEL": "answer", "PKA_JUDGE_MODEL": "judge"}
    )
    assert settings.model_id == "judge"


def test_adapter_passes_only_faithfulness_messages_to_language_model() -> None:
    model = FakeLanguageModel(ModelOutput("judge", FAITHFUL_JSON, "stop"))
    result = OpenAICompatibleAnswerJudge(model).judge_faithfulness(faithfulness_request())
    assert result.score == 2
    assert "reference_answers" not in "\n".join(item.content for item in model.calls[0])
\`\`\`

- [ ] **Step 2: Verify failure.**

Run: \`uv run pytest tests/unit/test_llm_settings.py tests/unit/test_openai_compatible_answer_judge.py -v\`

Expected: FAIL because the new API is absent.

- [ ] **Step 3: Implement fallback and reuse the existing provider transport.**

\`\`\`python
def load_judge_llm_settings(environment: Mapping[str, str]) -> LLMSettings:
    settings = load_llm_settings(environment)
    return LLMSettings(
        settings.api_key,
        settings.base_url,
        _optional_setting(environment, "PKA_JUDGE_MODEL", settings.model_id),
    )


class OpenAICompatibleAnswerJudge(FaithfulnessJudge, ReferenceQualityJudge):
    def __init__(self, language_model: LanguageModel) -> None:
        self._language_model = language_model

    def judge_faithfulness(self, request: FaithfulnessJudgeRequest) -> JudgeMetricScore:
        output = self._language_model.generate(build_faithfulness_judge_messages(request))
        return parse_faithfulness_judge_output(output)
\`\`\`

Implement \`judge_reference_quality\` with the other builder/parser. Do not duplicate
OpenAI SDK error mapping; \`OpenAICompatibleChatModel\` remains the sole transport adapter.

- [ ] **Step 4: Test and commit.**

Run: \`uv run pytest tests/unit/test_llm_settings.py tests/unit/test_openai_compatible_answer_judge.py -v && uv run mypy src\`

Expected: PASS.

\`\`\`bash
git add src/personal_knowledge_agent/infrastructure/openai_compatible_answer_judge.py src/personal_knowledge_agent/infrastructure/llm_settings.py tests/unit/test_llm_settings.py tests/unit/test_openai_compatible_answer_judge.py
git commit -m "feat: add openai compatible answer judge"
\`\`\`

## Task 4: Orchestrate ordered Judge calls and summaries

**Files:**
- Modify: \`src/personal_knowledge_agent/application/evaluate.py\`
- Modify: \`src/personal_knowledge_agent/infrastructure/evaluation_metrics.py\`
- Modify: \`tests/unit/test_evaluation_service.py\`
- Modify: \`tests/unit/test_evaluation_metrics.py\`

**Interfaces:**
- \`EvaluationService(..., faithfulness_judge=None, reference_quality_judge=None, judge_configuration=None)\`
- \`summarize_judge(evaluations: Sequence[JudgeEvaluation]) -> JudgeSummary\`, with
  a zero-count summary whose three averages are \`None\` when no case is answered.

- [ ] **Step 1: Write failing orchestration and failure-propagation tests.**

\`\`\`python
def test_run_calls_judges_in_case_order(corpus: EvaluationCorpus) -> None:
    calls: list[str] = []
    report = EvaluationService(
        FakeCorpusLoader(corpus), FakeIngester(), answered_pair(),
        FakeFaithfulnessJudge(calls), FakeReferenceQualityJudge(calls), judge_configuration(),
    ).run()
    assert calls == ["faithfulness:case:en", "quality:case:en", "faithfulness:case:zh", "quality:case:zh"]
    assert report.judge_overall_summary is not None


def test_run_stops_when_first_judge_fails(corpus: EvaluationCorpus) -> None:
    with pytest.raises(LanguageModelError, match="unavailable"):
        EvaluationService(
            FakeCorpusLoader(corpus), FakeIngester(), answered_pair(),
            FailingFaithfulnessJudge(), FakeReferenceQualityJudge([]), judge_configuration(),
        ).run()
\`\`\`

- [ ] **Step 2: Verify failure.**

Run: \`uv run pytest tests/unit/test_evaluation_service.py -v\`

Expected: FAIL because the service constructor has no Judge dependencies.

- [ ] **Step 3: Implement the no-Judge-preserving branch.**

\`\`\`python
def _judge_case(self, case: EvaluationCase, answer: GroundedAnswer) -> JudgeEvaluation | None:
    if answer.status is not AnswerStatus.ANSWERED:
        return None
    assert self._faithfulness_judge is not None
    assert self._reference_quality_judge is not None
    faithfulness = self._faithfulness_judge.judge_faithfulness(
        FaithfulnessJudgeRequest(case.question, answer.answer, answer.evidence)
    )
    quality = self._reference_quality_judge.judge_reference_quality(
        ReferenceQualityJudgeRequest(case.question, answer.answer, case.reference_answers)
    )
    return JudgeEvaluation(quality.relevance, faithfulness, quality.completeness)
\`\`\`

Require all three Judge dependencies or none in the constructor. Add
\`summarize_judge\` as normalized macro averages over non-None Judge values. When
no case is answered, it returns a zero-count summary with \`None\` averages; Judge
mode still produces a complete schema-v2 report without inventing refusal scores.

- [ ] **Step 4: Test and commit.**

Run: \`uv run pytest tests/unit/test_evaluation_service.py tests/unit/test_evaluation_metrics.py tests/unit/test_evaluation_models.py -v\`

Expected: PASS.

\`\`\`bash
git add src/personal_knowledge_agent/application/evaluate.py src/personal_knowledge_agent/infrastructure/evaluation_metrics.py tests/unit/test_evaluation_service.py tests/unit/test_evaluation_metrics.py tests/unit/test_evaluation_models.py
git commit -m "feat: score rag evaluations with judges"
\`\`\`

## Task 5: Serialize complete schema-v2 Judge reports

**Files:**
- Modify: \`src/personal_knowledge_agent/infrastructure/evaluation_report.py\`
- Modify: \`tests/unit/test_evaluation_report.py\`

**Interfaces:**
- Schema v1 remains unchanged for a report without \`judge_configuration\`.
- Schema v2 adds \`judge_configuration\`, per-case \`judge\`, and \`judge_summary\`.

- [ ] **Step 1: Write failing report compatibility tests.**

\`\`\`python
def test_no_judge_report_remains_schema_v1() -> None:
    payload = json.loads(report_as_json(phase_5a_report()))
    assert payload["schema_version"] == 1
    assert "judge_configuration" not in payload


def test_judge_report_serializes_schema_v2_and_reasons() -> None:
    payload = json.loads(report_as_json(judge_report()))
    assert payload["schema_version"] == 2
    assert payload["cases"][0]["judge"]["faithfulness"] == {"reason": "supported", "score": 2}
\`\`\`

- [ ] **Step 2: Verify failure.**

Run: \`uv run pytest tests/unit/test_evaluation_report.py -v\`

Expected: FAIL because the serializer always writes schema version 1.

- [ ] **Step 3: Implement conditional versioned serialization.**

\`\`\`python
def report_as_json(report: EvaluationReport) -> str:
    payload = _phase_5a_payload(report)
    if report.judge_configuration is not None:
        payload["schema_version"] = 2
        payload["judge_configuration"] = _judge_configuration_as_dict(report)
        payload["judge_summary"] = _judge_summary_as_dict(report)
        for item, evaluation in zip(payload["cases"], report.cases, strict=True):
            if evaluation.judge is not None:
                item["judge"] = _judge_evaluation_as_dict(evaluation.judge)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
\`\`\`

Preserve sorted keys, case order, UTF-8, final newline, and atomic \`write_report\`.
Report validation guarantees schema-v2 case Judge values are never null.

- [ ] **Step 4: Test and commit.**

Run: \`uv run pytest tests/unit/test_evaluation_report.py -v\`

Expected: PASS.

\`\`\`bash
git add src/personal_knowledge_agent/infrastructure/evaluation_report.py tests/unit/test_evaluation_report.py
git commit -m "feat: write versioned judge evaluation reports"
\`\`\`

## Task 6: Add explicit CLI composition and safe failure behavior

**Files:**
- Modify: \`src/personal_knowledge_agent/cli.py\`
- Modify: \`tests/test_cli.py\`

**Interfaces:**
- Parser adds \`evaluate --judge\` with default \`False\`.
- \`_evaluate(limit, mode, output, judge)\` constructs Judge dependencies only when \`judge\` is true.

- [ ] **Step 1: Write failing CLI tests.**

\`\`\`python
def test_evaluate_parser_defaults_judge_to_false() -> None:
    assert build_parser().parse_args(["evaluate"]).judge is False


def test_evaluate_judge_prints_summary_and_writes_schema_v2(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = run_cli("evaluate", "--judge", "--output", str(output), judge=scripted_judge())
    assert result.returncode == 0
    assert "Judge faithfulness:" in result.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 2


def test_evaluate_judge_failure_writes_no_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = run_cli("evaluate", "--judge", "--output", str(output), judge=FailingJudge())
    assert result.returncode == 1
    assert not output.exists()
\`\`\`

- [ ] **Step 2: Verify failure.**

Run: \`uv run pytest tests/test_cli.py -k evaluate -v\`

Expected: FAIL with \`unrecognized arguments: --judge\`.

- [ ] **Step 3: Add parser flag and opt-in composition.**

\`\`\`python
evaluate_parser.add_argument(
    "--judge",
    action="store_true",
    help="Score answered cases with the configured LLM Judge.",
)
\`\`\`

Load normal answer settings always. Load Judge settings only when the flag is true,
construct a second \`OpenAICompatibleChatModel\` using its resolved model ID, wrap it
in one \`OpenAICompatibleAnswerJudge\`, and pass it into both Judge protocol slots
alongside \`JudgeConfiguration\`. Preserve fixture validation before database reset
and all model construction. Call \`write_report\` only after \`service.run\` returns,
then print three Judge percentages only if a Judge summary exists.

- [ ] **Step 4: Test and commit.**

Run: \`uv run pytest tests/test_cli.py -k evaluate -v\`

Expected: PASS.

\`\`\`bash
git add src/personal_knowledge_agent/cli.py tests/test_cli.py
git commit -m "feat: add llm judge evaluation command"
\`\`\`

## Task 7: Document Phase 5B and run the quality gate

**Files:**
- Modify: \`README.md\`
- Modify: \`AGENTS.md\`
- Create: \`docs/learning-log/phase-5b.md\`

**Interfaces:**
- Documents \`pka evaluate --mode hybrid --limit 5 --judge\`, \`PKA_JUDGE_MODEL\`, 48-call cost ceiling, input isolation, and failure behavior.

- [ ] **Step 1: Update documentation with the exact invocation and boundaries.**

Add the command below to README and state that \`PKA_JUDGE_MODEL\` falls back to
\`PKA_LLM_MODEL\`.

\`\`\`bash
uv run pka evaluate --mode hybrid --limit 5 --judge
\`\`\`

Explain that a complete XQuAD-mini run performs up to 48 extra live model calls,
faithfulness sees only evidence, relevance/completeness see only references, Phase 5A
reports remain schema v1, failure writes no report, and Agent/Memory/Graph/refusal
evaluation remains future work. Record the same final scope in AGENTS and explain
LLM Judge trade-offs, structured outputs, injection-resistant data boundaries, cost,
and model bias in the Phase 5B learning log.

- [ ] **Step 2: Run the complete quality gate.**

Run:

\`\`\`bash
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run mypy src
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pka health
\`\`\`

Expected: all tests pass, Ruff has no findings, mypy has no errors, and health prints
\`Personal Knowledge Agent is healthy.\`

- [ ] **Step 3: Inspect scope and commit documentation.**

Run: \`git diff --check && git status --short\`

Expected: only Phase 5B files appear and no whitespace error is reported.

\`\`\`bash
git add README.md AGENTS.md docs/learning-log/phase-5b.md
git commit -m "docs: explain llm judge evaluation"
\`\`\`

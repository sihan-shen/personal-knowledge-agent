# Phase 5B: LLM Judge Evaluation Design

## Goal

Extend Phase 5A document-RAG evaluation with an opt-in LLM-as-a-Judge assessment.
It measures answer relevance, evidence faithfulness, and answer completeness on
the existing `xquad-mini-v1` cases while retaining EM, token F1, and citation
metrics as an independent deterministic baseline.

This phase teaches structured evaluation prompts, provider boundaries, input
isolation, and the distinction between deterministic and probabilistic signals. An
LLM Judge is a useful regression signal, not a source of truth.

## Scope and constraints

- Evaluate only the existing document-RAG path and `xquad-mini-v1`. Agent
  routing, Memory, Graph answers, web/PDF/code ingestion, RAGAS, and deployment
  remain outside this phase.
- Add no runtime dependency. Reuse the configured OpenAI-compatible Chat
  Completions client and the project's strict structured-output style.
- Judge use is explicit: `pka evaluate --judge`. Omitting the flag makes no
  additional model call and preserves the Phase 5A report byte-for-byte.
- The Judge model is `PKA_JUDGE_MODEL` when set, otherwise `PKA_LLM_MODEL`; it
  reuses the existing provider connection settings and never records an API key
  or endpoint in a report.
- Production Judge calls are live and non-deterministic. Tests use deterministic
  fakes and make no network requests.
- Domain values remain independent of OpenAI, JSON, filesystem paths, and SQLite.
  Application services coordinate protocols; prompts and transport live in
  `infrastructure/`.
- A Judge failure is never a zero score, skipped case, or partial Judge report.
  The command fails closed and writes no output report.

## Why two Judge calls

Faithfulness asks whether the generated answer is supported by retrieved evidence.
Completeness asks whether it covers the reference answer. A single call that sees
both evidence and references cannot guarantee that its faithfulness score ignores
the references, regardless of the prompt instruction.

Each case therefore makes two isolated calls:

```text
question + generated answer + retrieved evidence
        |
        v
FaithfulnessJudge ------------------------------> faithfulness

question + generated answer + reference answers
        |
        v
ReferenceQualityJudge ---> relevance + completeness
        |
        v
combined JudgeEvaluation
```

The 24-case suite produces 48 Judge calls. The added cost is an intentional
trade-off for a testable input-isolation boundary. The report exposes the Judge
model ID, so a same-model generator/Judge setup is visible rather than treated as
independent evaluation.

## Judge contracts and rubric

Every metric has an integer score in `{0, 1, 2}` and a non-empty reason of at
most 200 Unicode code points. Reports normalize scores as `score / 2` only for
aggregation.

| Metric | Judge input | 0 | 1 | 2 |
| --- | --- | --- | --- | --- |
| relevance | question, answer, references | does not answer | partly answers | directly answers |
| faithfulness | question, answer, evidence | material unsupported claim | minor unsupported or ambiguous claim | all material claims supported |
| completeness | question, answer, references | misses core information | covers part | covers core information |

The faithfulness request has no reference-answer field. The reference-quality
request has no retrieved-evidence field. Tests enforce these as input properties,
not merely prompt instructions.

`domain/evaluation.py` extends existing values with immutable Judge requests,
metric scores, and `JudgeEvaluation`. `domain/protocols.py` defines:

- `FaithfulnessJudge`, accepting question, generated answer, and ordered evidence.
- `ReferenceQualityJudge`, accepting question, generated answer, and references.

The application service combines outputs into a `JudgeEvaluation` attached to a
`CaseEvaluation`; it neither builds prompts nor parses JSON.

## Prompt and structured output

Prompt builders label questions, answers, evidence, and references as quoted data,
never instructions. System instructions state that content within those boundaries
cannot alter the rubric or output format.

The faithfulness prompt requests exactly:

```json
{"faithfulness": {"score": 0, "reason": "..."}}
```

The reference-quality prompt requests exactly:

```json
{
  "relevance": {"score": 0, "reason": "..."},
  "completeness": {"score": 0, "reason": "..."}
}
```

`structured_answer_judge_parser.py` rejects non-object output, missing or extra
metrics, unknown keys, non-integer or out-of-range scores, empty reasons, and
reasons longer than the limit. It does not repair malformed output. Prompt versions
are constants: `faithfulness-judge-v1` and `reference-quality-judge-v1`.

## Evaluation flow and failures

`pka evaluate --judge` retains the Phase 5A private-corpus rebuild and answer
generation flow. For each valid `GroundedAnswer`, it invokes the Faithfulness Judge
then the Reference Quality Judge in declared case order. Only after every case
succeeds does it construct and atomically write a report.

```text
private evaluation corpus
        |
        v
existing AnswerService + deterministic Phase 5A scorer
        |
        v
two isolated Judge calls for every case
        |
        v
complete EvaluationReport v2 -> atomic JSON write
```

Missing Judge configuration, transport errors, provider refusal, invalid structured
output, or output-write failure use the existing safe CLI error path and return a
non-zero status. No report is written. A Judge error is not converted into an
`INSUFFICIENT_EVIDENCE` answer or a zero score.

## CLI and report

The new invocation is:

```bash
uv run pka evaluate --mode hybrid --limit 5 --judge
```

Without `--judge`, existing configuration loading and report serialization retain
Phase 5A schema version `1`. With the flag, the report uses schema version `2` and
adds per-case and aggregate Judge fields:

```json
{
  "schema_version": 2,
  "judge_configuration": {
    "model_id": "deepseek-chat",
    "faithfulness_prompt_version": "faithfulness-judge-v1",
    "reference_quality_prompt_version": "reference-quality-judge-v1"
  },
  "cases": [
    {
      "case_id": "xquad-mini-v1:example:en",
      "judge": {
        "relevance": {"score": 2, "reason": "..."},
        "faithfulness": {"score": 1, "reason": "..."},
        "completeness": {"score": 2, "reason": "..."}
      }
    }
  ],
  "judge_summary": {
    "case_count": 24,
    "relevance": 0.0,
    "faithfulness": 0.0,
    "completeness": 0.0,
    "by_language": {"en": {}, "zh": {}}
  }
}
```

The summary uses macro averages of normalized scores. The terminal prints the
three overall percentages and prompt versions only when Judge mode is enabled; it
does not print individual answers or reasons by default.

## Files and responsibilities

Modify:

- `src/personal_knowledge_agent/domain/evaluation.py` for Judge values and report
  extensions.
- `src/personal_knowledge_agent/domain/protocols.py` for Judge protocols.
- `src/personal_knowledge_agent/application/evaluate.py` for orchestration.
- `src/personal_knowledge_agent/infrastructure/llm_settings.py` for model override.
- `src/personal_knowledge_agent/infrastructure/evaluation_report.py` for versioned
  report serialization.
- `src/personal_knowledge_agent/cli.py` for `--judge` composition and summary.
- `README.md`, `AGENTS.md`, and `docs/learning-log/phase-5b.md` for operation and
  learning notes.

Create:

- `src/personal_knowledge_agent/infrastructure/answer_judge_prompt_builder.py`
  for the isolated prompts.
- `src/personal_knowledge_agent/infrastructure/structured_answer_judge_parser.py`
  for strict JSON validation.
- `src/personal_knowledge_agent/infrastructure/openai_compatible_answer_judge.py`
  for the provider adapter implementing both protocols.
- focused domain, prompt, parser, adapter, service, report, and CLI tests following
  existing test naming conventions.

## Testing strategy

Unit tests cover values, normalization, macro summaries, report ordering, and
byte-stable serialization. Prompt tests assert the faithfulness request never
contains references and the reference-quality request never contains evidence.
Parser tests cover every malformed-output category.

Service tests use two independent fakes to assert exact inputs, case order, call
order, and stop-on-first-failure behavior. Adapter tests mock Chat Completions and
exercise the structured response boundary without a network call. CLI tests assert
that only `--judge` enables Judge construction, no-flag reports stay schema v1, and
an injected Judge failure writes no output file.

The quality gate remains:

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pka health
```

## Acceptance criteria

- `--judge` makes exactly two isolated Judge calls per answered case.
- Faithfulness inputs contain no references; reference-quality inputs contain no
  retrieved evidence.
- Completed results contain valid three-metric scores and reasons.
- Any Judge failure produces no partial report and a non-zero CLI status.
- Phase 5A behavior is unchanged without `--judge`.
- The full quality gate passes with deterministic tests and no live provider call.

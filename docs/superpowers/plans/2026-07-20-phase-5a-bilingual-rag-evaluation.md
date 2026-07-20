# Phase 5A: Bilingual RAG Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add pka evaluate, a reproducible bilingual XQuAD-mini benchmark that rebuilds an isolated corpus, calls the existing document RAG answer service, scores deterministic answer/citation metrics, and writes a stable JSON report.

**Architecture:** Package 24 XQuAD-derived public fixture documents. An evaluation service ingests them into a private SQLite index, calls the existing GroundedQuestionAnswerer in fixed order, and delegates scoring/JSON formatting to pure infrastructure functions. The CLI only composes existing components; it never accepts a user knowledge database.

**Tech Stack:** Python 3.12+, standard-library json/unicodedata/collections, existing MixedLanguageTokenizer, SQLite, E5, Chat Completions, pytest, Ruff, mypy.

## Global Constraints

- Keep domain values independent of filesystem, SQLite, provider code, JSON, and package resources.
- Use complete annotations, frozen slotted dataclasses, and strict mypy-compatible code.
- Add no runtime dependency, framework, RAGAS, pandas, or LLM judge.
- Reuse AnswerService, IngestService, MarkdownLoader, MarkdownChunker, SQLiteKeywordIndex, E5Embedder, and the existing OpenAI-compatible adapter.
- The command only uses .pka/evaluations/xquad-mini-v1/knowledge.db. It never reads or writes user knowledge, memory, or graph databases.
- Default reports stay under .pka/reports/; generated database/report artifacts are untracked.
- Tests use packaged fixtures plus deterministic fakes and never make a network request.
- Prefix project commands with TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache.

## File Structure

| Path | Responsibility |
|---|---|
| src/personal_knowledge_agent/domain/evaluation.py | Validated evaluation values. |
| src/personal_knowledge_agent/domain/errors.py | Evaluation-specific KnowledgeError subclasses. |
| src/personal_knowledge_agent/application/evaluate.py | Filesystem-aware corpus boundary, controlled ingestion, answering, citation mapping, report assembly. |
| src/personal_knowledge_agent/infrastructure/evaluation_metrics.py | Pure normalization, EM, F1, citation metrics, macro summary. |
| src/personal_knowledge_agent/infrastructure/evaluation_report.py | Stable JSON and atomic report writer. |
| src/personal_knowledge_agent/infrastructure/xquad_suite.py | Packaged-suite loader and validator. |
| src/personal_knowledge_agent/evaluation_suites/xquad_mini/ | 24 documents, manifest, attribution, license. |
| src/personal_knowledge_agent/cli.py | evaluate parser/composition/summary. |
| tests/unit/test_evaluation_models.py | Domain behavior. |
| tests/unit/test_evaluation_metrics.py | Metric behavior. |
| tests/unit/test_evaluation_report.py | Serialization/writer behavior. |
| tests/unit/test_xquad_suite.py | Fixture and loader behavior. |
| tests/unit/test_evaluation_service.py | Orchestration with fakes. |
| tests/test_cli.py | CLI process contract. |

---

### Task 1: Define domain values and evaluation errors

**Files:**
- Create: src/personal_knowledge_agent/domain/evaluation.py
- Modify: src/personal_knowledge_agent/domain/errors.py
- Create: src/personal_knowledge_agent/application/evaluate.py
- Test: tests/unit/test_evaluation_models.py

**Interfaces:**
- Consumes: AnswerStatus, GroundedAnswer, and SearchMode from domain.models.
- Produces: EvaluationLanguage, EvaluationDocument, EvaluationCase, EvaluationSuite, CaseMetrics, CaseEvaluation, MetricSummary, EvaluationReport, EvaluationFeatureError, InvalidEvaluationError, EvaluationStorageError, and the application-layer EvaluationCorpus/EvaluationCorpusLoader boundary.

- [ ] **Step 1: Write failing validation tests**

~~~python
def test_case_rejects_empty_reference_answers() -> None:
    with pytest.raises(ValueError, match="reference_answers"):
        EvaluationCase("case:en", EvaluationLanguage.EN, "question", (), "doc:en")


def test_report_requires_english_and_chinese_summaries() -> None:
    with pytest.raises(ValueError, match="language summaries"):
        EvaluationReport("xquad-mini", "v1", SearchMode.HYBRID, 5, (), summary, {})
~~~

Cover blank IDs/questions, duplicate document IDs, a gold ID absent from the suite, duplicate cited logical IDs, non-finite/out-of-range metrics, and a limit outside 1..10.

- [ ] **Step 2: Verify failure**

Run:

~~~bash
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/test_evaluation_models.py -v
~~~

Expected: FAIL because domain.evaluation does not exist.

- [ ] **Step 3: Implement values and protocol**

Place `EvaluationLanguage`, `EvaluationCase`, `EvaluationDocument`, and
`EvaluationSuite` from the following block in `domain/evaluation.py`. Place only
`EvaluationCorpus` and `EvaluationCorpusLoader` in `application/evaluate.py`;
they are intentionally not domain values because they contain local paths.

~~~python
class EvaluationLanguage(StrEnum):
    EN = "en"
    ZH = "zh"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    language: EvaluationLanguage
    question: str
    reference_answers: tuple[str, ...]
    gold_document_id: str


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    document_id: str
    resource_name: str


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    suite_id: str
    version: str
    documents: tuple[EvaluationDocument, ...]
    cases: tuple[EvaluationCase, ...]
    attribution: str


@dataclass(frozen=True, slots=True)
class EvaluationCorpus:
    suite: EvaluationSuite
    document_paths: Mapping[str, Path]


class EvaluationCorpusLoader(Protocol):
    def load(self) -> EvaluationCorpus:
        ...
~~~

Define EvaluationCorpus and EvaluationCorpusLoader in application/evaluate.py, not domain/evaluation.py: the corpus pairs an EvaluationSuite with an exact document-ID-to-existing-Path mapping, so it is an application filesystem boundary. CaseMetrics contains four finite floats in [0.0, 1.0]. CaseEvaluation preserves case, answer status, optional model ID, first-seen unique cited document IDs, and CaseMetrics. MetricSummary has a positive count plus four bounded metrics. EvaluationReport preserves ordered cases, mode, limit, overall summary, and exactly EN/ZH summaries.

Add EvaluationFeatureError(KnowledgeError), InvalidEvaluationError(EvaluationFeatureError), and EvaluationStorageError(EvaluationFeatureError).

- [ ] **Step 4: Verify pass**

Run the Task 1 pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/personal_knowledge_agent/domain/evaluation.py src/personal_knowledge_agent/domain/errors.py src/personal_knowledge_agent/application/evaluate.py tests/unit/test_evaluation_models.py
git commit -m "feat: define rag evaluation values"
~~~

### Task 2: Implement deterministic answer and citation scoring

**Files:**
- Create: src/personal_knowledge_agent/infrastructure/evaluation_metrics.py
- Test: tests/unit/test_evaluation_metrics.py

**Interfaces:**
- Consumes: EvaluationCase, GroundedAnswer, AnswerStatus, and MixedLanguageTokenizer.
- Produces: score_case(case, answer, cited_document_ids) -> CaseMetrics and summarize(metrics) -> MetricSummary.

- [ ] **Step 1: Write failing metric tests**

~~~python
def test_score_case_removes_only_document_citations_before_exact_match() -> None:
    answer = answered("Ｐａｒｉｓ [S1]")
    case = case_with_answers("Paris", "City of Paris")

    assert score_case(case, answer, ("gold-doc",)) == CaseMetrics(1.0, 1.0, 1.0, 1.0)


def test_score_case_uses_chinese_bigram_overlap_and_best_alias() -> None:
    answer = answered("检索增强生成 [S1]")
    case = case_with_answers("错误答案", "检索增强生成")

    assert score_case(case, answer, ("gold-doc",)).answer_token_f1 == 1.0


def test_score_case_penalizes_wrong_and_missing_citations() -> None:
    assert score_case(case, answered("answer [S1]"), ("other-doc",)).citation_precision == 0.0
    assert score_case(case, answered("answer [S1]"), ()).citation_recall == 0.0
~~~

Also test repeated cited IDs, case/punctuation variation, no overlap, insufficient evidence, empty aggregate rejection, and two-case macro averages.

- [ ] **Step 2: Verify failure**

Run:

~~~bash
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/test_evaluation_metrics.py -v
~~~

Expected: FAIL because evaluation_metrics does not exist.

- [ ] **Step 3: Implement pure metrics**

Use a compiled pattern for only [S<positive integer>] and then NFKC, casefold, punctuation removal, and whitespace collapse. Use Counter intersection over MixedLanguageTokenizer tokens. Compare every reference alias and retain maximum F1. Deduplicate cited logical document IDs before computing precision and recall. Return all zeros for INSUFFICIENT_EVIDENCE. Reject empty summary inputs with InvalidEvaluationError and use math.fsum for macro metrics.

~~~python
_DOCUMENT_CITATION = re.compile(r"\[S[1-9]\d*\]")


def score_case(
    case: EvaluationCase,
    answer: GroundedAnswer,
    cited_document_ids: tuple[str, ...],
) -> CaseMetrics:
    if answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        return CaseMetrics(0.0, 0.0, 0.0, 0.0)
    # derive answer EM/F1 and citation precision/recall exactly as specified above
~~~

- [ ] **Step 4: Verify pass**

Run the Task 2 pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/personal_knowledge_agent/infrastructure/evaluation_metrics.py tests/unit/test_evaluation_metrics.py
git commit -m "feat: score rag evaluation cases"
~~~

### Task 3: Add stable JSON report serialization

**Files:**
- Create: src/personal_knowledge_agent/infrastructure/evaluation_report.py
- Test: tests/unit/test_evaluation_report.py

**Interfaces:**
- Consumes: EvaluationReport.
- Produces: report_as_json(report) -> str and write_report(report, output) -> None.

- [ ] **Step 1: Write failing report tests**

~~~python
def test_report_as_json_is_byte_stable_and_uses_logical_ids_only() -> None:
    assert report_as_json(report()) == report_as_json(report())
    payload = json.loads(report_as_json(report()))
    assert payload["cases"][0]["cited_document_ids"] == ["gold-doc"]


def test_write_report_wraps_filesystem_errors() -> None:
    with pytest.raises(EvaluationStorageError, match="report"):
        write_report(report(), Path("/missing-parent/report.json"))
~~~

Assert sorted object keys, preserved case order, UTF-8 final newline, no absolute source path, and cleanup of failed temporary writes.

- [ ] **Step 2: Verify failure**

Run:

~~~bash
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/test_evaluation_report.py -v
~~~

Expected: FAIL because evaluation_report does not exist.

- [ ] **Step 3: Implement stable writer**

Create primitive dict/list data only, use enum .value, json.dumps(ensure_ascii=False, sort_keys=True, indent=2), and append one newline. Write to NamedTemporaryFile in output.parent, flush and fsync, then Path.replace(output). Catch OSError, delete a created temporary file with missing_ok=True, and raise EvaluationStorageError("Could not write evaluation report"). Do not serialize timestamps, raw answers, retrieval scores, absolute paths, or credentials.

- [ ] **Step 4: Verify pass**

Run the Task 3 pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/personal_knowledge_agent/infrastructure/evaluation_report.py tests/unit/test_evaluation_report.py
git commit -m "feat: write stable evaluation reports"
~~~

### Task 4: Package and validate XQuAD-mini-v1

**Files:**
- Create: src/personal_knowledge_agent/infrastructure/xquad_suite.py
- Create: src/personal_knowledge_agent/evaluation_suites/xquad_mini/cases.json
- Create: src/personal_knowledge_agent/evaluation_suites/xquad_mini/ATTRIBUTION.md
- Create: src/personal_knowledge_agent/evaluation_suites/xquad_mini/LICENSE.txt
- Create: 24 Markdown documents below src/personal_knowledge_agent/evaluation_suites/xquad_mini/documents/
- Modify: pyproject.toml only if uv build omits non-Python resources
- Test: tests/unit/test_xquad_suite.py

**Interfaces:**
- Consumes: Task 1 values and package resources.
- Produces: XQuadMiniSuiteLoader.load() -> EvaluationCorpus.

- [ ] **Step 1: Write failing loader tests**

~~~python
def test_xquad_mini_loads_a_balanced_bilingual_public_corpus() -> None:
    corpus = XQuadMiniSuiteLoader().load()

    assert (corpus.suite.suite_id, corpus.suite.version) == ("xquad-mini", "v1")
    assert len(corpus.suite.cases) == 24
    assert Counter(case.language for case in corpus.suite.cases) == {
        EvaluationLanguage.EN: 12,
        EvaluationLanguage.ZH: 12,
    }
    assert len(corpus.document_paths) == 24
~~~

Also assert order/IDs/aliases, complete attribution, all paths are files, every gold document exists, malformed JSON rejects, and each Markdown file is one title plus one context.

- [ ] **Step 2: Verify failure**

Run:

~~~bash
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/test_xquad_suite.py -v
~~~

Expected: FAIL because loader/resources do not exist.

- [ ] **Step 3: Derive and package deterministic fixture data**

Download pinned xquad.en.json and xquad.zh.json to /tmp from the exact upstream revision recorded in ATTRIBUTION.md; never commit the full upstream files. Traverse English questions in original order. Retain an ID only when Chinese has the same ID, both answer arrays are non-empty, and each primary answer occurs in its language context. Stop at 12 IDs.

For each retained ID/language create a document containing exactly:

~~~markdown
# xquad-mini-v1:<question-id>:<language>

<matching XQuAD context>
~~~

cases.json includes suite_id xquad-mini, version v1, attribution, 24 document entries with logical ID/resource name, and 24 cases with question, original-order distinct answers, language, and same-language gold ID. ATTRIBUTION.md includes source URLs, pinned revision, both SHA-256 checksums, XQuAD citation, derivative-fixture notice, and CC BY-SA 4.0 attribution. LICENSE.txt contains CC BY-SA 4.0 text.

Implement the loader with a package-relative Path, json.loads, explicit values, and InvalidEvaluationError for every malformed invariant. Add Hatch include configuration only if uv build proves package resources are omitted.

- [ ] **Step 4: Verify suite and wheel**

Run:

~~~bash
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/test_xquad_suite.py -v
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv build
~~~

Expected: PASS; built wheel contains cases.json and a fixture Markdown file.

- [ ] **Step 5: Commit**

~~~bash
git add src/personal_knowledge_agent/infrastructure/xquad_suite.py src/personal_knowledge_agent/evaluation_suites pyproject.toml tests/unit/test_xquad_suite.py
git commit -m "feat: package bilingual rag evaluation suite"
~~~

### Task 5: Orchestrate private ingestion, answers, and scoring

**Files:**
- Modify: src/personal_knowledge_agent/application/evaluate.py
- Test: tests/unit/test_evaluation_service.py

**Interfaces:**
- Consumes: EvaluationCorpusLoader, IngestService, GroundedQuestionAnswerer, score_case, summarize, SearchMode.
- Produces: EvaluationService.run(*, mode=SearchMode.HYBRID, limit=5) -> EvaluationReport.

- [ ] **Step 1: Write failing application tests**

~~~python
def test_run_ingests_documents_then_answers_cases_in_manifest_order() -> None:
    report = service.run(mode=SearchMode.SEMANTIC, limit=3)

    assert ingester.paths == [english_path, chinese_path]
    assert answerer.calls == [
        (english_question, 3, SearchMode.SEMANTIC),
        (chinese_question, 3, SearchMode.SEMANTIC),
    ]
    assert [item.case.case_id for item in report.cases] == ["case:en", "case:zh"]
~~~

Also test evidence-source-to-logical-ID mapping, no citations, insufficient evidence all zeros, unknown source rejection, answerer exception propagation without a result, and invalid limit before dependency calls.

- [ ] **Step 2: Verify failure**

Run:

~~~bash
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/test_evaluation_service.py -v
~~~

Expected: FAIL because EvaluationService does not exist.

- [ ] **Step 3: Implement small orchestration**

Load corpus, build a resolved-path-to-logical-ID map, ingest corpus.document_paths in declared document order, call answerer.answer(case.question, limit=limit, mode=mode) exactly once per declared case, select only evidence cited by answer.citation_ids, resolve each Evidence.source, map it to a logical ID, preserve first-seen citation order, then call score_case. Make overall and per-language MetricSummary values with summarize. Raise InvalidEvaluationError if an evidence source is not a suite document. Do not inspect retrieval rank or score.

- [ ] **Step 4: Verify pass**

Run the Task 5 pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/personal_knowledge_agent/application/evaluate.py tests/unit/test_evaluation_service.py
git commit -m "feat: run isolated rag evaluations"
~~~

### Task 6: Add pka evaluate and CLI tests

**Files:**
- Modify: src/personal_knowledge_agent/cli.py
- Modify: tests/test_cli.py

**Interfaces:**
- Consumes: Tasks 4–5, existing settings/model/index/ingestion components, write_report.
- Produces: pka evaluate [--mode MODE] [--limit 1..10] [--output PATH].

- [ ] **Step 1: Write failing CLI tests**

~~~python
def test_evaluate_uses_a_private_database_and_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = run_cli("evaluate", "--output", str(output), language_model=scripted_24_case_model())

    assert result.returncode == 0
    assert "Suite: xquad-mini v1" in result.stdout
    assert "Answer token F1:" in result.stdout
    assert output.is_file()


def test_evaluate_rejects_invalid_limit_before_runtime_construction() -> None:
    result = run_cli(
        "evaluate", "--limit", "11",
        forbid_model_construction=True, forbid_embedder_construction=True,
    )
    assert result.returncode == 2
    assert "between 1 and 10" in result.stderr
~~~

Add mode forwarding, default report location, no --db argument, missing LLM configuration, model failure with no report, and safe report-writer error tests.

- [ ] **Step 2: Verify failure**

Run:

~~~bash
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli.py -k evaluate -v
~~~

Expected: FAIL because evaluate is not a command.

- [ ] **Step 3: Implement parser, composition root, and summary**

Add constants:

~~~python
_DEFAULT_EVALUATION_DATABASE = Path(".pka/evaluations/xquad-mini-v1/knowledge.db")
_DEFAULT_EVALUATION_REPORT = Path(".pka/reports/xquad-mini-v1.json")
~~~

The parser uses existing _answer_limit and SearchMode choices. It has --output but no --db. _evaluate loads settings before provider construction, builds exactly one E5Embedder, SQLiteKeywordIndex on _DEFAULT_EVALUATION_DATABASE, IngestService, AnswerService, XQuadMiniSuiteLoader, and EvaluationService. It writes after a successful run and prints suite/version/case count, four overall percentages, English and Chinese token F1, and requested report path. Dispatch from main and rely on the existing KnowledgeError safe exit. Do not print source paths, fixture text, credentials, or raw provider detail.

- [ ] **Step 4: Verify pass**

Run the Task 6 pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/personal_knowledge_agent/cli.py tests/test_cli.py
git commit -m "feat: add rag evaluation command"
~~~

### Task 7: Document the feature and run final verification

**Files:**
- Modify: README.md
- Modify: AGENTS.md
- Create: docs/learning-log/phase-5a.md
- Modify: tests/test_cli.py

**Interfaces:**
- Consumes: completed CLI contract and suite/metric behavior.
- Produces: reproducible user documentation and portfolio learning log.

- [ ] **Step 1: Add a failing documented-summary behavior test**

Assert the documented command prints the suite/version, four aggregate metric labels, two language token-F1 labels, and output report path using fakes.

- [ ] **Step 2: Verify failure**

Run:

~~~bash
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli.py -k evaluation_summary -v
~~~

Expected: FAIL until the CLI summary exactly matches the documented contract.

- [ ] **Step 3: Write focused documentation**

README adds an Evaluate document RAG section with command, API requirement, private paths, XQuAD attribution link, metric definitions, and limitations of extractive EM/F1. AGENTS marks Phase 5A complete only after verification and leaves broader evaluation future work. The learning log explains gold data, reproducibility, EM/F1, citation precision/recall, deterministic local scoring versus live generation, and suite isolation from private knowledge.

- [ ] **Step 4: Run the complete quality gate**

Run:

~~~bash
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run mypy src
TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pka health
git diff --check
~~~

Expected: all tests pass; Ruff and mypy report no issues; health is healthy; diff check is silent.

- [ ] **Step 5: Commit documentation**

~~~bash
git add README.md AGENTS.md docs/learning-log/phase-5a.md tests/test_cli.py
git commit -m "docs: explain rag evaluation workflow"
~~~

## Plan self-review

- Spec coverage: Tasks 1–2 provide values and all four metrics; Task 3 provides stable/atomic reporting; Task 4 provides a licensed bilingual fixture; Task 5 isolates/reuses RAG; Task 6 provides explicit CLI behavior; Task 7 documents and verifies the result.
- Placeholder scan: every task names paths, interfaces, test behavior, implementation rules, commands, and a narrow commit.
- Type consistency: EvaluationCorpusLoader.load returns EvaluationCorpus; EvaluationService.run returns EvaluationReport; score_case returns CaseMetrics; summarize returns MetricSummary; write_report accepts EvaluationReport.

## Execution Handoff

Plan complete and saved to docs/superpowers/plans/2026-07-20-phase-5a-bilingual-rag-evaluation.md. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

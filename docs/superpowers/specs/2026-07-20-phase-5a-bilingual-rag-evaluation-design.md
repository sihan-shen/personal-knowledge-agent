# Phase 5A: Bilingual RAG Evaluation Design

## Goal

Build a small, repeatable evaluation command for the document-only RAG path. The
command runs the existing `AnswerService` over a controlled bilingual corpus,
scores each answer locally, and writes a machine-readable report. It provides a
portfolio-quality baseline for measuring future retrieval and prompting changes.

Phase 5A evaluates only `pka ask`-equivalent document RAG behavior. It does not
evaluate the Agent router, long-term memory, graph answers, web/PDF/code ingestion,
LLM-as-a-Judge, RAGAS, or deployment.

## Why this scope

The project already validates that an individual answer uses only citations from
the retrieved evidence. That is a safety boundary, not a quality measurement. A
fixed suite adds an answer-quality regression signal without mixing all later
subsystems into one uninformative score.

The suite uses a bilingual subset of XQuAD. XQuAD supplies professionally
translated English and Chinese extractive question-answer pairs based on the
SQuAD 1.1 development data, and the upstream repository distributes it under CC
BY-SA 4.0. The exact source URL, source revision, checksum, attribution, and
license text will accompany the derived fixture.

## Constraints

- Python 3.12+ with strict mypy; use complete type annotations.
- Add no runtime dependency. The evaluation implementation uses the standard
  library and the project's existing mixed tokenizer.
- The domain layer remains independent of SQLite, OpenAI, filesystem paths, and
  JSON serialization.
- Core metrics are deterministic and run locally. A live evaluation still uses
  the configured Chat Completions model to obtain each `GroundedAnswer`.
- Unit and CLI tests use deterministic fakes and never call a network service.
- The command must not read or mutate a user's knowledge database, Memory
  database, or graph database.
- Generated evaluation indexes and default reports remain under `.pka/` and are
  never committed.
- Existing `ask`, `agent`, Memory, and Graph behavior is unchanged.

## Evaluation suite

### `xquad-mini-v1`

The repository contains 12 aligned XQuAD question IDs in both English and
Chinese, producing 24 evaluation cases and 24 language-specific Markdown
documents. A selected English question and its selected Chinese counterpart have
the same upstream question ID, but each points to its own translated document and
answer aliases.

Selection is deterministic:

1. Load the pinned English and Chinese XQuAD JSON files.
2. Walk English questions in their original file order.
3. Retain a question only when the Chinese file has the same question ID, both
   answer arrays are non-empty, and every retained primary answer is a substring
   of its language-specific context.
4. Stop after 12 retained IDs.

The implementation commits the resulting derived Markdown documents and a
`cases.json` manifest. Tests only use those committed files; they do not download
XQuAD. The suite attribution file records the upstream URL, commit or release
identifier, SHA-256 values of the two source JSON files, and the CC BY-SA 4.0
notice.

Each source document has a logical ID such as `xquad-mini-v1:<question-id>:en`.
The logical ID, not an absolute resource path, is the gold evidence target and the
only document identifier stored in an evaluation report.

Each case has this conceptual shape:

```json
{
  "case_id": "xquad-mini-v1:<question-id>:zh",
  "language": "zh",
  "question": "...",
  "reference_answers": ["...", "..."],
  "gold_document_id": "xquad-mini-v1:<question-id>:zh"
}
```

The fixture documents are packaged with the application so the installed CLI can
load them. Every Markdown file contains one title derived from its logical ID and
the single upstream context paragraph; it does not contain answer annotations.

## Command and data flow

The CLI adds one command:

```bash
uv run pka evaluate \
  --mode hybrid \
  --limit 5 \
  --output .pka/reports/xquad-mini-v1.json
```

`--mode` has the existing `SearchMode` choices and defaults to `hybrid`.
`--limit` uses the existing answer limit validation of 1 through 10 and defaults
to 5. `--output` defaults to `.pka/reports/xquad-mini-v1.json`.

There is deliberately no `--db` option. Each invocation ingests the packaged
fixture documents into the private database
`.pka/evaluations/xquad-mini-v1/knowledge.db`, then uses that index for every
case. Re-ingestion replaces each fixture document through the current atomic
index boundary, so no user document can become evidence and no stale fixture
chunk survives a fixture update.

```text
packaged xquad-mini-v1 suite
        |
        v
IngestService + SQLiteKeywordIndex + E5Embedder
        |
        v
private evaluation knowledge.db
        |
        v
AnswerService (existing retrieval, prompt, model, citation validation)
        |
        v
EvaluationService + deterministic metric functions
        |
        v
stable JSON report and concise terminal summary
```

The live model receives only the public evaluation question and the retrieved
public fixture chunks. API credentials, absolute local paths, vector values, and
the original user knowledge base remain local.

## Domain values and boundaries

`domain/evaluation.py` defines immutable values:

- `EvaluationLanguage`: `en` and `zh`.
- `EvaluationCase`: stable ID, language, question, non-empty reference answers,
  and gold document ID.
- `EvaluationSuite`: suite ID, version, ordered documents, ordered cases, and
  attribution metadata.
- `CaseMetrics`: answer exact-match boolean, answer token F1, citation precision,
  and citation recall.
- `CaseEvaluation`: one case, returned answer status/model ID, cited logical
  document IDs, and metrics.
- `MetricSummary`: a case count plus macro averages for the four metrics.
- `EvaluationReport`: suite identity, requested search configuration, ordered
  case evaluations, overall summary, and summaries keyed by language.

`infrastructure/xquad_suite.py` owns package-resource access. It implements the
application-layer `EvaluationCorpusLoader` and returns its `EvaluationCorpus`,
which pairs the pure `EvaluationSuite` with resolved local fixture paths. It
validates the manifest before any model or index is
constructed: 24 cases, 12 of each language, unique case IDs, unique document IDs,
every gold document present, non-empty answer aliases, and complete attribution.

`application/evaluate.py` owns the filesystem-aware `EvaluationCorpus`/
`EvaluationCorpusLoader` boundary and orchestration. It accepts the existing ingestion
and grounded-answer boundaries plus a suite loader and metric scorer. It ingests
the suite documents in declared order, calls `answer(question, limit, mode)` once
per case in declared order, maps returned citation IDs through `Evidence.source`
to logical document IDs, and returns an `EvaluationReport`.

## Deterministic scoring

`infrastructure/evaluation_metrics.py` is a pure implementation with no I/O.

Before answer scoring, remove only valid inline document citation tokens matching
`[S<positive integer>]`. Then normalize using Unicode NFKC, Unicode case-folding,
whitespace collapse, and punctuation removal. Do not remove ordinary bracketed
text or Memory/Graph citation families, because those are outside this suite.

Answer exact match is `1.0` when the normalized answer equals any normalized
reference answer; otherwise it is `0.0`.

Answer token F1 uses the project's mixed Chinese/English tokenizer and multiset
token overlap. Compute F1 against every reference answer and retain the maximum.
If either normalized token sequence is empty, F1 is zero unless both normalized
answers are exactly equal, which is already handled by exact match.

Citation scoring uses unique cited logical document IDs, not request-local labels:

- citation precision = cited IDs matching the one gold document / all cited IDs;
  it is zero for an empty citation set;
- citation recall = `1.0` if the gold document appears in the cited set, otherwise
  `0.0`.

An `INSUFFICIENT_EVIDENCE` answer is a valid RAG outcome, but receives zero for
all four metrics. A provider error, model transport error, malformed structured
answer, or fixture/index failure is not converted into a partial score: the CLI
prints the existing safe error category, exits non-zero, and writes no report.

Each summary is an arithmetic macro average across cases. The report contains an
overall summary and separate `en` and `zh` summaries. The fixed balanced suite
makes the overall macro average the mean of the two language means.

## Report format

`infrastructure/evaluation_report.py` serializes a report with `json.dumps`,
sorted object keys, UTF-8, a final newline, and deterministic case order. The
report does not contain a timestamp, API key, absolute path, raw embedding, or
retrieval score.

The report schema has a fixed `schema_version` of `1` and contains:

```json
{
  "schema_version": 1,
  "suite": {"id": "xquad-mini", "version": "v1"},
  "configuration": {"mode": "hybrid", "limit": 5},
  "cases": [
    {
      "case_id": "xquad-mini-v1:example:en",
      "language": "en",
      "status": "answered",
      "model_id": "provider-model",
      "cited_document_ids": ["xquad-mini-v1:example:en"],
      "metrics": {
        "answer_exact_match": 1.0,
        "answer_token_f1": 1.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0
      }
    }
  ],
  "summary": {
    "overall": {"case_count": 24, "answer_exact_match": 0.0,
      "answer_token_f1": 0.0, "citation_precision": 0.0,
      "citation_recall": 0.0},
    "by_language": {"en": {}, "zh": {}}
  }
}
```

The terminal prints the suite/version, case count, four overall percentages,
English and Chinese answer F1, and the local report path. It does not print each
question or model answer by default.

## Files

Create:

- `src/personal_knowledge_agent/domain/evaluation.py`
- `src/personal_knowledge_agent/application/evaluate.py`
- `src/personal_knowledge_agent/infrastructure/evaluation_metrics.py`
- `src/personal_knowledge_agent/infrastructure/evaluation_report.py`
- `src/personal_knowledge_agent/infrastructure/xquad_suite.py`
- `src/personal_knowledge_agent/evaluation_suites/xquad_mini/cases.json`
- `src/personal_knowledge_agent/evaluation_suites/xquad_mini/ATTRIBUTION.md`
- `src/personal_knowledge_agent/evaluation_suites/xquad_mini/LICENSE.txt`
- the 24 selected Markdown fixture documents beneath
  `src/personal_knowledge_agent/evaluation_suites/xquad_mini/documents/`
- `tests/unit/test_evaluation_models.py`
- `tests/unit/test_evaluation_metrics.py`
- `tests/unit/test_evaluation_report.py`
- `tests/unit/test_xquad_suite.py`
- `tests/unit/test_evaluation_service.py`

Modify:

- `src/personal_knowledge_agent/cli.py`
- `pyproject.toml` only if needed to include non-Python package resources in the
  built distribution
- `tests/test_cli.py`
- `README.md`
- `AGENTS.md`
- `docs/learning-log/phase-5a.md`

## Testing strategy

Unit tests cover value validation, suite manifest validation, NFKC/case/punctuation
normalization, citation stripping without accidental bracket removal, English and
Chinese token F1, alias selection, empty outputs, duplicate citations, correct
citation, incorrect citation, missing citation, insufficient evidence, macro
averages, and byte-stable JSON.

Application-service tests use a fake answerer and fake ingestion boundary to prove
document ingestion order, case execution order, mode/limit forwarding, source-path
to logical-ID mapping, and no partial report after a raised answer error.

CLI tests patch `E5Embedder` and `OpenAICompatibleChatModel`, use a temporary
evaluation root and output path, verify the command never receives the normal
knowledge database, and assert safe non-zero behavior for missing model
configuration and output write errors.

The full quality gate after implementation is:

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pka health
```

## Acceptance criteria

- `pka evaluate` rebuilds and queries only the named suite's private index.
- A complete report has exactly 24 ordered cases: 12 English and 12 Chinese.
- The four metrics are deterministic for identical `GroundedAnswer` values.
- The report is byte-stable for identical input values.
- Live failures never create a misleading partial report.
- Existing commands retain their current behavior and full quality gate passes.

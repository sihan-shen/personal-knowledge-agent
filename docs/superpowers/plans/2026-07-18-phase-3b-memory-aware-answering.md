# Phase 3B Memory-Aware Answering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieve relevant confirmed active memories for non-fallback Agent answers, inject at most three as request-local `M` context, and validate visible mixed `S`/`M` citations without changing `pka ask`.

**Architecture:** Add a dedicated `MemoryAwareAnswerService` beside the existing document-only `AnswerService`. A local `MemoryRecallService` uses existing SQLite hybrid search, an absolute relevance gate, slot-atomic budgeting, and defense-in-depth safety checks; `AgentService` lazily constructs this path only for a validated non-fallback retrieve route. New prompt and parser modules keep document and memory context separated while returning the existing grounded-answer shape plus a typed recall report.

**Tech Stack:** Python 3.12+, dataclasses and protocols, SQLite, `multilingual-e5-small`, exact cosine similarity, RRF, OpenAI-compatible Chat Completions, pytest, Ruff, strict mypy, uv.

## Global Constraints

- Implement the approved design at `docs/superpowers/specs/2026-07-18-phase-3b-memory-aware-answering-design.md`; do not add Phase 4 or later capabilities.
- Work in an isolated worktree created at execution time through `using-git-worktrees`; preserve main-workspace `papers/`, `references/`, and `research/`.
- Use Python 3.12+ features, complete annotations, strict mypy, deterministic fakes, and a failing behavioral test before every production change.
- Keep `domain/` independent of infrastructure libraries and keep SQLite, tokenization, ranking, prompts, and parsing in `infrastructure/`.
- Do not add dependencies or change the SQLite schema/version.
- Keep `pka ask`, direct templates, fallback retrieval, and Phase 3A lifecycle/proposal behavior compatible unless this plan explicitly changes an Agent retrieve path.
- Direct and fallback routes must not construct or open a memory store; the route model must never receive active memory.
- Recall only `active` memories, search at most 12 candidates, inject at most 3, and use the fixed semantic threshold `0.84`.
- Provider-visible memory fields are exactly `citation_id` and `content`; never send IDs, type/key, hashes, timestamps, scores, embeddings, database paths, or credentials.
- Use the selected `--memory-db` for both Phase 3B recall and the existing Phase 3A post-answer proposal flow.
- On this mounted workspace, prefix uv commands with `TMPDIR=/tmp UV_LINK_MODE=copy`; use a worktree-specific `UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3b-venv` and `UV_CACHE_DIR=/tmp/uv-cache` during execution.
- Before merge, run `uv run pytest`, `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy src`, and `uv run pka health` with the required environment.

## Planned file structure

### New files

- `src/personal_knowledge_agent/application/memory_recall.py` — application coordination for conservative, slot-atomic recall and safe warnings.
- `src/personal_knowledge_agent/application/memory_aware_answer.py` — document/memory answer orchestration and the local no-context refusal.
- `src/personal_knowledge_agent/infrastructure/memory_relevance.py` — deterministic bilingual lexical gate plus the fixed cosine threshold.
- `src/personal_knowledge_agent/infrastructure/memory_aware_prompt_builder.py` — separated provider payload and memory-aware system policy.
- `src/personal_knowledge_agent/infrastructure/structured_contextual_answer_parser.py` — strict JSON and mixed `S`/`M` citation validation.
- `tests/unit/test_memory_relevance.py` — relevance-policy boundary tests.
- `tests/unit/test_memory_recall_service.py` — recall, safety, warning, slot, and budget tests.
- `tests/unit/test_memory_aware_prompt_builder.py` — prompt policy and disclosure tests.
- `tests/unit/test_structured_contextual_answer_parser.py` — exact schema and citation tests.
- `tests/unit/test_memory_aware_answer_service.py` — document/memory quadrant and degradation tests.
- `docs/learning-log/phase-3b.md` — portfolio learning record.

### Modified files

- `src/personal_knowledge_agent/domain/memory.py` — memory evidence, recall report/warnings, and semantic signal.
- `src/personal_knowledge_agent/domain/models.py` — memory-aware answer result and Agent response recall metadata.
- `src/personal_knowledge_agent/domain/protocols.py` — relevance, recaller, and two answerer boundaries.
- `src/personal_knowledge_agent/infrastructure/sqlite_memory_store.py` — retain per-candidate cosine values without schema changes.
- `src/personal_knowledge_agent/application/agent.py` — choose lazy memory-aware versus document-only answerer.
- `src/personal_knowledge_agent/cli.py` — composition, warning rendering, and full cited-memory display.
- Existing domain, helper, application, SQLite, Agent, and CLI tests — compatibility and fixture updates.
- `README.md` and `AGENTS.md` — Phase 3B usage, privacy, costs, limitations, and roadmap status.

---

### Task 1: Define request-local memory recall contracts

**Files:**
- Modify: `src/personal_knowledge_agent/domain/memory.py`
- Modify: `src/personal_knowledge_agent/domain/models.py`
- Modify: `src/personal_knowledge_agent/domain/protocols.py`
- Modify: `tests/unit/test_memory_models.py`
- Modify: `tests/unit/test_models.py`
- Modify: `tests/unit/test_helpers.py`
- Modify: `tests/unit/test_memory_admin_service.py`

**Interfaces:**
- Consumes: existing `MemoryRecord`, `MemorySearchResult`, `GroundedAnswer`, `SearchMode`, and `MemoryStore`.
- Produces: `MAX_RECALLED_MEMORIES`, `MemoryRecallWarning`, `MemoryEvidence`, `MemoryRecallReport`, `MemoryAwareAnswerResult`, `MemoryRelevancePolicy`, `MemoryRecaller`, `GroundedQuestionAnswerer`, and `MemoryAwareQuestionAnswerer`.
- Changes: `MemorySearchResult(memory, score, semantic_similarity)` and `AgentResponse(..., memory_recall=None)`.

- [ ] **Step 1: Write failing domain contract tests**

Add tests that construct one active record and require:

```python
evidence = MemoryEvidence("M1", record, 0.75)
report = MemoryRecallReport((evidence,), ())

assert evidence.memory is record
assert report.memories == (evidence,)
assert MemoryRecallReport((), (MemoryRecallWarning.STORAGE_ERROR,)).warnings == (
    MemoryRecallWarning.STORAGE_ERROR,
)
```

Add parametrized failures for blank/non-`M` labels, non-finite scores, more than
three memories, non-sequential labels, duplicate warnings, and semantic similarity
outside `[-1.0, 1.0]`. Add `AgentResponse` tests requiring no recall report for
direct/fallback and a report for a non-fallback retrieve response.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_models.py tests/unit/test_models.py -q
```

Expected: FAIL because the recall types and `semantic_similarity` field do not exist.

- [ ] **Step 3: Add the domain values and invariants**

Implement in `domain/memory.py`:

```python
MAX_RECALLED_MEMORIES = 3
_MEMORY_CITATION_PATTERN = re.compile(r"M[1-9]\d*")


class MemoryRecallWarning(StrEnum):
    EMBEDDING_ERROR = "embedding_error"
    STORAGE_ERROR = "storage_error"
    UNSAFE_MEMORY = "unsafe_memory"
    AMBIGUOUS_MEMORY_SLOT = "ambiguous_memory_slot"


@dataclass(frozen=True, slots=True)
class MemoryEvidence:
    citation_id: str
    memory: MemoryRecord
    retrieval_score: float

    def __post_init__(self) -> None:
        if _MEMORY_CITATION_PATTERN.fullmatch(self.citation_id) is None:
            raise ValueError("memory citation_id must use the M request-label format")
        if not isfinite(self.retrieval_score):
            raise ValueError("memory retrieval_score must be finite")


@dataclass(frozen=True, slots=True)
class MemoryRecallReport:
    memories: tuple[MemoryEvidence, ...]
    warnings: tuple[MemoryRecallWarning, ...]

    def __post_init__(self) -> None:
        if len(self.memories) > MAX_RECALLED_MEMORIES:
            raise ValueError("memory recall exceeds the request limit")
        expected = tuple(f"M{index}" for index in range(1, len(self.memories) + 1))
        if tuple(item.citation_id for item in self.memories) != expected:
            raise ValueError("memory citations must be sequential request labels")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("memory recall warnings must be unique")
```

Extend `MemorySearchResult` with `semantic_similarity: float = 0.0`, validate it is
finite and in `[-1.0, 1.0]`, and update relevance-sensitive fixtures to pass an
explicit value. The compatibility default keeps Task 1 independently buildable;
Task 2 integration tests require SQLite to replace it with the real cosine.

- [ ] **Step 4: Add answer result and protocol boundaries**

In `domain/models.py`, import `MemoryRecallReport`, define:

```python
@dataclass(frozen=True, slots=True)
class MemoryAwareAnswerResult:
    answer: GroundedAnswer
    recall: MemoryRecallReport
```

Add `memory_recall: MemoryRecallReport | None = None` to `AgentResponse` and reject
recall reports on direct and fallback responses. Preserve the temporarily valid
non-fallback/no-report combination until Task 8 updates `AgentService` atomically;
Task 8 then adds the final required-report invariant. In `domain/protocols.py`, add:

```python
class MemoryRelevancePolicy(Protocol):
    def allows(self, query: str, result: MemorySearchResult) -> bool: ...


class MemoryRecaller(Protocol):
    def recall(self, query: str, *, limit: int = 3) -> MemoryRecallReport: ...


class GroundedQuestionAnswerer(Protocol):
    def answer(
        self,
        question: str,
        *,
        limit: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> GroundedAnswer: ...


class MemoryAwareQuestionAnswerer(Protocol):
    def answer(
        self,
        question: str,
        *,
        limit: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> MemoryAwareAnswerResult: ...
```

- [ ] **Step 5: Run focused and compatibility tests**

Run the Task 1 files plus `tests/unit/test_memory_admin_service.py` and
`tests/unit/test_helpers.py`. Expected: PASS, with every existing
`MemorySearchResult` fixture updated explicitly.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/personal_knowledge_agent/domain/memory.py src/personal_knowledge_agent/domain/models.py src/personal_knowledge_agent/domain/protocols.py tests/unit/test_memory_models.py tests/unit/test_models.py tests/unit/test_helpers.py tests/unit/test_memory_admin_service.py
git commit -m "feat: define memory recall contracts"
```

---

### Task 2: Preserve semantic similarity in SQLite hybrid results

**Files:**
- Modify: `src/personal_knowledge_agent/infrastructure/sqlite_memory_store.py`
- Modify: `tests/integration/test_sqlite_memory_store.py`

**Interfaces:**
- Consumes: Task 1 `MemorySearchResult(memory, score, semantic_similarity)`.
- Produces: every returned active-memory result includes its actual normalized-query cosine value while retaining existing RRF order.

- [ ] **Step 1: Write failing integration tests**

Use deterministic fake embeddings to create lexical-only, semantic-first, and mixed
candidates. Assert:

```python
results = store.search("agent goal", limit=3)

assert [item.memory.memory_id for item in results] == expected_rrf_order
assert results[0].semantic_similarity == pytest.approx(expected_cosine)
assert all(-1.0 <= item.semantic_similarity <= 1.0 for item in results)
```

Also snapshot all four memory tables before and after search and assert search does
not rewrite rows or require a schema-version change.

- [ ] **Step 2: Run the new SQLite tests and verify RED**

Expected: FAIL because the adapter currently discards cosine values and constructs
the old two-field result.

- [ ] **Step 3: Retain ranking and score maps separately**

Replace `_semantic_memory_ids` with a helper returning both values:

```python
def _semantic_memory_ranking(
    self,
    connection: sqlite3.Connection,
    query: str,
    *,
    memory_type: MemoryType | None,
    limit: int,
) -> tuple[list[str], dict[str, float]]:
    ...
    scores[memory_id] = cosine
    ranked = sorted(
        ((memory_id, score) for memory_id, score in scores.items() if score > 0),
        key=lambda item: (-item[1], item[0]),
    )
    return [memory_id for memory_id, _ in ranked[:limit]], scores
```

Use only the positive sorted IDs for existing semantic ranking, but retain every
active row's finite cosine in the score map. Construct fused results with:

```python
MemorySearchResult(
    memory=self._record_by_id(connection, memory_id),
    score=rrf_score,
    semantic_similarity=semantic_scores[memory_id],
)
```

- [ ] **Step 4: Run SQLite integration and Phase 3A search tests**

Run `tests/integration/test_sqlite_memory_store.py`, relevant CLI memory-search
tests, Ruff on the modified files, and mypy. Expected: PASS and unchanged
administrative order/output.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/personal_knowledge_agent/infrastructure/sqlite_memory_store.py tests/integration/test_sqlite_memory_store.py
git commit -m "feat: expose memory semantic relevance"
```

---

### Task 3: Implement the conservative bilingual relevance policy

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/memory_relevance.py`
- Create: `tests/unit/test_memory_relevance.py`

**Interfaces:**
- Consumes: `MemoryRelevancePolicy`, `MemorySearchResult`, and `MixedLanguageTokenizer`.
- Produces: `ConservativeMemoryRelevancePolicy.allows(query, result)` and `MIN_MEMORY_COSINE_SIMILARITY = 0.84`.

- [ ] **Step 1: Write table-driven failing relevance tests**

Cover English and Chinese descriptor matches, key separators, stopword-only overlap,
and cosine boundary behavior:

```python
@pytest.mark.parametrize(
    ("query", "memory_key", "content", "similarity", "expected"),
    [
        ("What is my learning goal?", "learning_goal", "Master agents", 0.70, True),
        ("How should you answer me?", "answer_style", "Be concise", 0.70, True),
        ("what is my", "unrelated", "Prefer dark mode", 0.8399, False),
        ("完全无关的问题", "answer_style", "简洁回答", 0.84, True),
        ("完全无关的问题", "answer_style", "简洁回答", 0.8399, False),
    ],
)
def test_conservative_relevance_gate(...): ...
```

Add explicit dot/underscore/hyphen key normalization and blank-query rejection.

- [ ] **Step 2: Run the file and verify RED**

Expected: import failure because `memory_relevance.py` does not exist.

- [ ] **Step 3: Implement the minimal policy**

Define the exact constants from the spec, normalize key separators with
`re.sub(r"[._-]+", " ", key)`, tokenize
`f"{memory_type.value} {normalized_key} {content}"`, case-normalize stopwords, and
return:

```python
meaningful_overlap = bool(query_tokens & memory_tokens)
return meaningful_overlap or (
    result.semantic_similarity >= MIN_MEMORY_COSINE_SIMILARITY
)
```

Reject a blank query with `InvalidQueryError`; do not load an embedding model or
make a network call in this policy.

- [ ] **Step 4: Run tests, Ruff, and mypy**

Expected: all relevance tests pass, including exact `0.84`, and the module is fully
typed.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/personal_knowledge_agent/infrastructure/memory_relevance.py tests/unit/test_memory_relevance.py
git commit -m "feat: gate memory recall relevance"
```

---

### Task 4: Coordinate safe slot-atomic memory recall

**Files:**
- Create: `src/personal_knowledge_agent/application/memory_recall.py`
- Create: `tests/unit/test_memory_recall_service.py`
- Modify: `tests/helpers.py`

**Interfaces:**
- Consumes: `MemoryStore.search`, `MemoryStore.list_memories`, `MemoryRelevancePolicy`, `MemorySafetyPolicy`, Task 1 reports, and Task 3 policy behavior.
- Produces: `MemoryRecallService.recall(query, limit=3) -> MemoryRecallReport`.

- [ ] **Step 1: Add deterministic fakes and failing behavior tests**

Create fakes that record search/list/safety/relevance calls. Test:

- search limit is exactly 12;
- only eligible candidates pass;
- labels are `M1...Mn` in fused candidate order;
- all active same-slot peers are included atomically in created-at/ID order;
- a slot with four peers is skipped with `AMBIGUOUS_MEMORY_SLOT`;
- a two-item slot that cannot fit the remaining budget is skipped whole;
- unsafe candidate or peer skips the whole slot with `UNSAFE_MEMORY`;
- `EmbeddingError` and `MemoryStorageError` return no memories and one safe warning;
- an `InvalidMemoryError` caused by no searchable tokens returns a normal empty report;
- list failure discards any partially considered candidates;
- limit values outside `1..3` fail before dependencies.

- [ ] **Step 2: Run the file and verify RED**

Expected: import failure because `MemoryRecallService` does not exist.

- [ ] **Step 3: Implement recall with no partial disclosure on infrastructure failure**

Implement constants and the service skeleton:

```python
MEMORY_RECALL_CANDIDATE_LIMIT = 12


class MemoryRecallService:
    def __init__(self, store, relevance_policy, safety_policy) -> None: ...

    def recall(self, query: str, *, limit: int = 3) -> MemoryRecallReport:
        if not query.strip():
            raise InvalidQueryError("Memory recall query must not be blank")
        if not 1 <= limit <= MAX_RECALLED_MEMORIES:
            raise InvalidQueryError("Memory recall limit must be between 1 and 3")
        try:
            candidates = self._store.search(
                query,
                limit=MEMORY_RECALL_CANDIDATE_LIMIT,
            )
            if not candidates:
                return MemoryRecallReport((), ())
            active = self._store.list_memories(status=MemoryStatus.ACTIVE)
        except EmbeddingError:
            return MemoryRecallReport((), (MemoryRecallWarning.EMBEDDING_ERROR,))
        except MemoryStorageError:
            return MemoryRecallReport((), (MemoryRecallWarning.STORAGE_ERROR,))
        except InvalidMemoryError:
            return MemoryRecallReport((), ())
        return self._select(query, candidates, active, limit=limit)
```

Build a slot map from the complete active snapshot. Consider each slot once, apply
relevance to its ranked representative, safety-screen type/key/content for every
peer, and never split a slot. Deduplicate warnings in enum declaration order before
constructing the report.

- [ ] **Step 4: Verify GREEN and service isolation**

Run the new unit file and the Phase 3A memory service/model files. Expected: PASS;
no SQLite or E5 object is needed by unit tests.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/personal_knowledge_agent/application/memory_recall.py tests/unit/test_memory_recall_service.py tests/helpers.py
git commit -m "feat: recall safe active memory context"
```

---

### Task 5: Build the separated memory-aware prompt

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/memory_aware_prompt_builder.py`
- Create: `tests/unit/test_memory_aware_prompt_builder.py`

**Interfaces:**
- Consumes: `Evidence`, `MemoryEvidence`, and `ChatMessage`.
- Produces: `build_memory_aware_messages(question, evidence, memories) -> tuple[ChatMessage, ChatMessage]`.

- [ ] **Step 1: Write failing prompt and disclosure tests**

Test S-only, M-only, mixed, and hostile text. Parse the two payload sections back
from the user message and require exact allowlists:

```python
assert set(document_payload[0]) == {"citation_id", "heading_path", "text"}
assert set(memory_payload[0]) == {"citation_id", "content"}
```

Use sentinels for local memory ID, type/key, hash, timestamp, score, vector, database
path, and API key and assert none occur in outgoing messages. Assert the system
message contains untrusted-data, external-fact, user-specific-memory, conflict,
sensitive-inference, citation, and insufficient-context rules. Reject blank question
and empty-both contexts.

- [ ] **Step 2: Run the file and verify RED**

Expected: import failure because the prompt builder does not exist.

- [ ] **Step 3: Implement exact JSON serialization**

Build two arrays with `json.dumps(..., ensure_ascii=False)` and one user message:

```python
user_content = (
    f"QUESTION:\n{question}\n\n"
    f"DOCUMENT_EVIDENCE_JSON:\n{serialized_evidence}\n\n"
    f"MEMORY_CONTEXT_JSON:\n{serialized_memories}"
)
```

Serialize documents exactly like the existing grounded builder and memories with
only `citation_id` and `memory.content`. Keep all policy text in the system message,
not in data arrays.

- [ ] **Step 4: Run tests, Ruff, and mypy**

Expected: PASS with byte-for-byte disclosure assertions.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/personal_knowledge_agent/infrastructure/memory_aware_prompt_builder.py tests/unit/test_memory_aware_prompt_builder.py
git commit -m "feat: build separated memory answer prompts"
```

---

### Task 6: Parse strict mixed document and memory citations

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/structured_contextual_answer_parser.py`
- Create: `tests/unit/test_structured_contextual_answer_parser.py`

**Interfaces:**
- Consumes: `ModelOutput`, document `Evidence`, and `MemoryEvidence`.
- Produces: `parse_contextual_answer(output, evidence, memories) -> GroundedAnswer`.

- [ ] **Step 1: Write failing parser tests**

Cover valid S-only, M-only, mixed, and insufficient answers. Reject:

- invalid JSON, Markdown fences, non-object JSON, duplicate JSON keys, extra/missing fields;
- wrong field types, blank answer, unknown status;
- empty citations for answered output;
- duplicate list labels, unknown S/M labels, `S0`, `M0`, malformed labels;
- body/list mismatch and first-use order mismatch;
- citations on insufficient output.

Require the returned `GroundedAnswer.evidence` and `model_id` to remain intact.

- [ ] **Step 2: Run the file and verify RED**

Expected: import failure because the contextual parser does not exist.

- [ ] **Step 3: Implement strict decoding and validation**

Use an `object_pairs_hook` that raises `InvalidAnswerSchemaError` on a repeated key,
the exact required field set, and:

```python
_CITATION_PATTERN = re.compile(r"\[((?:S|M)[1-9]\d*)\]")
known_ids = {
    *(item.citation_id for item in evidence),
    *(item.citation_id for item in memories),
}
```

Match the existing answer parser's status and first-use validation while allowing
only request-known positive `S`/`M` labels. Do not modify or broaden
`parse_grounded_answer`.

- [ ] **Step 4: Run both parser suites**

Run the new parser file and `tests/unit/test_structured_answer_parser.py`. Expected:
PASS and the old parser continues rejecting `M` labels.

- [ ] **Step 5: Commit Task 6**

```bash
git add src/personal_knowledge_agent/infrastructure/structured_contextual_answer_parser.py tests/unit/test_structured_contextual_answer_parser.py
git commit -m "feat: validate mixed answer citations"
```

---

### Task 7: Coordinate document and memory-aware answering

**Files:**
- Create: `src/personal_knowledge_agent/application/memory_aware_answer.py`
- Create: `tests/unit/test_memory_aware_answer_service.py`

**Interfaces:**
- Consumes: `SearchService`, `MemoryRecaller`, `LanguageModel`, `build_evidence`, Task 5 prompt builder, and Task 6 parser.
- Produces: `MemoryAwareAnswerService.answer(...) -> MemoryAwareAnswerResult`.

- [ ] **Step 1: Write failing quadrant and failure tests**

Use deterministic fake index, recaller, and model. Test:

| Documents | Memories | Expected |
| --- | --- | --- |
| present | absent | one S-only model request |
| absent or knowledge DB empty | present | one M-only model request |
| present | present | one mixed model request |
| absent | absent | local insufficient result and zero model calls |

Also require input validation before dependencies, `KnowledgeBaseEmptyError` mapped
to an empty document context, recall warnings preserved on a document-only answer,
all other document search failures propagated, and model/parser failures propagated
without unvalidated fallback text.

- [ ] **Step 2: Run the file and verify RED**

Expected: import failure because the service does not exist.

- [ ] **Step 3: Implement the small coordinator**

Follow the existing `AnswerService` validation and implement:

```python
try:
    results = self._search_service.search(question, limit=limit, mode=mode)
except KnowledgeBaseEmptyError:
    results = []
recall = self._memory_recaller.recall(question, limit=MAX_RECALLED_MEMORIES)
evidence = build_evidence(results)
if not evidence and not recall.memories:
    answer = GroundedAnswer(
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        answer=_INSUFFICIENT_EVIDENCE,
        citation_ids=(),
        evidence=(),
        model_id=None,
    )
    return MemoryAwareAnswerResult(answer, recall)
messages = build_memory_aware_messages(question, evidence, recall.memories)
output = self._language_model.generate(messages)
answer = parse_contextual_answer(output, evidence, recall.memories)
return MemoryAwareAnswerResult(answer, recall)
```

Do not catch document errors other than `KnowledgeBaseEmptyError`, provider errors,
or parser errors here.

- [ ] **Step 4: Run new and existing answer-service tests**

Expected: PASS; `tests/unit/test_answer_service.py` remains unchanged and green.

- [ ] **Step 5: Commit Task 7**

```bash
git add src/personal_knowledge_agent/application/memory_aware_answer.py tests/unit/test_memory_aware_answer_service.py
git commit -m "feat: answer with document and memory context"
```

---

### Task 8: Route lazily to the memory-aware answerer

**Files:**
- Modify: `src/personal_knowledge_agent/application/agent.py`
- Modify: `tests/unit/test_agent_service.py`

**Interfaces:**
- Consumes: `GroundedQuestionAnswerer`, `Callable[[], MemoryAwareQuestionAnswerer]`, `MemoryAwareAnswerResult`, and `MemoryRecallReport`.
- Produces: direct/fallback responses with no recall report and non-fallback retrieve responses with the exact recall report.

- [ ] **Step 1: Write failing route/laziness tests**

Add a factory fake with a call counter. Require:

- direct rule/LLM routes call neither answerer nor factory;
- fallback calls only the existing document answerer;
- non-fallback retrieve calls the factory exactly once and only its returned answerer;
- route observer still runs before any answerer failure;
- the memory-aware result's report is attached to `AgentResponse` unchanged.

- [ ] **Step 2: Run Agent tests and verify RED**

Expected: FAIL because `AgentService` has no lazy memory-aware dependency.

- [ ] **Step 3: Implement explicit route selection**

Change the constructor to accept:

```python
def __init__(
    self,
    router: QuestionRouter,
    answer_service: GroundedQuestionAnswerer,
    memory_answerer_factory: Callable[[], MemoryAwareQuestionAnswerer],
) -> None: ...
```

Until Task 9 migrates the CLI composition root, keep the factory optional. Its
temporary compatibility path uses the existing document answerer and attaches an
empty recall report, so the domain invariant and static checks remain valid. Task 9
must remove this default once every production caller supplies the lazy factory.

After direct handling, use the old answerer only when
`route.source is RouteSource.FALLBACK`; otherwise invoke the factory once, call the
memory-aware answerer, and construct `AgentResponse` with its recall report.
In the same GREEN step, tighten `AgentResponse.__post_init__` so every non-fallback
retrieve response requires a non-`None` recall report; update its domain test to
prove a missing report is rejected only after all production callers are migrated.

- [ ] **Step 4: Run Agent and domain-model tests**

Expected: PASS, including route observer and safe failure behavior.

- [ ] **Step 5: Commit Task 8**

```bash
git add src/personal_knowledge_agent/application/agent.py tests/unit/test_agent_service.py
git commit -m "feat: route agent answers through memory recall"
```

---

### Task 9: Compose Phase 3B in the CLI and render visible memory use

**Files:**
- Modify: `src/personal_knowledge_agent/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 3–8 concrete services and existing `--memory-db`.
- Produces: lazy recall composition, safe warning labels, separate non-empty `Sources` and `Memories used` sections, and unchanged post-answer extraction.

- [ ] **Step 1: Write failing end-to-end CLI tests**

Use temporary document and memory databases plus fake model outputs to cover:

- memory-only `[M1]` answer and full `type.key — content` display;
- mixed `[S1]`/`[M1]` answer with separate sections;
- supplied but uncited memory not printed under `Memories used`;
- no empty `Sources:` or `Memories used:` heading;
- storage/embedding/unsafe/ambiguous warning labels without raw exception details;
- direct and fallback routes do not call `_memory_store`, load E5 for memory, or create the custom memory DB;
- non-fallback retrieve uses the selected custom DB;
- an empty knowledge database plus relevant active memory produces a memory-only
  answer instead of the old empty-knowledge-base error;
- Phase 3A extraction still occurs after the validated answer and uses the same DB.

- [ ] **Step 2: Run focused CLI tests and verify RED**

Expected: FAIL because Agent construction lacks the factory and output assumes every
citation is an `S` source.

- [ ] **Step 3: Add lazy composition**

Inside `_agent`, retain the existing document answer service and define a closure:

```python
def memory_answerer() -> MemoryAwareAnswerService:
    store = _memory_store(memory_database)
    recaller = MemoryRecallService(
        store,
        ConservativeMemoryRelevancePolicy(),
        LocalMemorySafetyPolicy(),
    )
    return MemoryAwareAnswerService(
        SearchService(index),
        recaller,
        language_model,
    )
```

Pass this closure to `AgentService`. Do not invoke it in the CLI before routing.
Remove Task 8's optional factory default and compatibility branch in the same
change, making the lazy factory mandatory for every production caller.

- [ ] **Step 4: Render warnings and cited context safely**

Add fixed user-facing labels keyed by `MemoryRecallWarning`. Print warnings once in
enum order after the route. Split validated citation IDs by prefix:

```python
source_ids = tuple(item for item in answer.citation_ids if item.startswith("S"))
memory_ids = tuple(item for item in answer.citation_ids if item.startswith("M"))
```

Print only non-empty sections. Resolve `M` labels only against the attached recall
report and render complete normalized content locally:

```text
[M1] preference.answer_style — Prefer concise answers.
```

An unknown label should already be impossible after parsing; keep local assertions
without exposing provider content.

- [ ] **Step 5: Run full CLI and compatibility files**

Run all of `tests/test_cli.py`, Agent tests, and Phase 3A proposal/admin tests.
Expected: PASS with exact Phase 2 direct/fallback and Phase 3A extraction behavior.

- [ ] **Step 6: Commit Task 9**

```bash
git add src/personal_knowledge_agent/cli.py tests/test_cli.py
git commit -m "feat: show memory-aware agent answers"
```

---

### Task 10: Harden disclosure, lifecycle, and compatibility boundaries

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/integration/test_sqlite_memory_store.py`
- Modify: `tests/unit/test_memory_recall_service.py`
- Modify: `tests/unit/test_memory_aware_prompt_builder.py`
- Modify: `tests/unit/test_structured_contextual_answer_parser.py`

**Interfaces:**
- Consumes: the complete Phase 3B workflow.
- Produces: adversarial evidence that unsafe, irrelevant, non-active, partial-conflict, and metadata-bearing memory cannot cross the provider boundary.

- [ ] **Step 1: Add adversarial tests before changing any production code**

Cover at least:

- pending and superseded sentinel content absent from outgoing model messages;
- irrelevant active memory below `0.84` with no meaningful overlap absent;
- one unsafe peer causes the complete slot to disappear;
- four keep-both peers cause zero members of that slot to be sent;
- two peers with only one remaining budget position cause neither to be sent;
- hostile memory containing `SYSTEM`, fake JSON delimiters, and `[S99]/[M99]` stays
  JSON data and cannot validate an unknown citation;
- provider payload and combined stdout/stderr omit memory ID, path, hash, timestamp,
  score, vector, API key, and traceback sentinels;
- recall failure plus document evidence still produces the validated document answer
  and exit zero;
- recall failure plus no documents returns local insufficient evidence and zero
  answer-model calls;
- `pka ask`, direct, and fallback exact stdout remain byte-for-byte compatible.

- [ ] **Step 2: Run adversarial tests and observe RED only for real gaps**

If every new test immediately passes, strengthen it until mutating/removing the
intended guard would make it fail. If any test fails, invoke `systematic-debugging`,
identify the root cause, and keep fixes within the approved Phase 3B scope.

- [ ] **Step 3: Stop and make any discovered production gap explicit**

No production edit is authorized by this generic hardening step. If a new test
exposes a real gap, use `systematic-debugging`, record the root cause, and add an
explicit fix subtask to this plan with the exact production path, failing assertion,
minimal patch, and GREEN command before editing production code.

#### Verified fix subtask: reject hidden malformed body citations

The merge review found that
`infrastructure/structured_contextual_answer_parser.py` extracts only already-valid
positive labels. A body containing valid `[M1]` plus malformed `[M0]`, `[S01]`, or
`[M-1]` can therefore match `citation_ids=["M1"]` and bypass validation. Add failing
cases to `tests/unit/test_structured_contextual_answer_parser.py`, then scan every
bracketed token beginning with `S` or `M`, require the complete token to match
`(?:S|M)[1-9]\d*`, and only then apply the existing first-use order and membership
checks. Run the focused parser file, both answer-parser files, and the full quality
gate before committing the fix.

#### Review alignment subtask: preserve the approved warning text contract

The merge review also confirmed that the implemented `Memory recall warning:` prefix
and `unsafe memory` label differ from the approved failure matrix. Update CLI tests
first to require `Memory recall skipped:` and `unsafe stored memory`, then align
`cli.py`, README, and the Phase 3B learning log. Run focused CLI tests before the
full quality gate.

#### Review robustness subtask: compute bounded cosine after float32 storage

The merge review noted that a raw dot product between a float64 query vector and a
float32 stored vector can drift outside `[-1, 1]`. A 100-dimensional identical unit
vector reproduces `1.0000000149`, causing a valid search to become a storage error.
Add an integration regression in `tests/integration/test_sqlite_memory_store.py`,
then compute cosine with both norms in `sqlite_memory_store.py`, reject zero stored
norms as corrupt data, and clamp only the final floating-point boundary noise. Run
the focused integration file and full quality gate.

- [ ] **Step 4: Run the combined Phase 2/3 integration gate**

Run the full CLI file, SQLite memory integration, all memory unit files, Agent,
answer, prompt, and parser tests. Expected: PASS with no network calls.

- [ ] **Step 5: Commit Task 10**

```bash
git add tests/test_cli.py tests/integration/test_sqlite_memory_store.py tests/unit/test_memory_recall_service.py tests/unit/test_memory_aware_prompt_builder.py tests/unit/test_structured_contextual_answer_parser.py
git diff --cached --name-only
git commit -m "test: harden memory-aware answer boundaries"
```

Before committing, unstage any production file not tied to a verified RED case and
confirm `.env`, `.pka`, and user-owned directories are absent.

---

### Task 11: Document Phase 3B and run the merge gate

**Files:**
- Modify: `README.md`
- Create: `docs/learning-log/phase-3b.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: completed Phase 3B behavior and the approved design.
- Produces: accurate user guidance, portfolio learning notes, contributor scope, and merge-ready verification evidence.

- [ ] **Step 1: Update README usage, privacy, and cost documentation**

Document:

- eligible non-fallback Agent recall and direct/fallback/ask exclusions;
- memory-only and mixed output examples with `S` and `M` labels;
- the 12-candidate, `0.84`, maximum-three conservative policy;
- slot-atomic keep-both behavior and warnings;
- full local `Memories used` display;
- provider-visible memory fields and excluded metadata;
- safe degradation and no-context local refusal;
- zero additional remote calls for recall, possible local E5 work, and unchanged
  optional Phase 3A extraction request/retry cost;
- unencrypted storage, external-provider retention, and secure-erasure limitations.

- [ ] **Step 2: Write the Phase 3B learning log**

Explain with small English/Chinese examples:

- retrieval rank versus an absolute disclosure gate;
- lexical key signals and cosine threshold trade-offs;
- document evidence versus personal memory semantics;
- untrusted context separation and prompt injection;
- request-local `S`/`M` structured citations;
- slot-atomic conflict handling;
- optional-subsystem degradation;
- why router/direct/fallback paths do not receive memory;
- Phase 4 remains future work.

- [ ] **Step 3: Update contributor scope**

Mark Phase 3B complete only after all implementation tests pass. State that Phase 4
knowledge-graph design is next, while PDF/web/code ingestion, formal evaluation, and
deployment remain future work.

- [ ] **Step 4: Run the complete quality gate**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff format --check .
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff check .
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run mypy src
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3b-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pka health
```

Expected: all tests pass; Ruff reports no changes/issues; mypy reports no issues;
health prints `Personal Knowledge Agent is healthy.`

- [ ] **Step 5: Inspect scope and commit documentation**

```bash
git status --short
git diff --check
git diff --cached --name-only
git add README.md AGENTS.md docs/learning-log/phase-3b.md
git commit -m "docs: explain memory-aware agent answers"
```

Expected staged scope: exactly those three files. Never stage `.env`, `.pka`,
`papers/`, `references/`, or `research/`.

- [ ] **Step 6: Request final branch review before integration**

Use `requesting-code-review` against the approved spec and this plan. Resolve every
verified Critical or Important finding with TDD, rerun the complete quality gate,
obtain a clean re-review, then use `finishing-a-development-branch` to present the
four integration options.

---

## Plan self-review checklist

- [x] Every approved product decision maps to at least one task and test.
- [x] Every new type/function used by a later task is produced by an earlier task.
- [x] Direct/fallback laziness is tested at both application and CLI layers.
- [x] Memory-only, mixed, empty, and degraded paths are all covered.
- [x] Provider disclosure has exact field allowlists and sentinel tests.
- [x] Slot-atomic conflict budgeting is specified without partial injection.
- [x] SQLite schema remains unchanged and read-only recall is verified.
- [x] Old document-only parser and `pka ask` remain isolated.
- [x] No task adds chat history, autonomous writes, tools, graphs, evaluation, or deployment.
- [x] Full gate and final independent review are mandatory before merge.

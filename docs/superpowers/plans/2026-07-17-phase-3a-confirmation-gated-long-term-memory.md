# Phase 3A Confirmation-Gated Long-Term Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, confirmation-gated long-term memory lifecycle with explicit memory administration, safe automatic candidate extraction after successful `pka agent` answers, and active-only hybrid memory search.

**Architecture:** New memory records and protocols live in the domain layer. Pure local prefilter and safety adapters guard a strict Chat Completions extractor; small application services coordinate proposal and administration workflows. A dedicated SQLite adapter at `.pka/memory.db` persists lifecycle state, postings, E5 vectors, replacement links, and rejection fingerprints without treating memories as knowledge documents.

**Tech Stack:** Python 3.12+, frozen dataclasses, `StrEnum`, standard-library JSON/regex/Unicode/hashlib/SQLite, OpenAI-compatible Chat Completions, multilingual E5, mixed Chinese/English tokenization, BM25, cosine similarity, RRF, pytest, Ruff, strict mypy, and Git.

**Approved design:** `docs/superpowers/specs/2026-07-17-phase-3a-confirmation-gated-long-term-memory-design.md`

## Global Constraints

- Execute in an isolated worktree created with the `using-git-worktrees` skill; use a `codex/phase-3a-memory-lifecycle` branch unless that name already exists.
- Use Python 3.12+ features, complete type annotations, and strict mypy.
- Keep `domain/` independent of SQLite, OpenAI, E5, and other infrastructure libraries.
- Automatic extraction creates `pending` candidates only; only explicit `remember` may create an active memory immediately.
- Allowed types are exactly `preference`, `goal`, `project`, and `learning_progress`.
- Keys must match `[a-z][a-z0-9_.-]{0,63}`; content is at most 500 Unicode code points; extraction input is at most 4,000 code points; one extraction returns at most three candidates.
- Never persist passwords, API keys, access tokens, private keys, identity documents, health information, financial information, or privileged legal details.
- Screen the raw current message locally before the extraction call, and screen every parsed key/content pair again before persistence.
- The extractor receives only the current user message; it never receives existing memories, knowledge evidence, paths, hashes, timestamps, embeddings, or credentials.
- Automatic candidates are written atomically; expected extraction failures must not change a successful agent command to a non-zero exit status.
- Rejection deletes candidate content and retains only a type-key-content fingerprint plus rejection time.
- Forgetting hard-deletes the complete connected replacement chain, postings, vectors, and matching rejection fingerprints; it never reactivates old values.
- Memory search returns active records only and uses hybrid BM25 plus exact cosine rankings fused with RRF.
- Keep `pka ask` behavior unchanged. Phase 3A must not inject active memories into routes, prompts, or answers.
- Keep generated databases under `.pka/` or pytest temporary directories and never commit them.
- Keep all provider and embedding tests offline with deterministic fakes; never expose secrets, raw provider bodies, prompts, vectors, or stack traces in expected user-facing errors.
- Preserve user-owned untracked `papers/`, `references/`, and `research/` directories.
- On `/mnt/c`, run uv commands with `TMPDIR=/tmp`, `UV_LINK_MODE=copy`, `UV_CACHE_DIR=/tmp/uv-cache`, and a task-specific `UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv`.

---

## File Map

- Create `src/personal_knowledge_agent/domain/memory.py`: memory enums, immutable records, limits, normalization, IDs, and fingerprints.
- Modify `src/personal_knowledge_agent/domain/errors.py`: memory validation, safety, conflict, lookup, storage, and schema errors.
- Modify `src/personal_knowledge_agent/domain/protocols.py`: `MemoryExtractor`, `MemorySafetyPolicy`, and `MemoryStore` boundaries.
- Create `src/personal_knowledge_agent/infrastructure/memory_prefilter.py`: local English/Chinese first-person durable-signal heuristic.
- Create `src/personal_knowledge_agent/infrastructure/memory_safety.py`: conservative local prohibited-data detector.
- Create `src/personal_knowledge_agent/infrastructure/memory_prompt_builder.py`: minimized extraction messages and exact JSON contract.
- Create `src/personal_knowledge_agent/infrastructure/structured_memory_parser.py`: strict duplicate-key-aware candidate parser.
- Create `src/personal_knowledge_agent/infrastructure/llm_memory_extractor.py`: compose `LanguageModel`, prompt builder, and parser.
- Create `src/personal_knowledge_agent/application/memory_proposal.py`: guarded automatic proposal workflow and safe skip categories.
- Create `src/personal_knowledge_agent/application/memory_admin.py`: explicit remember/list/search/approve/reject/forget use cases.
- Modify `src/personal_knowledge_agent/infrastructure/retrieval_fusion.py`: add reusable ID-based RRF scoring while preserving chunk fusion.
- Create `src/personal_knowledge_agent/infrastructure/sqlite_memory_store.py`: schema, lifecycle transactions, BM25/cosine retrieval, and RRF fusion.
- Modify `src/personal_knowledge_agent/cli.py`: nested `pka memory` commands, `--memory-db`, candidate display, and non-fatal extraction output.
- Modify `tests/helpers.py`: deterministic memory-store fake plus clock/ID helpers if repeated tests need them.
- Create `tests/unit/test_memory_models.py`.
- Create `tests/unit/test_memory_prefilter.py`.
- Create `tests/unit/test_memory_safety.py`.
- Create `tests/unit/test_memory_prompt_builder.py`.
- Create `tests/unit/test_structured_memory_parser.py`.
- Create `tests/unit/test_llm_memory_extractor.py`.
- Create `tests/unit/test_memory_proposal_service.py`.
- Create `tests/unit/test_memory_admin_service.py`.
- Modify `tests/unit/test_retrieval_fusion.py`.
- Create `tests/integration/test_sqlite_memory_store.py`.
- Modify `tests/test_cli.py`.
- Modify `README.md`: document memory lifecycle, commands, privacy boundary, and Phase 3A/3B split.
- Create `docs/learning-log/phase-3a.md`: explain memory versus history/knowledge, write authority, state transitions, privacy, and retrieval.
- Modify `AGENTS.md`: record Phase 2B and Phase 3A completion and move memory-aware answers to future Phase 3B scope.

### Task 1: Define memory domain contracts

**Files:**
- Create: `src/personal_knowledge_agent/domain/memory.py`
- Modify: `src/personal_knowledge_agent/domain/errors.py`
- Modify: `src/personal_knowledge_agent/domain/protocols.py`
- Create: `tests/unit/test_memory_models.py`

**Interfaces:**
- Consumes: existing `EmbeddingModel`-independent domain conventions and `KnowledgeError`.
- Produces: `MemoryType`, `MemoryStatus`, `MemoryActivationAction`, `MemoryExtractionSkipReason`, `MemoryProposal`, `MemoryRecord`, `MemorySearchResult`, `MemoryExtractionReport`, normalization/hash/ID helpers, `MemoryExtractor`, `MemorySafetyPolicy`, and `MemoryStore`.

- [ ] **Step 1: Write failing domain tests**

Create focused tests with these exact behavior families:

```python
_NOW = datetime(2026, 7, 17, tzinfo=UTC)


def make_record(
    *,
    created_at: datetime = _NOW,
    updated_at: datetime = _NOW,
) -> MemoryRecord:
    content = "Build an agent"
    return MemoryRecord(
        memory_id="mem_00000000000000000000000000000001",
        memory_type=MemoryType.GOAL,
        memory_key="goal.agent",
        content=content,
        content_hash=memory_content_hash(content),
        status=MemoryStatus.PENDING,
        supersedes_id=None,
        source_hash=memory_source_hash("My goal is to build an agent"),
        created_at=created_at,
        updated_at=updated_at,
    )


def test_proposal_normalizes_nfkc_and_whitespace() -> None:
    proposal = MemoryProposal(
        MemoryType.GOAL,
        "goal.ai_engineer",
        "  Build\u3000ＡＩ   projects  ",
    )
    assert proposal.content == "Build AI projects"


@pytest.mark.parametrize("key", ["Goal.ai", "1goal", "goal/a", "a" * 65])
def test_proposal_rejects_invalid_key(key: str) -> None:
    with pytest.raises(ValueError, match="memory_key"):
        MemoryProposal(MemoryType.GOAL, key, "Build an agent")


def test_hashes_are_normalized_but_slot_specific() -> None:
    assert memory_content_hash(" RAG  Notes ") == memory_content_hash("rag notes")
    assert memory_proposal_hash(
        MemoryType.GOAL, "goal.rag", "RAG Notes"
    ) != memory_proposal_hash(
        MemoryType.PROJECT, "project.rag", "RAG Notes"
    )


def test_memory_record_requires_utc_timestamps_and_finite_score() -> None:
    now = datetime(2026, 7, 17, tzinfo=UTC)
    record = make_record(created_at=now, updated_at=now)
    with pytest.raises(ValueError, match="finite"):
        MemorySearchResult(record, float("nan"))


def test_report_cannot_mix_candidates_and_skip_reason() -> None:
    with pytest.raises(ValueError, match="skip reason"):
        MemoryExtractionReport(
            (make_record(),), MemoryExtractionSkipReason.PREFILTER_MISS
        )
```

Also cover all exact enum values, blank/501-character content, the `mem_` plus UUID-hex ID shape, 64-character lowercase hashes, timestamp ordering, and immutable records.

- [ ] **Step 2: Run the domain tests to verify RED**

Run:

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_models.py -v
```

Expected: collection fails because `personal_knowledge_agent.domain.memory` does not exist.

- [ ] **Step 3: Add the minimum domain implementation**

Define these public values and signatures in `domain/memory.py`:

```python
MAX_MEMORY_CONTENT_LENGTH = 500
MAX_MEMORY_EXTRACTION_INPUT_LENGTH = 4_000
MAX_MEMORY_CANDIDATES = 3


class MemoryType(StrEnum):
    PREFERENCE = "preference"
    GOAL = "goal"
    PROJECT = "project"
    LEARNING_PROGRESS = "learning_progress"


class MemoryStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class MemoryActivationAction(StrEnum):
    ACTIVATE = "activate"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"


class MemoryExtractionSkipReason(StrEnum):
    PREFILTER_MISS = "prefilter_miss"
    INPUT_TOO_LONG = "input_too_long"
    SENSITIVE_INPUT = "sensitive_input"
    NO_NEW_CANDIDATES = "no_new_candidates"
    PROVIDER_ERROR = "provider_error"
    INVALID_OUTPUT = "invalid_output"
    UNSAFE_OUTPUT = "unsafe_output"
    EMBEDDING_ERROR = "embedding_error"
    STORAGE_ERROR = "storage_error"
```

Implement `normalize_memory_content(value: str) -> str` with Unicode NFKC, whitespace collapse, strip, nonblank validation, and the 500-code-point limit. Validate keys with `re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", value)`. Add:

```python
def memory_content_hash(content: str) -> str: ...
def memory_source_hash(message: str) -> str: ...
def memory_proposal_hash(
    memory_type: MemoryType, memory_key: str, content: str
) -> str: ...
def make_memory_id() -> str:
    return f"mem_{uuid4().hex}"
```

All hashes use SHA-256 over normalized, case-folded input. The proposal hash joins type, key, and comparison content with `"\0"` separators.

Add frozen, slotted dataclasses with the approved fields. `MemoryProposal.__post_init__` normalizes content with `object.__setattr__`; `MemoryRecord` validates ID, type/key/content hashes, optional source/supersedes values, UTC-aware timestamps, and `created_at <= updated_at`; `MemorySearchResult` requires a finite score; `MemoryExtractionReport` requires exactly one of nonempty candidates or a skip reason.

Add these errors under `KnowledgeError` in `domain/errors.py`:

```python
class MemoryFeatureError(KnowledgeError): ...
class InvalidMemoryError(MemoryFeatureError): ...
class UnsafeMemoryError(MemoryFeatureError): ...
class MemoryConflictError(MemoryFeatureError): ...
class MemoryNotFoundError(MemoryFeatureError): ...
class MemoryStorageError(MemoryFeatureError): ...
class InvalidMemorySchemaError(MemoryFeatureError): ...
```

Add these exact protocols to `domain/protocols.py`:

```python
class MemoryExtractor(Protocol):
    def extract(self, message: str) -> tuple[MemoryProposal, ...]: ...


class MemorySafetyPolicy(Protocol):
    def allows(self, *values: str) -> bool:
        """Return false when any supplied local text matches a prohibited category."""
        ...


class MemoryStore(Protocol):
    def create_pending(
        self, proposals: Sequence[MemoryProposal], *, source_hash: str
    ) -> tuple[MemoryRecord, ...]: ...

    def remember(
        self,
        memory_type: MemoryType,
        memory_key: str,
        content: str,
        *,
        action: MemoryActivationAction = MemoryActivationAction.ACTIVATE,
        replace_id: str | None = None,
    ) -> MemoryRecord: ...

    def approve(
        self,
        candidate_id: str,
        *,
        action: MemoryActivationAction,
        replace_id: str | None = None,
    ) -> MemoryRecord: ...

    def reject(self, candidate_id: str) -> None: ...
    def forget(self, memory_id: str) -> int: ...

    def list_memories(
        self,
        *,
        status: MemoryStatus | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]: ...

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
    ) -> list[MemorySearchResult]: ...
```

- [ ] **Step 4: Verify GREEN and type quality**

Run:

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_models.py -v
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff check src/personal_knowledge_agent/domain tests/unit/test_memory_models.py
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run mypy src
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/domain tests/unit/test_memory_models.py
git commit -m "feat: define long-term memory contracts"
```

### Task 2: Add the local extraction prefilter and safety policy

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/memory_prefilter.py`
- Create: `src/personal_knowledge_agent/infrastructure/memory_safety.py`
- Create: `tests/unit/test_memory_prefilter.py`
- Create: `tests/unit/test_memory_safety.py`

**Interfaces:**
- Consumes: plain current-message, key, and content strings.
- Produces: `should_extract_memory(message: str) -> bool` and `LocalMemorySafetyPolicy.allows(*values: str) -> bool`.

- [ ] **Step 1: Write failing prefilter and safety tests**

Use parameterized cases:

```python
@pytest.mark.parametrize(
    "message",
    [
        "My goal is to become an AI Engineer.",
        "I prefer concise explanations.",
        "I am working on a Personal Knowledge Agent.",
        "I have completed the RAG retrieval module.",
        "我的目标是完成 AI Engineer 项目。",
        "我更喜欢先解释原理。",
        "我正在开发个人知识 Agent。",
        "我已经学完向量检索。",
    ],
)
def test_prefilter_accepts_durable_first_person_signals(message: str) -> None:
    assert should_extract_memory(message) is True


@pytest.mark.parametrize(
    "message",
    ["What is RAG?", "Explain my notes", "It is raining", "我应该学什么？", ""],
)
def test_prefilter_rejects_ordinary_or_nonstatement_text(message: str) -> None:
    assert should_extract_memory(message) is False
```

Safety tests must reject representative password, `sk-` token, PEM key, passport/身份证, diagnosis/病历, bank/card/债务, and lawsuit/法律案件 text in both the raw-message and key/content positions. They must allow the four normal learning examples above.

- [ ] **Step 2: Run both files to verify RED**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_prefilter.py tests/unit/test_memory_safety.py -v
```

Expected: missing-module collection failures.

- [ ] **Step 3: Implement conservative local matching**

In `memory_prefilter.py`, normalize with NFKC and whitespace collapse. Use compiled, case-insensitive whole-signal regular expressions for the approved English and Chinese phrases; do not call a model and do not match a bare pronoun alone.

In `memory_safety.py`, implement `LocalMemorySafetyPolicy` with:

```python
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bsk-[a-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"\b(?:password|passwd|api[_ -]?key|access[_ -]?token)\s*[:=]", re.IGNORECASE),
)

_PROHIBITED_TERMS = (
    "passport", "social security", "government id", "身份证", "护照",
    "diagnosis", "medical record", "treatment", "诊断", "病历", "治疗",
    "bank account", "credit card", "income", "debt", "银行卡", "信用卡", "收入", "债务",
    "lawsuit", "privileged legal", "诉讼", "法律案件",
)
```

Normalize and case-fold every supplied value. Return `False` on the first secret-pattern or prohibited-term match; return `True` only when all values pass. Keep the detector deterministic and document that conservative false positives are intentional.

- [ ] **Step 4: Verify GREEN, Ruff, and mypy**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_prefilter.py tests/unit/test_memory_safety.py -v
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff check src/personal_knowledge_agent/infrastructure/memory_prefilter.py src/personal_knowledge_agent/infrastructure/memory_safety.py tests/unit/test_memory_prefilter.py tests/unit/test_memory_safety.py
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run mypy src
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/infrastructure/memory_prefilter.py src/personal_knowledge_agent/infrastructure/memory_safety.py tests/unit/test_memory_prefilter.py tests/unit/test_memory_safety.py
git commit -m "feat: guard memory extraction locally"
```

### Task 3: Build and validate the structured memory extractor

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/memory_prompt_builder.py`
- Create: `src/personal_knowledge_agent/infrastructure/structured_memory_parser.py`
- Create: `src/personal_knowledge_agent/infrastructure/llm_memory_extractor.py`
- Create: `tests/unit/test_memory_prompt_builder.py`
- Create: `tests/unit/test_structured_memory_parser.py`
- Create: `tests/unit/test_llm_memory_extractor.py`

**Interfaces:**
- Consumes: `LanguageModel.generate(messages)` and `ModelOutput`.
- Produces: `build_memory_extraction_messages(message)`, `parse_memory_proposals(output)`, and `LLMMemoryExtractor.extract(message)`.

- [ ] **Step 1: Write failing prompt tests**

Assert exactly one system and one user message, four allowed types, key regex, sensitive exclusions, three-candidate limit, durable-versus-temporary guidance, and exact JSON-without-fences instruction. Assert the user message is exactly `CURRENT USER MESSAGE:\n{message}` and contains no old memory, evidence, path, hash, embedding, or API-key field. Blank and 4,001-character messages raise `InvalidMemoryError` before a model call.

- [ ] **Step 2: Implement the prompt builder and verify GREEN**

The system contract must show this exact shape:

```json
{"candidates":[{"type":"goal","key":"goal.ai_engineer_internship","content":"The user aims to complete an AI Engineer internship project."}]}
```

It must say an empty list is valid, preserve the user's language when practical, and forbid extra fields or commentary. Run:

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_prompt_builder.py -v
```

Expected: PASS.

- [ ] **Step 3: Write failing strict-parser tests**

Accept zero to three valid proposals. Reject invalid JSON, fences, duplicate JSON keys at any object level, non-object roots, wrong/missing/extra root fields, non-list candidates, wrong/missing/extra candidate fields, non-string values, unknown types, invalid keys, blank/501-character content, and four candidates. Every failure raises `InvalidMemorySchemaError` without echoing provider content.

- [ ] **Step 4: Implement strict parsing and verify GREEN**

Use `json.loads(..., object_pairs_hook=_build_unique_object)`, `type(value) is ...` checks, exact field sets, and `MemoryProposal` construction. Convert `ValueError` from type/key/content validation into a safe `InvalidMemorySchemaError`. Never strip a fence, infer a field, or keep a partial list.

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_structured_memory_parser.py -v
```

Expected: PASS.

- [ ] **Step 5: Write the extractor adapter test and implementation**

Prove the adapter forwards exactly the builder's messages and returns the parser's tuple:

```python
model = FakeLanguageModel(
    ModelOutput(
        "extractor",
        '{"candidates":[{"type":"goal","key":"goal.ai","content":"Build AI systems"}]}',
        "stop",
    )
)
assert LLMMemoryExtractor(model).extract("My goal is AI") == (
    MemoryProposal(MemoryType.GOAL, "goal.ai", "Build AI systems"),
)
assert len(model.calls) == 1
```

`LLMMemoryExtractor` must not catch `LanguageModelError` or `InvalidMemorySchemaError`; the application service owns safe skip mapping. Unexpected exceptions also propagate.

- [ ] **Step 6: Run all extractor tests, Ruff, mypy, and commit**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_prompt_builder.py tests/unit/test_structured_memory_parser.py tests/unit/test_llm_memory_extractor.py -v
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff check src/personal_knowledge_agent/infrastructure tests/unit/test_memory_prompt_builder.py tests/unit/test_structured_memory_parser.py tests/unit/test_llm_memory_extractor.py
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run mypy src
```

Expected: all pass.

```bash
git add src/personal_knowledge_agent/infrastructure/memory_prompt_builder.py src/personal_knowledge_agent/infrastructure/structured_memory_parser.py src/personal_knowledge_agent/infrastructure/llm_memory_extractor.py tests/unit/test_memory_prompt_builder.py tests/unit/test_structured_memory_parser.py tests/unit/test_llm_memory_extractor.py
git commit -m "feat: add structured memory extraction"
```

### Task 4: Coordinate automatic pending-memory proposals

**Files:**
- Create: `src/personal_knowledge_agent/application/memory_proposal.py`
- Create: `tests/unit/test_memory_proposal_service.py`
- Modify: `tests/helpers.py`
- Modify: `tests/unit/test_helpers.py`

**Interfaces:**
- Consumes: `MemoryExtractor`, `MemorySafetyPolicy`, `MemoryStore.create_pending`, prefilter callable, and domain hash helpers.
- Produces: `MemoryProposalService.propose(message: str) -> MemoryExtractionReport`.

- [ ] **Step 1: Add a deterministic memory-store fake and failing service tests**

Add a typed `FakeMemoryStore` to `tests/helpers.py` that records every protocol call and returns configured values or raises a configured exception. Represent pending calls as `list[tuple[tuple[MemoryProposal, ...], str]]`, where the second tuple value is `source_hash`. Do not use SQLite in unit tests.

Test this exact call order:

```python
report = service.propose("My goal is to build an agent")
assert events == ["prefilter", "raw_safety", "extract", "proposal_safety", "store"]
assert report.candidates == (pending_record,)
assert report.skipped_reason is None
assert store.create_pending_calls[0][1] == memory_source_hash(message)
```

Also prove: blank raises `InvalidMemoryError`; 4,001 characters skip before dependencies; prefilter miss makes zero safety/model/store calls; raw safety hit makes zero extractor/store calls; empty extractor output skips; any unsafe parsed proposal rejects the complete batch; provider/schema/embedding/storage expected errors map to their exact skip reasons; zero rows returned by duplicate filtering maps to `NO_NEW_CANDIDATES`; unexpected `RuntimeError` propagates.

- [ ] **Step 2: Run the service tests to verify RED**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_proposal_service.py tests/unit/test_helpers.py -v
```

Expected: missing service and helper failures.

- [ ] **Step 3: Implement the thin coordinator**

Constructor:

```python
class MemoryProposalService:
    def __init__(
        self,
        extractor: MemoryExtractor,
        store: MemoryStore,
        safety_policy: MemorySafetyPolicy,
        prefilter: Callable[[str], bool],
    ) -> None: ...

    def propose(self, message: str) -> MemoryExtractionReport: ...
```

Validate blank, then length, prefilter, raw safety, extractor, complete proposal safety, source hash, and one `create_pending` call in that order. A false post-extraction safety result maps directly to `UNSAFE_OUTPUT`. Catch only `LanguageModelError`, `InvalidMemorySchemaError`, `EmbeddingError`, and `MemoryStorageError` and map them to approved reasons. Treat ordinary no-op reasons as reports, never exceptions.

- [ ] **Step 4: Verify GREEN, existing helper tests, Ruff, and mypy**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_proposal_service.py tests/unit/test_helpers.py -v
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff check src/personal_knowledge_agent/application/memory_proposal.py tests/helpers.py tests/unit/test_memory_proposal_service.py tests/unit/test_helpers.py
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run mypy src
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/application/memory_proposal.py tests/helpers.py tests/unit/test_memory_proposal_service.py tests/unit/test_helpers.py
git commit -m "feat: coordinate pending memory proposals"
```

### Task 5: Add explicit memory administration use cases

**Files:**
- Create: `src/personal_knowledge_agent/application/memory_admin.py`
- Create: `tests/unit/test_memory_admin_service.py`

**Interfaces:**
- Consumes: `MemoryStore`, `MemorySafetyPolicy`, domain types, and normalization validation.
- Produces: `MemoryAdminService.remember`, `.list`, `.search`, `.approve`, `.reject`, and `.forget`.

- [ ] **Step 1: Write failing administration tests**

Prove explicit `remember` validates and safety-checks before calling the store, passes `ACTIVATE`, `REPLACE`, or `KEEP_BOTH` unchanged, and never calls a language model. Prove invalid combinations (`ACTIVATE` plus `replace_id`, `REPLACE` without ID, `KEEP_BOTH` plus ID), blank search, limits outside 1..20, and blank IDs fail before store calls. Verify all valid operations return exactly the store's value.

- [ ] **Step 2: Run to verify RED**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_admin_service.py -v
```

Expected: missing service failure.

- [ ] **Step 3: Implement explicit validation and delegation**

Use this public surface:

```python
class MemoryAdminService:
    def remember(
        self,
        memory_type: MemoryType,
        memory_key: str,
        content: str,
        *,
        action: MemoryActivationAction = MemoryActivationAction.ACTIVATE,
        replace_id: str | None = None,
    ) -> MemoryRecord: ...

    def list(
        self,
        *,
        status: MemoryStatus | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]: ...

    def search(
        self, query: str, *, limit: int = 5, memory_type: MemoryType | None = None
    ) -> list[MemorySearchResult]: ...

    def approve(
        self,
        candidate_id: str,
        *,
        action: MemoryActivationAction = MemoryActivationAction.ACTIVATE,
        replace_id: str | None = None,
    ) -> MemoryRecord: ...

    def reject(self, candidate_id: str) -> None: ...
    def forget(self, memory_id: str) -> int: ...
```

Construct a validated `MemoryProposal` for explicit writes before safety checking key and content. Keep the service free of SQLite and CLI formatting.
Convert domain-constructor `ValueError` into `InvalidMemoryError` so expected CLI input failures remain inside the project's user-facing error boundary. A false safety-policy result raises `UnsafeMemoryError` without calling the store.

- [ ] **Step 4: Verify GREEN and commit**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_memory_admin_service.py -v
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff check src/personal_knowledge_agent/application/memory_admin.py tests/unit/test_memory_admin_service.py
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run mypy src
```

Expected: all pass.

```bash
git add src/personal_knowledge_agent/application/memory_admin.py tests/unit/test_memory_admin_service.py
git commit -m "feat: add memory administration service"
```

### Task 6: Generalize RRF scoring for memory IDs

**Files:**
- Modify: `src/personal_knowledge_agent/infrastructure/retrieval_fusion.py`
- Modify: `tests/unit/test_retrieval_fusion.py`

**Interfaces:**
- Consumes: ordered sequences of stable string IDs.
- Produces: `reciprocal_rank_scores(*rankings: Sequence[str], limit: int, rank_constant: int = 60) -> list[tuple[str, float]]`; preserves `reciprocal_rank_fusion` behavior.

- [ ] **Step 1: Write failing ID-fusion tests**

```python
def test_rrf_scores_stable_ids_for_non_document_consumers() -> None:
    fused = reciprocal_rank_scores(["mem_a", "mem_b"], ["mem_b", "mem_c"], limit=3)
    assert [memory_id for memory_id, _ in fused] == ["mem_b", "mem_a", "mem_c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_id_scores_ignore_duplicates_and_break_ties_by_id() -> None:
    assert reciprocal_rank_scores(["z", "z"], ["a"], limit=1)[0][0] == "a"
```

- [ ] **Step 2: Verify RED, then implement the pure helper**

Run before implementation:

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_retrieval_fusion.py -v
```

Expected: FAIL because `reciprocal_rank_scores` is missing.

Move the score accumulation and stable `(-score, id)` ordering into `reciprocal_rank_scores`. Refactor existing `reciprocal_rank_fusion` to pass chunk IDs to the helper and reconstruct `SearchResult` values. Preserve all existing score and tie behavior.

- [ ] **Step 3: Verify all fusion tests and retrieval integration tests**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/unit/test_retrieval_fusion.py tests/integration/test_sqlite_keyword_index.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/personal_knowledge_agent/infrastructure/retrieval_fusion.py tests/unit/test_retrieval_fusion.py
git commit -m "refactor: share reciprocal-rank scoring"
```

### Task 7: Persist the SQLite memory lifecycle atomically

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/sqlite_memory_store.py`
- Create: `tests/integration/test_sqlite_memory_store.py`

**Interfaces:**
- Consumes: `EmbeddingModel`, `MixedLanguageTokenizer`, memory domain helpers, injected `clock: Callable[[], datetime]`, and `id_factory: Callable[[], str]`.
- Produces: `SQLiteMemoryStore.create_pending`, `.remember`, `.approve`, `.reject`, `.forget`, and `.list_memories` with restart-safe transactions.

- [ ] **Step 1: Write failing schema and write-path tests**

Use a fixed UTC clock, deterministic IDs, `FakeEmbedder`, and temporary databases. Assert schema version 1 and these tables: `schema_metadata`, `memories`, `memory_postings`, `memory_embeddings`, `memory_rejections`. Test pending batch persistence, automatic `source_hash`, explicit `source_hash=None`, explicit active write, NFKC content, postings, little-endian float32 vectors, model ID/dimensions, newest-first listing, status/type filters, persistence after reopening, duplicate active/pending triple suppression, and rejected-fingerprint suppression.

The schema test must inspect constraints and indexes, not only table names. Use this schema shape:

```sql
CREATE TABLE schema_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL
);
CREATE TABLE memories (
    memory_id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL CHECK (
        memory_type IN ('preference', 'goal', 'project', 'learning_progress')
    ),
    memory_key TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'superseded')),
    supersedes_id TEXT REFERENCES memories(memory_id) DEFERRABLE INITIALLY DEFERRED,
    source_hash TEXT,
    token_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_memory_live_duplicate
ON memories(memory_type, memory_key, content_hash)
WHERE status IN ('pending', 'active');
```

Add foreign-key-cascading posting and embedding tables and `memory_rejections(proposal_hash PRIMARY KEY, rejected_at)`. Add the approved status/type/key, postings-token, and embedding-model indexes.

- [ ] **Step 2: Run schema/write tests to verify RED**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/integration/test_sqlite_memory_store.py -v
```

Expected: missing store failure.

- [ ] **Step 3: Implement initialization, mapping, embeddings, writes, and listing**

Constructor:

```python
class SQLiteMemoryStore:
    def __init__(
        self,
        database_path: Path,
        embedding_model: EmbeddingModel,
        tokenizer: MixedLanguageTokenizer | None = None,
        *,
        clock: Callable[[], datetime] = utc_now,
        id_factory: Callable[[], str] = make_memory_id,
    ) -> None: ...
```

Use ISO-8601 UTC timestamps, `PRAGMA foreign_keys = ON`, explicit `BEGIN`, rollback on `sqlite3.Error`, and safe `MemoryStorageError` messages. Validate every embedding count/model/dimension and pack float32 before beginning a write transaction. Map corrupt enums, hashes, timestamps, IDs, and BLOBs to `MemoryStorageError` without leaking row content.

For `create_pending`, filter exact live duplicates and rejection proposal hashes, prepare all remaining embeddings, then write the complete remaining batch in one transaction. For explicit `remember`, enforce activation conflicts inside the same transaction; `REPLACE` marks the chosen active record superseded and sets the new record's `supersedes_id`; `KEEP_BOTH` requires a same-slot active conflict and leaves both active.

- [ ] **Step 4: Write failing lifecycle transition tests**

Cover bare approval with no conflict, bare approval conflict with actionable active IDs, valid replace, valid keep-both, wrong-slot replacement, stale/nonpending candidate, reject pending only, fingerprint persistence with candidate/content/vector/posting deletion, and unknown IDs. Verify injected duplicate-ID or SQL failure rolls back every status and child-row change.

For forgetting, construct `old <- replacement <- newest`, then forget the middle ID and assert all three memories, their postings/vectors, and matching rejection fingerprints are gone. Assert unrelated memories remain, the returned count is three, unknown ID errors, and no superseded memory is reactivated.

- [ ] **Step 5: Implement approval, rejection, and recursive forgetting**

`approve` and explicit `remember` share one private activation validator so CLI paths cannot diverge. Query exact active conflicts by `(memory_type, memory_key)` and validate action/replace combinations before updates.

Reject in one transaction: insert/update proposal fingerprint and UTC time, then delete the pending row so children cascade. Forget uses a recursive CTE that walks both `supersedes_id` directions, materializes the connected IDs and proposal hashes, deletes child/parent rows in a foreign-key-safe order, deletes matching rejection hashes, and commits once.

- [ ] **Step 6: Verify lifecycle GREEN, schema corruption handling, Ruff, and mypy**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/integration/test_sqlite_memory_store.py -v
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff check src/personal_knowledge_agent/infrastructure/sqlite_memory_store.py tests/integration/test_sqlite_memory_store.py
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run mypy src
```

Expected: all lifecycle tests pass; search tests are added in Task 8.

- [ ] **Step 7: Commit**

```bash
git add src/personal_knowledge_agent/infrastructure/sqlite_memory_store.py tests/integration/test_sqlite_memory_store.py
git commit -m "feat: persist the memory lifecycle"
```

### Task 8: Add active-only hybrid memory retrieval

**Files:**
- Modify: `src/personal_knowledge_agent/infrastructure/sqlite_memory_store.py`
- Modify: `tests/integration/test_sqlite_memory_store.py`

**Interfaces:**
- Consumes: active memory rows, mixed tokenizer, `EmbeddingModel.embed_query`, and `reciprocal_rank_scores`.
- Produces: `SQLiteMemoryStore.search(query, limit=5, memory_type=None) -> list[MemorySearchResult]`.

- [ ] **Step 1: Write failing lexical, semantic, and hybrid tests**

Prove BM25 term frequency, Chinese/mixed tokenization, positive cosine ordering, stable ID tie breaks, RRF cross-ranking behavior, 1..20 limit, optional type filtering, persistence after reopening, empty active store returning `[]` without an embedding call, punctuation-only query rejection, active-only filtering, and corrupt model/dimension/BLOB mapping to safe storage errors.

Use deterministic vectors so one memory ranks first lexically and another semantically; assert the shared candidate ranks first after RRF.

- [ ] **Step 2: Verify retrieval RED**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/integration/test_sqlite_memory_store.py -v
```

Expected: `search` is absent or the new behavioral tests fail.

- [ ] **Step 3: Implement active-only BM25, cosine, and RRF**

Tokenize and validate the query before opening retrieval. Count and average only active records matching the optional type. If none exist, return `[]` before `embed_query`. For nonempty stores, request `candidate_limit = max(20, limit * 4)` from lexical and semantic ranking, pass ordered memory-ID lists to `reciprocal_rank_scores`, and rebuild `MemorySearchResult` values with fused scores.

Require stored embedding model ID and dimensions to match the configured query embedding. Include only cosine scores greater than zero. Do not expose raw lexical/cosine scores after fusion and never return pending or superseded content.

- [ ] **Step 4: Verify all memory-store and existing retrieval tests**

Run:

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/integration/test_sqlite_memory_store.py tests/integration/test_sqlite_keyword_index.py tests/unit/test_retrieval_fusion.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/infrastructure/sqlite_memory_store.py tests/integration/test_sqlite_memory_store.py
git commit -m "feat: search active memories with hybrid retrieval"
```

### Task 9: Add explicit `pka memory` commands

**Files:**
- Modify: `src/personal_knowledge_agent/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `MemoryAdminService`, `LocalMemorySafetyPolicy`, `SQLiteMemoryStore`, `E5Embedder`, and memory enums.
- Produces: `pka memory remember|list|search|approve|reject|forget` and `--memory-db` with default `.pka/memory.db`.

- [ ] **Step 1: Write failing parser tests**

Assert nested command requirement, default path, exact type/status choices, search limit 1..20, required key/content/IDs, and mutually exclusive `--replace`/`--keep-both` on both remember and approve. Confirm `pka ask` and existing `--db` defaults are unchanged.

Use these command forms:

```text
pka memory remember --type goal --key goal.ai_engineer TEXT
pka memory remember --type goal --key goal.ai_engineer --replace mem_old TEXT
pka memory remember --type goal --key goal.ai_engineer --keep-both TEXT
pka memory list [--status pending] [--type project]
pka memory search QUERY [--limit 5] [--type learning_progress]
pka memory approve CANDIDATE [--replace ACTIVE | --keep-both]
pka memory reject CANDIDATE
pka memory forget MEMORY
```

- [ ] **Step 2: Add failing command behavior tests**

Across separate `run_cli` calls and one temporary database, prove explicit remember/list/search uses zero model calls, output shows IDs/type/key/status without hashes/vectors, approve activates, replace displays old as superseded, keep-both preserves two active values, reject removes content, forget reports the hard-delete count, and invalid/unsafe/unknown/conflicting operations return code 1 without traceback.

Stable output families:

```text
Remembered [mem_...] active goal goal.ai_engineer
[mem_...] active goal goal.ai_engineer
1. [score=0.033] [mem_...] goal goal.ai_engineer
Approved [mem_...] active
Rejected [mem_...]
Forgot 2 memory records.
```

Parse generated IDs with `re.search(r"\[(mem_[0-9a-f]{32})\]", output)` and pass the captured value to later CLI calls. This keeps production randomness intact and avoids a test-only composition hook.

- [ ] **Step 3: Implement nested parsers, composition, and formatting**

Add `_DEFAULT_MEMORY_DATABASE = Path(".pka/memory.db")`, one `_memory_store(path)` composition helper, and one handler per action. Convert argparse strings to enums only inside the existing `KnowledgeError` boundary. List newest first; search prints `No active memories found.` as success when empty; bare activation conflict errors include exact active IDs and the two valid follow-up forms.

Do not load LLM settings or construct `OpenAICompatibleChatModel` for any explicit memory command.

- [ ] **Step 4: Verify CLI GREEN and regressions**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_cli.py -v
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff check src/personal_knowledge_agent/cli.py tests/test_cli.py
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run mypy src
```

Expected: all pass, including unchanged Phase 0–2 CLI tests.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/cli.py tests/test_cli.py
git commit -m "feat: add explicit memory CLI controls"
```

### Task 10: Add post-answer memory extraction to `pka agent`

**Files:**
- Modify: `src/personal_knowledge_agent/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: existing successful `AgentResponse`, shared `LanguageModel`, `LLMMemoryExtractor`, `MemoryProposalService`, prefilter, safety policy, and memory store.
- Produces: optional visible pending-candidate/conflict section after a successful agent answer; adds `pka agent --memory-db`.

- [ ] **Step 1: Write failing no-call and candidate tests**

Prove an ordinary successful question leaves Phase 2B output byte-for-byte unchanged and makes no extractor call. For a first-person durable query with a successful retrieve answer, script route, answer, and extraction outputs in order; assert the answer prints first, then:

```text
Memory candidates:
[mem_candidate] pending goal goal.ai_engineer
  Build a portfolio-ready Personal Knowledge Agent.
Approve with:
  pka memory approve mem_candidate
```

Seed an active same-slot record and assert candidate output lists each conflict ID plus `--replace` and `--keep-both` commands. Confirm the candidate remains pending across processes and does not appear in active memory search.

- [ ] **Step 2: Write failing privacy and failure tests**

Prove 4,001-character input, prefilter miss, and raw sensitive hit make zero extraction calls. Prove the extractor call contains only system policy plus current message, never old memories or knowledge evidence. Provider, invalid output, unsafe output, embedding, and memory storage failures print only their safe category after the successful answer and keep return code 0. Verify provider body, key, prompt, vector, traceback, and API key are absent. Unexpected programming exceptions remain visible to tests rather than being converted to a normal skip.

Also prove failed routing/answer/retrieval paths create no candidate and preserve their existing non-zero behavior.

- [ ] **Step 3: Compose extraction strictly after successful output**

Extend `_agent` with `memory_database: Path`. Reuse the already configured `language_model`; do not create a second provider adapter. Call and print `_print_agent_answer(response)` before `MemoryProposalService.propose(query)`. Silence `PREFILTER_MISS` and `NO_NEW_CANDIDATES`; print concise lines for long/sensitive/error skip reasons; print pending records and locally computed active same-slot conflicts for successful candidates.

Do not pass memories into `HybridQuestionRouter`, `AnswerService`, prompt builders, or `AgentResponse`. Do not change `pka ask`.

- [ ] **Step 4: Verify all CLI and application regressions**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_cli.py tests/unit/test_agent_service.py tests/unit/test_answer_service.py tests/unit/test_memory_proposal_service.py tests/unit/test_memory_admin_service.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/personal_knowledge_agent/cli.py tests/test_cli.py
git commit -m "feat: propose memories after agent answers"
```

### Task 11: Complete corruption, rollback, and process-contract coverage

**Files:**
- Modify: `tests/integration/test_sqlite_memory_store.py`
- Modify: `tests/test_cli.py`
- Modify: implementation files only if a new behavioral test exposes a defect.

**Interfaces:**
- Consumes: the completed Phase 3A public interfaces.
- Produces: explicit evidence that privacy, atomicity, restart safety, and Phase 2 compatibility hold at boundaries.

- [ ] **Step 1: Add the remaining adversarial integration matrix**

Add tests for unsupported schema version, corrupt status/type/time/hash/supersedes rows, malformed vector length/type, stored model mismatch, mixed embedding dimensions, embedding count mismatch, duplicate generated ID, SQL constraint failure during a multi-candidate batch, stale replace target after reopening, and a replacement-chain delete started from each chain position.

For every injected write failure, snapshot memories, postings, embeddings, and rejections before the call and assert all four snapshots are unchanged afterward.

- [ ] **Step 2: Add the remaining CLI process-contract matrix**

Assert every expected memory error returns the specified 0 or 1 status, answer/route lines already printed remain visible, no expected failure prints a traceback, and `pka ask`, `pka search`, `pka ingest`, and `pka agent` without candidates retain their prior output contracts.

- [ ] **Step 3: Run targeted tests and implement only evidence-driven corrections**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/integration/test_sqlite_memory_store.py tests/test_cli.py -v
```

Expected: any new test fails for one reproducible boundary reason before its correction, then the same command passes. If a test fails, use the `systematic-debugging` skill before changing implementation. Keep every correction inside the approved design; do not add Phase 3B recall or answer injection.

- [ ] **Step 4: Run the full test suite and commit**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest
```

Expected: all tests pass.

```bash
git add tests/integration/test_sqlite_memory_store.py tests/test_cli.py src/personal_knowledge_agent/infrastructure/sqlite_memory_store.py src/personal_knowledge_agent/application/memory_proposal.py src/personal_knowledge_agent/cli.py
git commit -m "test: harden long-term memory boundaries"
```

### Task 12: Document Phase 3A and run the merge gate

**Files:**
- Modify: `README.md`
- Create: `docs/learning-log/phase-3a.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: completed CLI behavior and approved design.
- Produces: user documentation, portfolio learning notes, accurate contributor scope, and merge-ready verification evidence.

- [ ] **Step 1: Update README usage and privacy documentation**

Document every explicit memory command, `pka agent --memory-db`, pending confirmation, replacement/keep-both/reject/forget semantics, active-only search, the `.pka/memory.db` default, and the Phase 3A boundary that memory does not yet influence answers. State that the database is unencrypted and hard deletion does not guarantee physical secure erasure from SQLite pages, snapshots, caches, or provider systems.

Document the call-cost model explicitly: a prefilter miss adds zero model calls; an allowed hit adds one extraction request after a successful answer, subject to provider retry behavior; every explicit memory command makes zero remote model calls but local writes/search may load E5.

- [ ] **Step 2: Write the Phase 3A learning log**

Explain, with small concrete examples:

- long-term memory versus chat history and document knowledge;
- why an LLM proposes but does not receive write authority;
- pending/active/superseded transitions and transactional replacement;
- stable keys versus semantic conflict thresholds;
- local prefilter and two-stage safety screening;
- dedicated SQLite schema, hybrid memory retrieval, and rejection versus forgetting;
- provider disclosure and secure-erasure limitations;
- what Phase 3B will add: active-memory recall, prompt injection, and visible `M1...Mn` use.

- [ ] **Step 3: Update contributor scope**

In `AGENTS.md`, state Phase 2B and Phase 3A are complete. Move automatic retrieval/injection of active memories to Phase 3B future scope. Keep PDF/web/code ingestion, knowledge graph, evaluation, and deployment listed as future work.

- [ ] **Step 4: Run the complete quality gate**

```bash
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff format --check .
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run ruff check .
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run mypy src
TMPDIR=/tmp UV_PROJECT_ENVIRONMENT=/tmp/pka-phase3a-venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pka health
```

Expected: full suite passes, Ruff reports no formatting or lint changes, mypy reports success, and health prints `Personal Knowledge Agent is healthy.`

- [ ] **Step 5: Inspect scope and commit documentation**

Run:

```bash
git status --short
git diff --check
git diff --cached --name-only
```

Expected: no whitespace errors; `.pka/`, `.env`, `papers/`, `references/`, and `research/` are not staged.

```bash
git add README.md AGENTS.md docs/learning-log/phase-3a.md
git commit -m "docs: explain confirmation-gated memory"
```

- [ ] **Step 6: Request final code review before integration**

Use the `requesting-code-review` skill against the approved design and this plan. Resolve only verified in-scope findings, rerun the full quality gate, then use `finishing-a-development-branch` to present merge options.

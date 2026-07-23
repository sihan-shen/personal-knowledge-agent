# Phase 6C Local FastAPI RAG API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a local FastAPI `GET /health` and grounded-RAG `POST /ask` API backed by a public, reproducible Markdown demo corpus.

**Architecture:** Keep FastAPI in a top-level adapter module. Extract one shared outer-layer composition root so the CLI and API construct the same existing `AnswerService`; keep domain and application packages framework-independent. The API checks database and model configuration readiness before invoking the service and serializes only the validated `GroundedAnswer` fields.

**Tech Stack:** Python 3.12+, FastAPI, Uvicorn, HTTPX TestClient, pytest, existing SQLite/BM25/E5 hybrid retrieval, OpenAI-compatible Chat Completions.

## Global Constraints

- Bind only to `127.0.0.1`; Phase 6C performs no cloud deployment or Docker entrypoint change.
- Use `fastapi`, `uvicorn`, and `httpx`; do not add a configuration framework or automatic `.env` loader.
- API supports only existing grounded RAG, fixed `SearchMode.HYBRID`, and a fixed evidence limit of five.
- The demo Markdown corpus is committed under `docs/demo-knowledge/`; `.pka/demo-knowledge.db` remains ignored and is built explicitly through existing `pka ingest` commands.
- `POST /ask` error payloads must never reveal `PKA_LLM_*` values, provider responses, prompts, or secrets.
- Preserve domain/application dependency boundaries and use complete annotations under strict mypy.
- Run mounted-filesystem commands with `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache`.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `src/personal_knowledge_agent/composition.py` | Construct the existing `AnswerService` from a database path and supplied environment. |
| `src/personal_knowledge_agent/api.py` | FastAPI app factory, Pydantic contracts, readiness checks, and safe exception mapping. |
| `src/personal_knowledge_agent/cli.py` | Reuse the shared grounded-answer composition root. |
| `tests/test_composition.py` | Verify composition creates the expected application service with deterministic patched infrastructure. |
| `tests/test_api.py` | Verify HTTP contracts through `fastapi.testclient.TestClient` with fake answerers. |
| `docs/demo-knowledge/retrieval.md` | Public lesson about chunking, embeddings, and hybrid retrieval. |
| `docs/demo-knowledge/grounded-answers.md` | Public lesson about evidence, citations, and refusal. |
| `docs/demo-knowledge/prompt-engineering.md` | Public lesson about structured prompts and output validation. |
| `pyproject.toml`, `uv.lock` | Runtime FastAPI/Uvicorn and test HTTPX dependencies. |
| `README.md` | Explicit local corpus build, environment, server, and `/docs` instructions. |
| `docs/learning-log/phase-6c.md` | API-adapter and readiness learning notes. |

---

### Task 1: Extract the shared grounded-answer composition root

**Files:**
- Create: `src/personal_knowledge_agent/composition.py`
- Create: `tests/test_composition.py`
- Modify: `src/personal_knowledge_agent/cli.py:629-640`

**Interfaces:**
- Consumes: `LLMSettings`, `load_llm_settings`, `SQLiteKeywordIndex`, `E5Embedder`, `OpenAICompatibleChatModel`, `SearchService`, and `AnswerService`.
- Produces: `create_grounded_answer_service(database: Path, environment: Mapping[str, str]) -> AnswerService` for both CLI and API adapters.

- [ ] **Step 1: Write the failing composition test**

Create `tests/test_composition.py`:

```python
"""Tests for the outer grounded-answer composition root."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from personal_knowledge_agent.application.answer import AnswerService
from personal_knowledge_agent.composition import create_grounded_answer_service
from personal_knowledge_agent.domain.models import ModelOutput
from tests.helpers import FakeEmbedder, FakeLanguageModel


def test_create_grounded_answer_service_wires_existing_dependencies(
    tmp_path: Path,
) -> None:
    with (
        patch(
            "personal_knowledge_agent.composition.E5Embedder",
            return_value=FakeEmbedder({}, default_vector=(1.0, 0.0)),
        ),
        patch(
            "personal_knowledge_agent.composition.OpenAICompatibleChatModel",
            return_value=FakeLanguageModel(
                ModelOutput("fake-model", "{}", "stop")
            ),
        ),
    ):
        service = create_grounded_answer_service(
            tmp_path / "knowledge.db",
            {"PKA_LLM_API_KEY": "test-key"},
        )

    assert isinstance(service, AnswerService)
```

- [ ] **Step 2: Run the test to verify RED**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_composition.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'personal_knowledge_agent.composition'`.

- [ ] **Step 3: Implement the composition root and migrate CLI construction**

Create `src/personal_knowledge_agent/composition.py`:

```python
"""Outer-layer factories shared by executable adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from personal_knowledge_agent.application.answer import AnswerService
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.infrastructure.e5_embedder import E5Embedder
from personal_knowledge_agent.infrastructure.llm_settings import load_llm_settings
from personal_knowledge_agent.infrastructure.openai_compatible_chat_model import (
    OpenAICompatibleChatModel,
)
from personal_knowledge_agent.infrastructure.sqlite_keyword_index import (
    SQLiteKeywordIndex,
)


def create_grounded_answer_service(
    database: Path,
    environment: Mapping[str, str],
) -> AnswerService:
    """Construct the existing grounded-answer service from explicit runtime inputs."""
    settings = load_llm_settings(environment)
    index = SQLiteKeywordIndex(database, E5Embedder())
    language_model = OpenAICompatibleChatModel(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model_id=settings.model_id,
    )
    return AnswerService(SearchService(index), language_model)
```

Replace the construction in `cli._ask` with:

```python
answer = create_grounded_answer_service(database, os.environ).answer(
    query,
    limit=limit,
    mode=mode,
)
```

Remove now-unused CLI imports of `AnswerService`, `SearchService`, `E5Embedder`,
`load_llm_settings`, `OpenAICompatibleChatModel`, and `SQLiteKeywordIndex` only if
they are not used by another CLI command.

- [ ] **Step 4: Run focused tests and static checks**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_composition.py tests/test_cli.py -q`

Expected: PASS.

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/personal_knowledge_agent/composition.py src/personal_knowledge_agent/cli.py tests/test_composition.py`

Expected: `All checks passed!`

- [ ] **Step 5: Commit the composition root**

```bash
git add src/personal_knowledge_agent/composition.py src/personal_knowledge_agent/cli.py tests/test_composition.py
git commit -m "refactor: share grounded answer composition"
```

### Task 2: Add the local FastAPI readiness and grounded-answer API

**Files:**
- Create: `src/personal_knowledge_agent/api.py`
- Create: `tests/test_api.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `create_grounded_answer_service`, `GroundedQuestionAnswerer`, `GroundedAnswer`, `Evidence`, `AnswerStatus`, and existing domain errors.
- Produces: `create_app(*, database: Path = DEFAULT_DEMO_DATABASE, environment: Mapping[str, str] | None = None, answer_service_factory: Callable[[], GroundedQuestionAnswerer] | None = None) -> FastAPI`.

- [ ] **Step 1: Add FastAPI runtime/test dependencies and write failing API tests**

Add these dependencies:

```toml
dependencies = [
    "fastapi>=0.115.0",
    "openai>=2.0.0",
    "sentence-transformers>=5.0.0",
    "torch>=2.0.0",
    "uvicorn>=0.30.0",
]

[dependency-groups]
dev = [
    "httpx>=0.27.0",
    "mypy>=1.16.0",
    "pytest>=8.3.0",
    "ruff>=0.11.0",
]
```

Create `tests/test_api.py` with an API fixture using a real temporary database file,
an explicit environment mapping, and a fake protocol implementation:

```python
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from personal_knowledge_agent.api import create_app
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    Evidence,
    GroundedAnswer,
    SearchMode,
)


class FakeAnswerer:
    def answer(self, question: str, *, limit: int, mode: SearchMode) -> GroundedAnswer:
        evidence = Evidence("S1", "chunk-1", "retrieval.md", ("Hybrid retrieval",), "Evidence text", 1.0)
        return GroundedAnswer(AnswerStatus.ANSWERED, "Hybrid retrieval combines rankings. [S1]", ("S1",), (evidence,), "fake-model")


def test_health_reports_degraded_when_model_configuration_is_missing(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    database.touch()
    client = TestClient(create_app(database=database, environment={}))

    assert client.get("/health").json() == {
        "status": "degraded",
        "knowledge_base_ready": True,
        "model_configured": False,
    }


def test_ask_serializes_validated_grounded_citations(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    database.touch()
    client = TestClient(
        create_app(
            database=database,
            environment={"PKA_LLM_API_KEY": "test-key"},
            answer_service_factory=FakeAnswerer,
        )
    )

    response = client.post("/ask", json={"question": "What is hybrid retrieval?"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "answered",
        "answer": "Hybrid retrieval combines rankings. [S1]",
        "citations": [{"id": "S1", "source": "retrieval.md", "heading_path": ["Hybrid retrieval"]}],
        "model_id": "fake-model",
    }
```

Add tests for a whitespace-only question (`422`), a missing database (`503`
`knowledge_base_not_ready`), missing configuration (`503` `model_not_configured`),
and a fake answerer raising `LanguageModelError` (`502` `model_request_failed`).

- [ ] **Step 2: Verify RED**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv sync --all-groups`

Expected: dependency installation succeeds and updates `uv.lock`.

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_api.py -v`

Expected: FAIL because `personal_knowledge_agent.api` does not exist.

- [ ] **Step 3: Implement the typed API factory and safe mappings**

Create `src/personal_knowledge_agent/api.py` using this complete adapter shape:

```python
"""Local HTTP adapter for the grounded RAG workflow."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from personal_knowledge_agent.composition import create_grounded_answer_service
from personal_knowledge_agent.domain.errors import (
    ConfigurationError,
    IndexStorageError,
    KnowledgeBaseEmptyError,
    LanguageModelError,
    StructuredOutputError,
)
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    GroundedAnswer,
    SearchMode,
)
from personal_knowledge_agent.domain.protocols import GroundedQuestionAnswerer
from personal_knowledge_agent.infrastructure.llm_settings import load_llm_settings


DEFAULT_DEMO_DATABASE = Path(".pka/demo-knowledge.db")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class CitationResponse(BaseModel):
    id: str
    source: str
    heading_path: list[str]


class AskResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[CitationResponse]
    model_id: str | None


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded"]
    knowledge_base_ready: bool
    model_configured: bool


def create_app(
    *,
    database: Path = DEFAULT_DEMO_DATABASE,
    environment: Mapping[str, str] | None = None,
    answer_service_factory: Callable[[], GroundedQuestionAnswerer] | None = None,
) -> FastAPI:
    """Create a local-only grounded RAG API with explicit dependencies."""
    selected_environment = os.environ if environment is None else environment
    selected_database = database.expanduser()
    app = FastAPI(title="Personal Knowledge Agent API", version="0.1.0")

    def model_configured() -> bool:
        try:
            load_llm_settings(selected_environment)
        except ConfigurationError:
            return False
        return True

    def health_response() -> HealthResponse:
        knowledge_base_ready = selected_database.is_file()
        configured = model_configured()
        return HealthResponse(
            status="ready" if knowledge_base_ready and configured else "degraded",
            knowledge_base_ready=knowledge_base_ready,
            model_configured=configured,
        )

    def require_answerer() -> GroundedQuestionAnswerer:
        if not selected_database.is_file():
            raise HTTPException(
                status_code=503,
                detail={"code": "knowledge_base_not_ready"},
            )
        if not model_configured():
            raise HTTPException(
                status_code=503,
                detail={"code": "model_not_configured"},
            )
        if answer_service_factory is not None:
            return answer_service_factory()
        return create_grounded_answer_service(selected_database, selected_environment)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return health_response()

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            answer = require_answerer().answer(
                request.question,
                limit=5,
                mode=SearchMode.HYBRID,
            )
        except HTTPException:
            raise
        except (KnowledgeBaseEmptyError, IndexStorageError) as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "knowledge_base_not_ready"},
            ) from error
        except (LanguageModelError, StructuredOutputError) as error:
            raise HTTPException(
                status_code=502,
                detail={"code": "model_request_failed"},
            ) from error
        return _serialize_answer(answer)

    return app


def _serialize_answer(answer: GroundedAnswer) -> AskResponse:
    return AskResponse(
        status=(
            "answered"
            if answer.status is AnswerStatus.ANSWERED
            else "insufficient_evidence"
        ),
        answer=answer.answer,
        citations=[
            CitationResponse(
                id=evidence.citation_id,
                source=evidence.source,
                heading_path=list(evidence.heading_path),
            )
            for evidence in answer.evidence
        ],
        model_id=answer.model_id,
    )
```

Inside the factory, use `selected_environment = os.environ if environment is None else environment`.
Set `knowledge_base_ready` with `database.is_file()`. Set `model_configured` by
calling `load_llm_settings(selected_environment)` and catching only
`ConfigurationError`; never include its message in a response. `/health` returns
`ready` only when both flags are true, otherwise `degraded`.

For `/ask`, check database first, then configuration. Use the injected factory when
provided; otherwise create one that calls `create_grounded_answer_service(database,
selected_environment)`. Call `.answer(question, limit=5, mode=SearchMode.HYBRID)`.
Serialize each `Evidence` as `id`, `source`, and `heading_path` only. Map
`KnowledgeBaseEmptyError` and `IndexStorageError` to a `503` body
`{"detail":{"code":"knowledge_base_not_ready"}}`; map `LanguageModelError` and
`StructuredOutputError` to `502` body `{"detail":{"code":"model_request_failed"}}`.

- [ ] **Step 4: Verify GREEN, OpenAPI, and static checks**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_api.py -q`

Expected: all API tests pass without network access or real E5 construction.

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check src/personal_knowledge_agent/api.py tests/test_api.py`

Expected: both files are formatted.

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run mypy src`

Expected: `Success: no issues found`.

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from fastapi.testclient import TestClient; from personal_knowledge_agent.api import create_app; assert TestClient(create_app()).get('/openapi.json').status_code == 200"`

Expected: exit code 0.

- [ ] **Step 5: Commit the API adapter**

```bash
git add pyproject.toml uv.lock src/personal_knowledge_agent/api.py tests/test_api.py
git commit -m "feat: add local grounded RAG API"
```

### Task 3: Add reproducible demo material and local operator documentation

**Files:**
- Create: `docs/demo-knowledge/retrieval.md`
- Create: `docs/demo-knowledge/grounded-answers.md`
- Create: `docs/demo-knowledge/prompt-engineering.md`
- Create: `docs/learning-log/phase-6c.md`
- Modify: `README.md`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: existing `pka ingest <path> --db .pka/demo-knowledge.db`, API factory from Task 2, and committed demo Markdown.
- Produces: public teaching material that can be ingested locally and documented commands that start the API only on loopback.

- [ ] **Step 1: Write failing corpus-ingestion test**

Add this parameterized test to `tests/test_api.py`:

```python
@pytest.mark.parametrize(
    "filename",
    ["retrieval.md", "grounded-answers.md", "prompt-engineering.md"],
)
def test_demo_documents_can_be_ingested(filename: str, tmp_path: Path) -> None:
    source = ROOT / "docs" / "demo-knowledge" / filename
    index = SQLiteKeywordIndex(
        tmp_path / "demo.db",
        FakeEmbedder({}, default_vector=(1.0, 0.0)),
    )

    result = IngestService(MarkdownLoader(), MarkdownChunker(), index).ingest(source)

    assert result.chunk_count >= 2
```

Define `ROOT = Path(__file__).resolve().parents[1]` and import `pytest`,
`IngestService`, `MarkdownLoader`, `MarkdownChunker`, `SQLiteKeywordIndex`, and
`FakeEmbedder` in the test module.

- [ ] **Step 2: Verify RED**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_api.py::test_demo_documents_can_be_ingested -v`

Expected: FAIL because the three demo Markdown files do not exist.

- [ ] **Step 3: Write the public demo corpus and operator guide**

Create three public documents with these headings and factual teaching content:

```markdown
# Retrieval and hybrid ranking

## Chunking creates retrievable evidence

RAG stores documents as smaller chunks so a query can retrieve a focused passage rather than an entire file. A useful chunk preserves its source and heading path, allowing the answer layer to cite where its evidence came from.

## Hybrid retrieval combines complementary signals

Lexical retrieval is strong when a query contains exact terms. Semantic retrieval compares embedding vectors and can recover related wording. Reciprocal rank fusion combines rank positions from both lists; it does not assume that raw BM25 and cosine scores share a scale.
```

```markdown
# Grounded answers and citations

## Evidence constrains the answer

A grounded answer is generated from retrieved chunks rather than from unsupported model memory. If no relevant evidence is retrieved, the safe result is an insufficient-evidence response instead of an invented answer.

## Citations are request-local identifiers

The prompt gives each evidence chunk a label such as S1. The structured answer must cite only labels supplied for that request. Validation rejects labels that do not match the retrieved evidence.
```

```markdown
# Prompt engineering for reliable RAG

## Instructions define the output contract

Prompts should state which evidence may be used, when the model must refuse, and the required JSON fields. A clear contract makes the output easier to parse and test.

## Validation completes the reliability boundary

Prompt instructions alone are not enforcement. The application parses the structured response and validates its status, citation IDs, and citation markers before returning an answer to a caller.
```

Append a README section with exact local commands:

```bash
uv sync --all-groups
uv run pka ingest docs/demo-knowledge/retrieval.md --db .pka/demo-knowledge.db
uv run pka ingest docs/demo-knowledge/grounded-answers.md --db .pka/demo-knowledge.db
uv run pka ingest docs/demo-knowledge/prompt-engineering.md --db .pka/demo-knowledge.db
export PKA_LLM_API_KEY="your-key"
export PKA_LLM_BASE_URL="https://api.deepseek.com"
export PKA_LLM_MODEL="deepseek-v4-flash"
uv run uvicorn personal_knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000
```

Document `http://127.0.0.1:8000/docs`, the `curl` request
`curl -X POST http://127.0.0.1:8000/ask -H 'content-type: application/json' -d '{"question":"What is hybrid retrieval?"}'`, the `503` behavior when configuration is absent, and that `.env` is not loaded automatically.

Create `docs/learning-log/phase-6c.md` explaining API adapters versus domain
services, application factories/dependency injection, readiness versus liveness,
and why external-provider tests use fakes.

- [ ] **Step 4: Verify corpus and full quality gate**

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_api.py -q`

Expected: PASS, including all three corpus-ingestion tests.

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

Expected: full suite passes.

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .`

Expected: all project files are formatted.

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`

Expected: `All checks passed!`

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run mypy src`

Expected: `Success: no issues found`.

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv run pka health`

Expected: `Personal Knowledge Agent is healthy.`

Run: `TMPDIR=/tmp UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache uv lock --check`

Expected: exit code 0.

- [ ] **Step 5: Commit local API documentation and corpus**

```bash
git add docs/demo-knowledge docs/learning-log/phase-6c.md README.md tests/test_api.py
git commit -m "docs: add local RAG API demo guide"
```

## Plan self-review

- Spec coverage: Task 1 implements the shared composition boundary; Task 2 covers
  the two endpoints, readiness, safe error codes, local-only factory, dependencies,
  and deterministic HTTP tests; Task 3 covers public corpus, explicit ingestion,
  README, learning log, and the full quality gate.
- Placeholder scan: all file paths, code contracts, test cases, commands, expected
  results, and commit messages are explicit; no deferred implementation markers are
  present.
- Type consistency: both adapters consume
  `create_grounded_answer_service(database, environment) -> AnswerService`; API
  injection uses `Callable[[], GroundedQuestionAnswerer]`; `POST /ask` always calls
  the existing `.answer(question, limit=5, mode=SearchMode.HYBRID)` contract.

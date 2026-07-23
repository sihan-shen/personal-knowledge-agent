# Phase 6D: Local API Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add privacy-safe request correlation and structured local diagnostics to
the FastAPI grounded-RAG API without changing its answer contract.

**Architecture:** A small HTTP middleware in `api.py` validates or generates a
request ID, measures monotonic elapsed time, adds `X-Request-ID` to handled
responses, and emits one JSON completion event through the standard-library
project logger. Pure helpers keep request-ID validation, outcome mapping, and
record construction independent of FastAPI endpoint logic.

**Tech Stack:** Python 3.12+, FastAPI/Starlette, standard-library `logging`,
`json`, `re`, `time`, and `uuid`, Pytest with FastAPI `TestClient` and `caplog`.

## Global Constraints

- Use Python 3.12+ syntax, complete annotations, and strict mypy.
- Keep domain and application packages independent of FastAPI and logging.
- Add no third-party dependency, cloud service, persistent telemetry store,
  metric endpoint, or debug endpoint.
- Log only fixed safe metadata: event, request ID, HTTP method, path, status,
  outcome, duration, and an unhandled exception class name when applicable.
- Never log questions, answers, citations, retrieved text, request/response
  bodies, query strings, arbitrary headers, environment values, API keys, or
  exception messages.
- Preserve existing `/health`, `/ask`, OpenAPI, error-body, and CLI behavior;
  `X-Request-ID` is the only additive HTTP contract.
- Keep tests offline and deterministic; do not load E5 weights or call a model.
- Run project commands with `TMPDIR=/tmp UV_LINK_MODE=copy` on this
  Windows-mounted workspace.

---

## File Structure

| Path | Change | Responsibility |
| --- | --- | --- |
| `src/personal_knowledge_agent/api.py` | Modify | Safe request-ID helpers and HTTP middleware around the existing endpoints. |
| `tests/test_api.py` | Modify | Deterministic header and captured-log contract coverage. |
| `README.md` | Modify | Local command and response-to-log correlation guidance. |
| `docs/learning-log/phase-6d.md` | Create | Learning notes for observability and privacy boundaries. |

### Task 1: Safe request middleware and API contract tests

**Files:**
- Modify: `src/personal_knowledge_agent/api.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `create_app(...) -> FastAPI`, existing FastAPI `Request`/`Response`
  lifecycle, and `personal_knowledge_agent.api` standard-library logger.
- Produces: `X-Request-ID` on every handled response and one JSON
  `api.request.completed` record per request.

- [ ] **Step 1: Write failing HTTP/logging contract tests**

Add `json` and `logging` imports to `tests/test_api.py`. Add a helper that
selects exactly one `personal_knowledge_agent.api` record from `caplog.records`
and decodes its `message`:

```python
def _request_record(caplog: pytest.LogCaptureFixture) -> dict[str, object]:
    records = [
        record
        for record in caplog.records
        if record.name == "personal_knowledge_agent.api"
    ]
    assert len(records) == 1
    decoded = json.loads(records[0].message)
    assert isinstance(decoded, dict)
    return decoded
```

Add these behavioral tests using existing temporary SQLite helpers and
deterministic answerers:

```python
def test_api_preserves_a_valid_request_id_and_logs_safe_completion(...):
    caplog.set_level(logging.INFO, logger="personal_knowledge_agent.api")
    response = client.get("/health", headers={"X-Request-ID": "demo-123"})
    assert response.headers["x-request-id"] == "demo-123"
    record = _request_record(caplog)
    assert record["event"] == "api.request.completed"
    assert record["request_id"] == "demo-123"
    assert record["method"] == "GET"
    assert record["path"] == "/health"
    assert record["status_code"] == 200
    assert record["outcome"] == "success"
    assert isinstance(record["duration_ms"], float)
    assert record["duration_ms"] >= 0

def test_api_replaces_an_invalid_request_id(...):
    response = client.get("/health", headers={"X-Request-ID": "invalid value"})
    assert response.headers["x-request-id"] != "invalid value"
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])

def test_api_logs_not_ready_outcome_and_omits_question(...):
    caplog.set_level(logging.INFO, logger="personal_knowledge_agent.api")
    question = "credential-like-value-must-not-appear"
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 503
    assert _request_record(caplog)["outcome"] == "not_ready"
    assert question not in caplog.text

def test_api_logs_model_error_outcome(...):
    caplog.set_level(logging.INFO, logger="personal_knowledge_agent.api")
    response = client.post("/ask", json={"question": "What is hybrid retrieval?"})
    assert response.status_code == 502
    assert _request_record(caplog)["outcome"] == "model_error"

def test_api_adds_request_id_to_validation_errors(...):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])
```

Use a client with a missing database for the `not_ready` test so the secret-like
question cannot reach an answerer. The tests intentionally inspect metadata, not
the exact elapsed time.

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/test_api.py -q
```

Expected: failures because `X-Request-ID` and the project JSON log records do
not yet exist.

- [ ] **Step 3: Implement minimal request observation in `api.py`**

Add these imports and module constants near the existing imports:

```python
import json
import logging
import re
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.middleware.base import RequestResponseEndpoint

_API_LOGGER = logging.getLogger("personal_knowledge_agent.api")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
```

Add pure helpers below `DEFAULT_DEMO_DATABASE`:

```python
def _select_request_id(candidate: str | None) -> str:
    if candidate is not None and _REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return uuid4().hex


def _outcome_for_status(status_code: int) -> str:
    if 200 <= status_code < 400:
        return "success"
    if status_code == 502:
        return "model_error"
    if status_code == 503:
        return "not_ready"
    if 400 <= status_code < 500:
        return "client_error"
    return "server_error"


def _log_completed_request(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    exception_type: str | None = None,
) -> None:
    record: dict[str, str | int | float] = {
        "event": "api.request.completed",
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "outcome": _outcome_for_status(status_code),
        "duration_ms": round(duration_ms, 2),
    }
    if exception_type is not None:
        record["exception_type"] = exception_type
    _API_LOGGER.info("%s", json.dumps(record, sort_keys=True))
```

Inside `create_app`, after creating `app`, install one decorator-based middleware.
Use `request.url.path`, never the full URL. Preserve FastAPI's handling of
unexpected failures by re-raising them, while logging a 500 classification in
the `except` branch:

```python
    @app.middleware("http")
    async def observe_request(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = _select_request_id(request.headers.get("X-Request-ID"))
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception as error:
            _log_completed_request(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=(perf_counter() - started_at) * 1_000,
                exception_type=type(error).__name__,
            )
            raise
        response.headers["X-Request-ID"] = request_id
        _log_completed_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=(perf_counter() - started_at) * 1_000,
        )
        return response
```

Use FastAPI/Starlette's `RequestResponseEndpoint` type exactly as shown. Do not
log the exception message or any request content.

- [ ] **Step 4: Run focused tests and static checks**

Run:

```bash
TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest -W error tests/test_api.py -q
TMPDIR=/tmp UV_LINK_MODE=copy uv run ruff format --check src/personal_knowledge_agent/api.py tests/test_api.py
TMPDIR=/tmp UV_LINK_MODE=copy uv run ruff check src/personal_knowledge_agent/api.py tests/test_api.py
TMPDIR=/tmp UV_LINK_MODE=copy uv run mypy src
```

Expected: all API tests pass with no warnings; Ruff and mypy report no errors.

- [ ] **Step 5: Commit the implementation**

```bash
git add src/personal_knowledge_agent/api.py tests/test_api.py
git commit -m "feat: add local API request diagnostics"
```

### Task 2: Local operator guide and learning log

**Files:**
- Modify: `README.md`
- Create: `docs/learning-log/phase-6d.md`

**Interfaces:**
- Consumes: The `X-Request-ID` header and `api.request.completed` JSON record
  from Task 1.
- Produces: A reproducible local observation workflow without exposing secrets.

- [ ] **Step 1: Add documentation assertions before prose changes**

Extend `tests/test_api.py` with a README-content test that reads the project
root `README.md` and asserts it documents all stable operator terms:

```python
def test_readme_documents_local_api_request_diagnostics() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "X-Request-ID" in readme
    assert "api.request.completed" in readme
    assert "question text" in readme
```

- [ ] **Step 2: Run the documentation assertion to verify it fails**

Run:

```bash
TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest tests/test_api.py::test_readme_documents_local_api_request_diagnostics -q
```

Expected: failure because the Phase 6C README has no request-diagnostics section.

- [ ] **Step 3: Document the safe local workflow**

Add a `### Inspect local API requests` subsection directly after the existing
local API `curl` example in `README.md`. Include this exact command shape:

```bash
curl -i -X POST http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -H 'X-Request-ID: study-demo-01' \
  -d '{"question":"What is hybrid retrieval?"}'
```

Explain that `-i` displays the returned `X-Request-ID`; Uvicorn's terminal
receives one `api.request.completed` JSON record whose `request_id` matches it;
and the record contains only request metadata. State explicitly that questions,
answers, evidence, headers, API keys, and `.env` values are not logged. Mention
that malformed supplied request IDs are replaced, and that no metrics endpoint
or cloud telemetry is enabled.

Create `docs/learning-log/phase-6d.md` with concise sections covering:

```markdown
# Phase 6D: Local API Observability

## What changed
...

## Engineering lessons
- Correlation IDs connect an HTTP response to a log event without exposing data.
- A monotonic clock measures elapsed work safely across wall-clock adjustments.
- Structured logs are a contract; fixed fields support filtering and tests.
- Privacy is designed by allowlisting metadata rather than redacting full payloads.
```

- [ ] **Step 4: Run documentation and complete quality checks**

Run:

```bash
TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest -W error tests/test_api.py -q
TMPDIR=/tmp UV_LINK_MODE=copy uv run pytest
TMPDIR=/tmp UV_LINK_MODE=copy uv run ruff format --check .
TMPDIR=/tmp UV_LINK_MODE=copy uv run ruff check .
TMPDIR=/tmp UV_LINK_MODE=copy uv run mypy src
TMPDIR=/tmp UV_LINK_MODE=copy uv run pka health
TMPDIR=/tmp UV_LINK_MODE=copy uv lock --check
```

Expected: all commands succeed; tests make no live provider or model call.

- [ ] **Step 5: Commit documentation and learning notes**

```bash
git add README.md docs/learning-log/phase-6d.md tests/test_api.py
git commit -m "docs: explain local API diagnostics"
```

## Plan self-review

- **Spec coverage:** Task 1 covers safe correlation, generated/replaced IDs,
  one JSON event, status outcomes, metadata-only logging, and contract
  preservation. Task 2 covers the operator workflow and learning explanation.
  No spec requirement is unassigned.
- **Placeholder scan:** The plan has no open placeholder, deferred
  implementation, or unspecified test requirement.
- **Type consistency:** `_select_request_id`, `_outcome_for_status`, and
  `_log_completed_request` are defined before their middleware consumer; tests
  consume only the logger name and public response header they produce.

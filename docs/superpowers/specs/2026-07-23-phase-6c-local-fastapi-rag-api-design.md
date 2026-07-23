# Phase 6C: Local FastAPI RAG API Design

**Date:** 2026-07-23

**Status:** Approved design

## Goal and scope

Expose the existing grounded RAG workflow through a small, local-only FastAPI
service. The service provides `GET /health` and `POST /ask`, backed by a public,
reproducible set of RAG teaching Markdown documents. It is a local learning and
portfolio demonstration, not a cloud deployment.

The API wraps only the existing `pka ask` grounded-answer behavior. It does not
expose Agent routing, memory, graph operations, uploads, knowledge-base
administration, authentication, background work, Docker serving, or deployment.

## Decisions

- Use FastAPI and Uvicorn. Start locally on `127.0.0.1:8000` with Uvicorn's
  application-factory mode.
- `POST /ask` accepts only `{"question": "..."}`. It uses the existing hybrid
  retrieval mode and evidence limit of five; these are intentionally not public
  request knobs in this MVP.
- Keep public RAG teaching Markdown sources under `docs/demo-knowledge/`.
  Generate the ignored `.pka/demo-knowledge.db` explicitly with the existing
  `pka ingest` command before starting the server.
- Add a top-level composition root shared by CLI and API. It creates the existing
  `AnswerService` from a database path and a supplied environment mapping; domain
  and application packages remain independent of FastAPI, SQLite, E5, and the
  OpenAI-compatible runtime.
- Do not automatically read `.env`. The operator supplies `PKA_LLM_API_KEY`,
  optional `PKA_LLM_BASE_URL`, and optional `PKA_LLM_MODEL` through the process
  environment before starting the server.
- Missing LLM configuration does not prevent server startup. The service remains
  inspectable through `/health`; `/ask` returns a structured `503` until the
  configuration is corrected.
- Do not change the Dockerfile or deploy this service in Phase 6C.

## API contract

```text
GET /health
  200 {"status":"ready"|"degraded", "knowledge_base_ready": bool,
       "model_configured": bool}

POST /ask
  request:  {"question":"What is hybrid retrieval?"}
  response: {"status":"answered"|"insufficient_evidence", "answer":"...",
             "citations":[{"id":"S1","source":"...","heading_path":["..."]}],
             "model_id":"..."}
```

`question` is trimmed and must not be blank. Request-shape and validation errors
use FastAPI's `422` response. A missing demo database produces `503`
`knowledge_base_not_ready`; absent or invalid LLM configuration produces `503`
`model_not_configured`; provider and structured-output failures produce `502`
`model_request_failed`. Error bodies never include environment values, provider
responses, prompts, or secrets.

## Data flow

```text
docs/demo-knowledge/*.md
        |
        v  (explicit pka ingest)
.pka/demo-knowledge.db
        |
        v
FastAPI create_app()
  |-- GET /health: inspect database/configuration readiness
  `-- POST /ask: composition root -> AnswerService
                               |
                               v
                 SearchService -> SQLiteKeywordIndex / E5
                               |
                               v
                    OpenAI-compatible LLM -> GroundedAnswer JSON
```

The API serializes the existing validated `GroundedAnswer`. Citation labels and
sources come from its request-local evidence; the API does not invent citations.

## File structure and tests

| Path | Responsibility |
| --- | --- |
| `src/personal_knowledge_agent/composition.py` | Shared outer-layer construction of `AnswerService`. |
| `src/personal_knowledge_agent/api.py` | App factory, HTTP models, endpoints, and exception mapping. |
| `docs/demo-knowledge/*.md` | Public RAG tutorial corpus. |
| `tests/test_api.py` | Offline API contract tests using deterministic fakes. |
| `pyproject.toml` | FastAPI and Uvicorn dependencies. |
| `README.md` | Local build, environment, run, and OpenAPI instructions. |
| `docs/learning-log/phase-6c.md` | Learning notes for API adapters and readiness. |

Tests must cover ready and degraded health responses; blank questions; missing
database and missing configuration `503`s; mapped provider failures; exact
citation serialization; and ingestion of each public demo document. They must not
make network calls, load real E5 weights, or require real provider credentials.

## Acceptance criteria

1. A developer can build the demo database with documented explicit `pka ingest`
   commands and run the API locally with `uv run uvicorn
   personal_knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000`.
2. `/docs` exposes the generated OpenAPI documentation.
3. `/health` reports `ready` only when the demo SQLite database is readable,
   contains at least one indexed chunk, and valid model configuration is available;
   otherwise it reports `degraded` without exposing configuration values.
4. A configured `/ask` request returns a validated grounded-answer JSON response
   with citations; unavailable prerequisites return the specified safe errors.
5. The full Python quality gate passes with deterministic tests.

## Out of scope

Cloud deployment, Docker entrypoint changes, CORS, authentication, rate limiting,
streaming, uploads, mutable knowledge-base APIs, Agent routing, memory, graph
endpoints, PDF/web/code ingestion, and automatic `.env` loading are out of scope.

# Phase 6D: Local API Observability Design

**Date:** 2026-07-23

**Status:** Approved design

## Goal and scope

Make the local FastAPI grounded-RAG service diagnosable during development and
portfolio demonstrations. Every handled HTTP request will emit one structured
JSON log line containing a safe correlation identifier, method, route path,
status, elapsed time, and a coarse outcome category. The server will return the
same identifier in `X-Request-ID` so an operator can match a client response to
its server-side log entry.

This phase is local-only. It adds no cloud service, telemetry collector,
database, metrics endpoint, dashboard, authentication, or change to the
grounded-answer contract.

## Alternatives and decision

1. Use only Uvicorn access logs. This has no implementation cost, but its
   format is not a stable project contract and cannot safely express an outcome
   category or correlation identifier.
2. Adopt OpenTelemetry and Prometheus. These support distributed production
   observability, but require additional dependencies, collection infrastructure,
   and a larger operational design.
3. Use standard-library JSON logging in a FastAPI middleware. This supplies the
   diagnostic data needed locally with no dependency or deployment expansion.

Choose option 3. It is explicit, inspectable, and proportionate to a
learning-first local API.

## Public behavior

All handled responses from `GET /health` and `POST /ask`, including FastAPI
validation errors and the existing `502`/`503` failures, include an
`X-Request-ID` response header.

If a caller supplies `X-Request-ID` containing 1--64 ASCII letters, digits,
underscores, or hyphens, the API preserves that value. Missing or invalid values
are replaced with a newly generated UUID4 hexadecimal identifier. Invalid input
is never echoed.

Each request produces one JSON log record through the
`personal_knowledge_agent.api` logger with exactly these common fields:

```json
{
  "event": "api.request.completed",
  "request_id": "e5f...",
  "method": "POST",
  "path": "/ask",
  "status_code": 200,
  "outcome": "success",
  "duration_ms": 12.34
}
```

`outcome` is one of `success`, `client_error`, `model_error`, `not_ready`, or
`server_error`. It is derived from the HTTP status: successful responses are
`success`, 4xx responses are `client_error`, the existing 502 mapping is
`model_error`, the existing 503 mapping is `not_ready`, and an unhandled
exception is `server_error`. Unhandled exceptions are re-raised so FastAPI keeps
its established error semantics; their log record additionally contains only
the exception class name.

## Privacy and security boundary

Logs never contain the question text, answer, citations, retrieved passages,
provider payloads, HTTP query strings, request/response bodies, arbitrary
headers, environment values, API keys, or exception messages. `path` is the URL
path only. The generated identifier is diagnostic metadata, not an
authentication token.

## Architecture and data flow

```text
HTTP request
    |
    v
FastAPI request-observability middleware
    |-- validate or generate request ID
    |-- start monotonic timer
    v
existing /health or /ask handler
    |
    v
HTTP response + X-Request-ID
    |
    `--> one JSON completion record on stderr/logger
```

The middleware is an outer HTTP-adapter concern in `api.py`. Domain models,
application services, the composition root, and storage/model adapters remain
unchanged. Small pure helper functions select request IDs and outcome
categories, keeping the middleware readable and directly testable without a
logging framework abstraction.

## Files and tests

| Path | Responsibility |
| --- | --- |
| `src/personal_knowledge_agent/api.py` | Request middleware, safe JSON logging, request-ID helpers, and existing endpoints. |
| `tests/test_api.py` | Header propagation/generation, exact safe log fields, success and error outcomes, and no question leakage. |
| `README.md` | Local logging workflow and correlation example. |
| `docs/learning-log/phase-6d.md` | Observability, correlation, and privacy lessons. |

Tests use FastAPI's test client and captured standard-library log records. They
must make no provider call, model download, network call, or filesystem write
outside pytest temporary directories.

## Acceptance criteria

1. Every handled API response includes a valid `X-Request-ID`; valid supplied
   identifiers are preserved and invalid identifiers are replaced.
2. Each request emits one parseable JSON completion record with the specified
   safe fields and correct status/outcome/duration.
3. A question deliberately containing a secret-looking string never appears in
   captured records.
4. Existing `POST /ask`, readiness, error, and OpenAPI contracts remain
   unchanged apart from the additive response header.
5. README explains viewing local logs and correlating a response with exactly
   one log record. The full quality gate passes.

## Out of scope

Cloud telemetry, OpenTelemetry, Prometheus, metrics or diagnostics endpoints,
log persistence or rotation, distributed tracing, request-body logging,
authentication, rate limiting, streaming, Agent/memory/graph API endpoints,
and changes to the RAG workflow are out of scope.

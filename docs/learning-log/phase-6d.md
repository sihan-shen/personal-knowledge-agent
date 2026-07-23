# Phase 6D: Local API Observability

## What changed

The local API now documents how to attach a request ID to a local `/ask`
request, see the same ID in the response, and correlate it with its structured
completion record in the Uvicorn terminal. The workflow is deliberately
metadata-only and does not enable metrics or cloud telemetry.

## Engineering lessons

- Correlation IDs connect an HTTP response to a log event without exposing data.
- A monotonic clock measures elapsed work safely across wall-clock adjustments.
- Structured logs are a contract; fixed fields support filtering and tests.
- Privacy is designed by allowlisting metadata rather than redacting full payloads.

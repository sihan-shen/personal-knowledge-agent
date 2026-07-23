# Phase 6C: local FastAPI RAG API

## What changed

The project now exposes its existing grounded-answer service through a local
FastAPI adapter. `GET /health` reports whether both the configured model and an
indexed SQLite knowledge base are ready. `POST /ask` uses the same grounded RAG
composition as `pka ask`, returning answer status, evidence, and citations as
JSON.

## Engineering lessons

- A composition root constructs concrete adapters; the API should depend on that
  boundary rather than duplicate retrieval or model setup.
- Health is a readiness contract, not a file-existence check: an empty or
  unreadable SQLite database cannot answer questions. This endpoint reports
  whether the service is currently able to answer, rather than merely whether
  its process is alive; this local API intentionally has no separate liveness
  endpoint.
- Tests inject deterministic answer services and fake embedders, keeping HTTP
  contracts reproducible without an API key, network call, or model download.
- A REST adapter is deliberately narrow here. Agent routing, memory, and graph
  answers remain separate capabilities until their API contracts are designed.

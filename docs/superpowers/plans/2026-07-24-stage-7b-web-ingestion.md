# Stage 7B: Web Ingestion Implementation Plan

**Goal:** Safely ingest one public static HTML URL through the existing CLI.

**Architecture:** A `WebLoader` owns URL validation, DNS/IP checks, bounded
fetching, and deterministic HTML extraction. `LocalDocumentLoader` dispatches
URLs to it; all downstream chunking and retrieval stay unchanged.

## Task 1: Safe static HTML loader

- Create `infrastructure/web_loader.py` and `tests/unit/test_web_loader.py`.
- Write failing deterministic tests with fake resolver/opener for URL validation,
  private-IP rejection, status/content-type/size failures, and title/body
  extraction.
- Implement injected resolver/opener seams, no-redirect bounded fetch, and an
  `HTMLParser` extractor that ignores unsafe/non-content nodes.
- Run targeted tests, Ruff, and mypy; commit `feat: add safe web loader`.

## Task 2: CLI dispatch and regression coverage

- Update `infrastructure/local_document_loader.py`, `cli.py`, and tests.
- Write failing CLI tests that ingest a mocked public URL and retain its source
  in search evidence; retain Markdown/PDF regressions.
- Change the CLI input boundary from `Path` to a source string only where needed;
  retain local path behavior unchanged.
- Run focused tests and commit `feat: ingest static web pages`.

## Task 3: Documentation and complete quality gate

- Update `README.md` and add `docs/learning-log/stage-7b.md`.
- Document the explicit single-URL command, static-HTML limit, and security
  refusals without suggesting private-network access.
- Run `pytest -W error`, full pytest, Ruff format/check, mypy, health, and lock
  checks; commit `docs: explain web ingestion`.

# Phase 1C: PDF Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Ingest text-based local PDFs through `pka ingest` while preserving
page-level evidence provenance.

**Architecture:** `PdfLoader` converts nonempty pages into `## Page N` sections
inside the existing `Document`; `LocalDocumentLoader` selects Markdown or PDF;
the existing `MarkdownChunker`, index, retrieval, and answer flow remain intact.

**Tech Stack:** Python 3.12+, pypdf, existing domain errors/models, pytest,
Ruff, mypy, uv.

## Global Constraints

- Support only local `.pdf`, `.md`, and `.markdown` files.
- Do not implement OCR, passwords, network loading, layout extraction, tables,
  images, uploads, or schema changes.
- Skip blank PDF pages; a fully textless PDF raises `EmptyDocumentError` and
  names OCR as the required next step.
- Encrypted, malformed, and unreadable files fail through existing safe errors.
- Preserve Markdown and all existing retrieval/API behavior.
- Use deterministic tests; no live model, LLM, network call, or user PDF.

---

### Task 1: Page-aware PDF loader

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/pdf_loader.py`
- Create: `tests/unit/test_pdf_loader.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] Add a bounded `pypdf` runtime dependency and regenerate `uv.lock` so the
  test-created PDFs use the same parser dependency as production.
- [ ] Write failing tests for a two-page PDF, skipped blank page, unsupported
  suffix, missing path, encrypted reader, parser error, and fully textless PDF.
- [ ] Run `uv run pytest tests/unit/test_pdf_loader.py -q` and observe failure
  because `PdfLoader` does not exist.
- [ ] Implement `PdfLoader.load(path: Path) -> Document`: resolve local path,
  enforce `.pdf`, check regular file, create `PdfReader`, reject encryption,
  extract/strip each page, and join nonempty pages as `## Page {number}`
  sections. Map parser/OS errors to `DocumentReadError` and no text to
  `EmptyDocumentError` mentioning OCR.
- [ ] Re-run targeted tests plus Ruff and mypy.
- [ ] Commit `feat: add page-aware PDF loader`.

### Task 2: Local loader dispatch and CLI ingestion

**Files:**
- Create: `src/personal_knowledge_agent/infrastructure/local_document_loader.py`
- Modify: `src/personal_knowledge_agent/cli.py`
- Modify: `tests/test_cli.py`

- [ ] Write failing CLI tests that ingest a generated two-page PDF, re-ingest
  it, and assert source output plus search/evidence provenance includes
  `Page 2`; retain a Markdown ingestion regression test.
- [ ] Run those tests and observe failure before dispatch/dependency wiring.
- [ ] Add `LocalDocumentLoader` selecting `MarkdownLoader` for `.md` and
  `.markdown`, `PdfLoader` for `.pdf`, and raising `UnsupportedDocumentError`
  otherwise. Replace CLI's direct `MarkdownLoader()` construction with it.
- [ ] Run focused tests, `uv lock --check`, Ruff, and mypy.
- [ ] Commit `feat: ingest PDF documents from the CLI`.

### Task 3: Documentation and complete verification

**Files:**
- Modify: `README.md`
- Create: `docs/learning-log/phase-1c.md`
- Modify: `tests/test_cli.py`

- [ ] Add a failing README-content test for `pka ingest notes/rag.pdf`, page
  provenance, text-only limitation, and OCR guidance.
- [ ] Add a README PDF section immediately after Markdown ingestion guidance;
  include the exact command and exclusions for scanned/encrypted PDFs.
- [ ] Add a concise learning log explaining extraction versus OCR, provenance
  via headings, and why adapters keep domain code format-neutral.
- [ ] Run `pytest -W error`, full `pytest`, Ruff format/check, mypy, `pka
  health`, and `uv lock --check`.
- [ ] Commit `docs: explain PDF ingestion`.

## Acceptance checklist

- [ ] PDF chunks preserve ordered `Page N` headings.
- [ ] Safe failures cover encrypted, malformed, unreadable, and textless PDFs.
- [ ] Markdown ingestion remains unchanged.
- [ ] CLI evidence exposes PDF page provenance.
- [ ] Full quality gate passes without external model or network calls.

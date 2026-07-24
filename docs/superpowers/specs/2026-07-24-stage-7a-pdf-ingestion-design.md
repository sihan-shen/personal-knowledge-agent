# Stage 7A: PDF Ingestion Design

**Date:** 2026-07-24

**Status:** Approved design

## Goal and scope

Extend the existing local ingestion command so it accepts text-based local PDF
files in addition to Markdown. Each extracted page must remain traceable in the
existing evidence display as `source-path > Page N`.

This phase supports only local, unencrypted PDFs that contain extractable text.
It intentionally excludes OCR, scanned-image recognition, password entry,
network sources, tables, images, annotations, and layout reconstruction.

## Decision

Use `pypdf` for local page text extraction. It offers a small pure-Python
`PdfReader` API, page-level extraction, encrypted-document detection, and typed
read errors. The loader will keep the domain model unchanged by expressing page
boundaries as Markdown headings in a normal `Document`; the existing
`MarkdownChunker` then supplies `Page N` as the chunk heading path.

## Data flow

```text
local .md/.markdown/.pdf path
        |
        v
LocalDocumentLoader
  |-- MarkdownLoader
  `-- PdfLoader -> PdfReader -> extract_text per page
        |
        v
Document content: ## Page 1 ... ## Page 2 ...
        |
        v
existing MarkdownChunker -> existing SQLite index -> existing retrieval
```

`pka ingest` remains the sole public command. It selects a local loader based on
the resolved filename suffix; no user-facing format flag is added.

## Behavior and failures

- `.md` and `.markdown` retain their current UTF-8 behavior exactly.
- `.pdf` resolves to a local regular file before parsing.
- A PDF with `PdfReader.is_encrypted` true fails without attempting to decrypt.
- Parser and operating-system read failures become the project's existing safe
  `DocumentReadError`; no underlying parser details are exposed by the CLI.
- Whitespace-only pages are skipped. If no page yields text, raise
  `EmptyDocumentError` with an actionable message that OCR is required.
- Extracted nonempty page text is concatenated as deterministic `## Page N`
  sections, preserving original page order and 1-based page numbers.
- An unknown suffix still raises `UnsupportedDocumentError`.

## File structure and tests

| Path | Responsibility |
| --- | --- |
| `src/personal_knowledge_agent/infrastructure/pdf_loader.py` | Validate and extract a safe page-structured document from one local PDF. |
| `src/personal_knowledge_agent/infrastructure/local_document_loader.py` | Route local suffixes to MarkdownLoader or PdfLoader. |
| `src/personal_knowledge_agent/cli.py` | Construct `IngestService` with LocalDocumentLoader. |
| `tests/unit/test_pdf_loader.py` | Deterministic parser tests with test-created PDFs and fake reader seams where required. |
| `tests/test_cli.py` | PDF ingestion, page provenance, and unchanged Markdown behavior. |
| `pyproject.toml`, `uv.lock` | Add locked `pypdf` runtime dependency. |
| `README.md`, `docs/learning-log/stage-7a.md` | Explain supported PDF limits and local usage. |

Tests must never download a model, call an LLM, depend on a user PDF, or perform
OCR. They must cover two nonempty pages, skipped blank pages, unsupported suffix,
missing path, encrypted input, malformed parser input, zero-text PDF, CLI
replacement, and evidence display with `Page N`.

## Acceptance criteria

1. `uv run pka ingest notes/rag.pdf --db .pka/knowledge.db` indexes extracted
   text through the existing persistence pipeline.
2. Search and grounded-answer evidence identify the PDF source and page heading.
3. Unsupported, unreadable, encrypted, malformed, and textless inputs fail with
   safe, actionable existing project errors.
4. Markdown behavior, retrieval, answers, memory, graph, and API contracts are
   unchanged.
5. The full locked quality gate passes.

## Out of scope

OCR, image-only PDF support, passwords, remote URLs, PDF metadata extraction,
page-layout reconstruction, tables, images, HTML, code repositories, and API
upload endpoints remain future phases.

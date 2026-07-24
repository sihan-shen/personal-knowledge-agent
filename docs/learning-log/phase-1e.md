# Phase 1E: Code Document Ingestion

## What changed

Explicit local source files now enter the same document pipeline as Markdown,
PDF, and web pages. Python declarations are represented as headings so evidence
can identify a class or function without a storage-schema change.

## Engineering lessons

- Treat source code as untrusted text: ingestion must never execute it.
- Bounded UTF-8 input and binary rejection provide a clear local resource limit.
- Heading-based provenance is a useful incremental step before AST-level models.

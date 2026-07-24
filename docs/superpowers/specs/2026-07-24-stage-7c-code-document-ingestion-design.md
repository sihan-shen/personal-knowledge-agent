# Stage 7C: Code Document Ingestion Design

**Status:** Approved by delegated authority

## Goal

Extend explicit `pka ingest <path>` to accept local source and development
document files while preserving existing Markdown, PDF, and web behavior.

## Scope

Support one explicitly named local file with suffixes `.py`, `.json`, `.yaml`,
`.yml`, `.toml`, `.txt`, and `.rst`. Decode UTF-8 only, reject files over a
fixed 1 MiB limit, reject binary/NUL-containing content, and retain the resolved
path as source. No recursive directory ingestion, hidden-file traversal,
repository ignore rules, code execution, AST parsing, syntax validation, or
language server integration.

## Design

`CodeDocumentLoader` validates the local path and text boundary. It converts
Python top-level `class` and `def` declarations into `##` headings; JSON/YAML/
TOML/text/RST retain a file-level `# <filename>` heading. The existing
`MarkdownChunker` then creates deterministic chunks and evidence paths without
schema changes. `LocalDocumentLoader` selects CodeDocumentLoader by suffix.

## Errors

Unknown suffixes use `UnsupportedDocumentError`; non-file, unreadable, invalid
UTF-8, oversized, binary, and empty inputs use existing safe document errors.
No source bytes or code text are included in error messages.

## Tests and acceptance

Tests cover Python declaration headings, structured/text files, unknown/binary/
oversized/invalid input, unchanged Markdown/PDF/web dispatch, CLI ingestion and
evidence provenance. Tests use generated temporary files and no executed code.

## Out of scope

Directory/repository ingestion, Git ignore parsing, JavaScript/Java/C++ syntax
trees, AST symbol graphs, code execution, secrets scanning, and remote source
repositories remain future work.

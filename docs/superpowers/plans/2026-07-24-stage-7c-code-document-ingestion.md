# Stage 7C: Code Document Ingestion Plan

1. Write failing loader tests, then implement `CodeDocumentLoader` with explicit
   suffix, size, binary, encoding, empty-content, and Python-heading behavior.
2. Add loader dispatch and deterministic CLI provenance tests while preserving
   Markdown/PDF/web behavior.
3. Document supported source files and limits, run the full quality gate, commit,
   merge to `main`, and remove the isolated worktree.

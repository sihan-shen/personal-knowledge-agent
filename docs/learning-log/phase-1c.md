# Phase 1C: PDF Ingestion

## What changed

The local ingestion boundary now accepts text-based PDFs alongside Markdown.
Each nonempty PDF page becomes a `Page N` section before the existing chunker
and index process it, so evidence remains page-traceable without a schema change.

## Engineering lessons

- PDF text extraction is not OCR: image-only scans must be converted to text
  before a text retriever can use them.
- Encoding page provenance as existing headings reuses stable chunk and citation
  behavior instead of adding a format-specific persistence path.
- Loader adapters keep PDF-library concerns outside the domain and application
  layers, preserving replaceability for future formats.

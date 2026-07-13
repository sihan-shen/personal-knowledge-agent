# Phase 1A Learning Log

## What this phase built

The CLI can load a UTF-8 Markdown document, preserve its heading structure while
creating chunks, persist a lexical index in SQLite, and retrieve ranked Chinese or
English evidence in a later process.

## Document and Chunk

A `Document` represents one source before indexing. A `Chunk` is an ordered,
searchable portion of that document with a stable ID and heading path. Keeping these
domain records independent of Markdown and SQLite lets later PDF loaders and vector
indexes reuse the same application workflow.

## Heading-aware chunking

The chunker treats paragraphs and fenced code blocks as indivisible units. It groups
adjacent blocks under the same heading toward an 800-character soft target. A large
paragraph can exceed the target because preserving semantic structure matters more
than enforcing an exact character count.

Phase 1A intentionally uses no overlap. This makes the lexical baseline easier to
inspect and prevents duplicate adjacent results from hiding ranking behavior.

## Mixed-language tokenization

English text becomes normalized lowercase words. Chinese sequences become
overlapping two-character tokens, so `检索增强` produces `检索`, `索增`, and `增强`.
This requires no dictionary and handles mixed technical vocabulary deterministically,
but it creates more index terms than word segmentation.

## Inverted index and BM25

The `postings` table maps each token to chunks and term frequencies. Search reads only
postings for query tokens rather than scanning every chunk. BM25 then balances local
term frequency, corpus rarity, and chunk length. This is a lexical match: related
concepts with different vocabulary may still be missed, which motivates embeddings
in the next retrieval iteration.

## SQLite transactions

Reingestion deletes old document data and inserts the new document, chunks, and
postings in one transaction. If any insertion fails, rollback restores the previous
version. `PRAGMA user_version` also prevents code from silently opening an unknown
schema version.

## RAG, LLM, Agent, and Prompt boundaries

This phase implements only the retrieval part of RAG. There is no LLM call or prompt,
so the CLI returns evidence rather than a generated answer. There is also no Agent:
the control flow always searches when the user invokes `pka search`. Later phases can
compare keyword retrieval with vector retrieval, add a prompt that cites retrieved
chunks, and finally let an Agent decide whether retrieval is necessary.

## Commands I can explain

```bash
uv run pka ingest tests/fixtures/knowledge/rag_notes.md --db /tmp/pka.db
uv run pka search "RAG 如何检索 knowledge" --db /tmp/pka.db
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

# Stage 7B: Web Ingestion Design

**Date:** 2026-07-24

**Status:** Approved design

## Goal and scope

Allow `pka ingest <https-url>` to ingest one public, static HTML page into the
existing local knowledge base. The result stores the canonical final URL as the
source, extracts readable page text, and reuses the existing heading-aware
chunker, index, retrieval, and citation display.

This is deliberately a one-page local fetcher: no crawl frontier, links,
JavaScript, authentication, cookies, forms, uploads, browser automation, or
background tasks.

## Security boundary

The fetcher accepts only absolute `http` or `https` URLs without credentials.
Before each connection it resolves the target host and rejects loopback,
private, link-local, multicast, unspecified, and reserved IP addresses. It
disables redirects in this phase, sets a fixed connect/read timeout, sends a
stable project User-Agent, limits response bytes, and accepts only
`text/html` responses. Any DNS, socket, TLS, timeout, HTTP-status, content-type,
or size failure becomes a safe `DocumentReadError` with no response body.

## Extraction and provenance

Use Python's standard-library `html.parser` to avoid a dependency expansion.
Ignore `script`, `style`, `noscript`, `svg`, and `template` elements. Prefer the
first `main` or `article` subtree when present; otherwise use visible `body`
text. The first nonempty `title` becomes a level-one Markdown heading, then
block-level elements (`h1`--`h6`, `p`, `li`, `pre`, `blockquote`) become plain
Markdown-like text. The existing `MarkdownChunker` therefore keeps semantic
headings without a new storage schema.

The stored `Document.source` is the normalized input URL. Citations display that
URL and its extracted heading path; the system does not claim a permanent copy
or freshness guarantee.

## Architecture

```text
https://public.example/article
        |
        v
LocalDocumentLoader
        |
        v
WebLoader
  URL validation -> DNS/IP safety -> bounded HTTP fetch -> HTML text extractor
        |
        v
Document -> existing MarkdownChunker -> SQLite / E5 / BM25
```

`LocalDocumentLoader.load` recognizes URL strings encoded by the CLI before
path-based Markdown/PDF dispatch. The CLI accepts the existing positional
`path` argument as a string and explicitly converts only local values to
`Path`, keeping URL parsing at the outer adapter boundary.

## Failures and tests

Tests use injected resolver and opener seams; no test resolves a real hostname
or opens the network. They cover unsupported schemes, credentials, private IPs,
DNS failure, non-HTML response, oversized response, error HTTP status, static
extraction, title/headings, empty visible text, and unchanged Markdown/PDF
ingestion. CLI tests verify a URL source prints unchanged after indexing.

## Acceptance criteria

1. A public static HTML URL can be explicitly ingested through `pka ingest`.
2. Unsafe targets and invalid or non-HTML responses fail closed before content
   reaches the knowledge base.
3. Extracted content has stable headings and citations retain the URL.
4. Existing local Markdown/PDF ingestion and all retrieval/API behavior remain
   unchanged.

## Out of scope

Redirects, crawling, JavaScript rendering, credentials, cookies, robots.txt,
rate limiting, PDF URLs, markdown URLs, site maps, persistence of raw HTML,
content freshness, browser automation, and web search are future work.

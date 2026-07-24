# Stage 7B: Web Ingestion

## What changed

The existing ingestion command now accepts one public static HTML URL. A
dedicated outer-layer loader validates the target, bounds the fetch, extracts
visible content, and reuses the existing document, chunking, and citation flow.

## Engineering lessons

- Network ingestion is an SSRF boundary: validate schemes and reject resolved
  private, loopback, and reserved addresses before fetching.
- Size and content-type limits turn unbounded remote input into a defined local
  resource contract.
- One-page extraction is intentionally different from crawling; explicit scope
  keeps provenance and failure behavior understandable.

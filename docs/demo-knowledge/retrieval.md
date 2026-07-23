# Retrieval and hybrid ranking

## Chunking creates retrievable evidence

Retrieval-augmented generation (RAG) stores a document as small, meaningful
chunks instead of one large prompt. A question can then retrieve the few chunks
that contain evidence for an answer. Heading-aware chunks retain their document
context while remaining small enough to rank precisely.

## Hybrid retrieval combines complementary signals

Lexical retrieval rewards exact words, names, and technical terms. Semantic
retrieval finds passages with similar meaning even when the wording differs.
Hybrid retrieval combines both rankings. Reciprocal rank fusion (RRF) is a
simple way to combine them: a passage receives credit from each list according
to its rank, so no single retrieval signal must be trusted alone.

# Grounded answers and citations

## Evidence constrains the answer

A grounded answer is limited to the evidence retrieved from the knowledge base.
The language model may explain and connect that evidence, but it should not add
unsupported facts. When the evidence is insufficient, a reliable RAG system
states that limitation rather than guessing.

## Citations are request-local identifiers

Each answer returns citations that identify the evidence chunks used for that
specific request. The identifier lets a reader inspect the source document and
heading path. Request-local identifiers are safer than permanent labels because
they remain tied to the exact retrieval result that supported the answer.

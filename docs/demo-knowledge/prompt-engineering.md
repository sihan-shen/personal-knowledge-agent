# Prompt engineering for reliable RAG

## Instructions define the output contract

Prompt engineering defines what a language model should produce and which
constraints it must follow. For grounded RAG, the prompt should require an
answer based only on supplied evidence, a clear refusal when evidence is absent,
and citations that connect claims to sources.

## Validation completes the reliability boundary

A prompt is not a complete reliability boundary because model output can still
be malformed or unsupported. Structured output validation checks the response
shape and citation identifiers before the application returns it. Failing closed
is preferable to returning an answer whose evidence cannot be verified.

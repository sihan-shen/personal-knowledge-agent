# Phase 2B Learning Log

## What this phase built

The new `pka agent` command makes one visible policy decision before responding. It
can render a deterministic local response for a greeting, capabilities question, or
thanks, or delegate to the existing evidence-grounded RAG pipeline. `pka ask` remains
unchanged so the fixed RAG and routed Agent workflows can be compared directly.

```text
Question
  -> exact safe-intent rule
     -> matched: local direct template
     -> unmatched: structured LLM route
        -> direct: local direct template
        -> retrieve: search -> grounded answer
        -> known router failure: visible fallback -> search -> grounded answer
```

## Agent design: policy is not generation

An Agent is not defined merely by calling an LLM. The key addition is control: the
system chooses an action from an explicit action space. Phase 2B keeps that space to
`direct` and `retrieve`, which makes every branch observable and testable.

Routing and answering have different contracts. The router returns only an action
and optional direct intent. It is not allowed to answer factual questions. The
grounded generator receives retrieved evidence and returns an answer with validated
citations. Separating these responsibilities prevents a routing prompt from quietly
becoming a second, uncited answer path.

## LLM routing and structured output

The router reuses the provider-neutral `LanguageModel` protocol and OpenAI-compatible
Chat Completions adapter. Its prompt asks for exactly:

```json
{"action":"retrieve","direct_intent":null}
```

or a direct decision with one of `greeting`, `capabilities`, and `thanks`. JSON mode
does not prove that fields, enum values, or combinations are correct. The local
parser rejects malformed JSON, code fences, arrays, missing or extra fields, wrong
types, duplicate field names, unknown values, direct without an intent, and retrieve
with an intent. It does not repair a response or echo provider content in an error.

## Why the hybrid router uses rules first

Exact greetings and thanks do not benefit from probabilistic classification. A
whole-message rule is faster, free of provider latency, deterministic, and easy to
explain. Rules are intentionally narrow: substring matching would classify
`hello, explain RAG` as a greeting and skip evidence.

Natural language has many phrasings, so rules alone would grow into a brittle hidden
classifier. The LLM handles unmatched phrasing, but it must choose from the same
small allowlist. This hybrid keeps obvious cases local without pretending that a few
regular expressions understand every request.

## Conservative routing and fail-closed behavior

The project is a personal knowledge agent, so the costly mistake is bypassing the
knowledge base and returning an unsupported factual answer. Uncertain or ambiguous
decisions therefore retrieve. This is a policy trade-off: it creates extra latency
and cost, but preserves the evidence boundary until evaluation data can justify a
broader direct path.

Known router failures also retrieve. `LanguageModelError` becomes a visible
`provider error` fallback, while an invalid route contract becomes `invalid router
output`. These categories expose that the intended classifier did not succeed
without leaking a provider body, API key, or stack trace. Unexpected exceptions are
not caught because converting programming bugs into ordinary fallbacks would hide
defects.

## RAG and vector database behavior after routing

Routing does not change retrieval. The retrieve branch still uses the Phase 1B
SQLite index, multilingual E5 embeddings, BM25 lexical ranking, exact cosine
semantic ranking, and RRF hybrid fusion. The existing `AnswerService` still assigns
request-local evidence labels, builds the grounded prompt, validates the answer
schema, and checks citation semantics.

The vector index is a tool behind the Agent decision, not the Agent itself. It solves
candidate selection in a semantic space; it does not decide whether retrieval is
appropriate or whether a generated claim is faithful.

## Prompt engineering and privacy boundary

The route prompt states the allowlist and gives a mixed-request example. More
importantly, the application restricts what enters that prompt: only the user
question is sent. Database paths, retrieved chunks, scores, chunk IDs, embeddings,
and API keys are absent.

If the route is retrieve, the later grounded-answer call receives the question and
minimum evidence fields as documented in Phase 2A. This staged data flow avoids
sending private retrieved content merely to decide whether retrieval is needed.

## Cost and latency model

- Exact rule direct: zero remote model calls; the local template is immediate.
- LLM direct: one routing call; the answer remains a local template.
- LLM retrieve with evidence: normally one routing call plus one answer call.
- LLM retrieve with no matches: one routing call followed by a local evidence
  refusal.
- Router provider failure: the adapter may perform its configured retries before
  the workflow falls back to retrieval and attempts grounded generation.

`pka ask` remains the cheaper explicit choice when the caller already knows that
retrieval is required, because it skips the route call.

## Alternatives and trade-offs

- Pure rules would be cheaper and fully offline but brittle for natural phrasing.
- Pure LLM routing would classify more phrasing but add latency and provider
  dependency even for `hi`.
- Broad direct answering would be convenient for general knowledge, but would create
  an uncited answer path before faithfulness and routing evaluation exist.
- A multi-tool ReAct loop would be more agentic, but would multiply action, stopping,
  permission, and error states before the two-action policy is measured.

The selected design is intentionally small enough to explain in an internship
interview and strict enough to support later evaluation.

## Deterministic validation

Tests cover domain invariants, whole-message rule matching, prompt data minimization,
strict route parsing, rule short-circuiting, LLM decisions, fail-closed categories,
AgentService coordination, CLI output, and the two-call routed RAG workflow. Model
fakes return complete project `ModelOutput` records and never contact DeepSeek.

## Known limitations and next phase

- There is no conversation history, multi-step loop, or tool call beyond retrieval.
- Direct intents use fixed English and Chinese templates.
- Routing quality has not yet been measured against a labelled dataset.
- Retrieval relevance and answer faithfulness still have no quantitative evaluator.
- PDF, web, and code ingestion remain outside the current phase.

The next useful engineering step is evaluation: create labelled routing cases and
measure direct/retrieve precision, fallback rate, latency, cost, retrieval relevance,
and grounded-answer quality before expanding the action space.

## Commands I can explain

```bash
export PKA_LLM_API_KEY="your-key"
uv run pka agent "你好"
uv run pka agent "RAG 为什么需要检索？" --mode hybrid --limit 5
uv run pka ask "RAG 为什么需要检索？" --mode hybrid --limit 5
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src
```

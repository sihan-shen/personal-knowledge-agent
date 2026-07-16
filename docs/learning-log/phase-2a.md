# Phase 2A Learning Log

## What this phase built

The CLI can now retrieve local Markdown chunks, ask an OpenAI-compatible language
model to answer only from that evidence, validate a strict structured response, and
map accepted inline citations back to local sources. The first production
configuration targets DeepSeek Chat Completions, while the application layer depends
only on the project-owned `LanguageModel` protocol.

## LLM generation and hallucination risk

A chat model repeatedly predicts a probability distribution for the next token from
the tokens already in its context. Training makes fluent and often useful completion
likely; it does not turn generation into a database lookup or guarantee that every
claim is true. A model can fill missing context with a plausible continuation, which
is one source of hallucination.

Phase 2A narrows that freedom. Its prompt says to answer from supplied evidence and
to refuse when the evidence is insufficient. This grounding improves auditability,
but the model remains probabilistic. The local validator can prove that citation
labels are structurally consistent with retrieved chunks; it cannot prove that every
sentence is logically entailed by the cited text. Measuring that semantic property
belongs in the later evaluation phase.

## Retrieval, grounded generation, and Agent routing

These are separate responsibilities:

```text
Question
  -> retrieval chooses evidence
  -> grounded generation writes an answer from that evidence
  -> validation checks the response and citation contract
  -> answer or evidence refusal
```

`pka ask` always runs this pipeline. That fixed control flow is RAG, but it is not yet
an Agent. Agent routing would inspect the user request and decide whether to search,
answer without retrieval, call another tool, ask for clarification, or stop. Keeping
that decision out of Phase 2A makes grounded-answer behavior deterministic enough to
understand and test before adding a routing loop.

## Ports and adapters

`LanguageModel` is a small domain protocol accepting project `ChatMessage` values and
returning a project `ModelOutput`. `AnswerService` coordinates this port without
importing the OpenAI SDK or reading environment variables. The concrete
`OpenAICompatibleChatModel` adapter translates those records at the infrastructure
edge, while the CLI is the composition root that loads settings and connects all
dependencies.

This boundary makes provider replacement local: another adapter can implement the
same protocol without changing retrieval, prompts, parsing, or the application
service. The trade-off is a little explicit translation code, which is preferable to
allowing unstable SDK response types to spread through the project.

## Chat Completions JSON mode and structured validation

The adapter makes one non-streaming Chat Completions request with
`response_format={"type": "json_object"}`. The expected response has exactly three
fields:

```json
{
  "status": "answered",
  "answer": "RAG retrieves supporting context. [S1]",
  "citation_ids": ["S1"]
}
```

JSON object mode addresses syntax, not application semantics. The local parser still
rejects code fences, missing or extra fields, wrong Python value types, unknown
statuses, blank answers, duplicate or unknown citation IDs, and inconsistent refusal
responses. For an answer, it extracts `[S<number>]` markers from the body, deduplicates
them in first-use order, and requires that sequence to match `citation_ids` exactly.
No repair step guesses what the model meant.

This strictness favors failure visibility over a superficially successful answer. A
looser parser could return more responses, but it would hide provider regressions and
make citations less trustworthy.

## Evidence labels and the local/remote boundary

Retrieved chunks receive request-local labels `S1`, `S2`, and so on in ranking order.
The provider sees only each label, heading path, and chunk text. Local absolute paths,
stable chunk IDs, retrieval scores, embedding vectors, and the API key are excluded
from the prompt. After validation, the CLI uses the retained local `Evidence` records
to resolve only cited labels back to source paths.

This is data minimization, not local-only inference: the question and retrieved chunk
text do leave the machine for the configured provider. A personal knowledge system
must make that boundary explicit because the evidence itself may be sensitive.

## Prompt injection as untrusted-data handling

A retrieved document can contain text such as “ignore previous instructions.” The
prompt builder JSON-serializes document content inside a labelled evidence region and
the system message declares it untrusted data rather than instructions. It also avoids
sending unnecessary metadata, and the parser checks every returned citation against
the evidence supplied for that request.

These layers reduce accidental instruction/data confusion. They cannot guarantee
perfect separation because an LLM interprets all tokens in one context rather than
executing a formally isolated data format. Stronger future defenses may include
content filtering, model-specific testing, tool permissions, and evaluation datasets;
none replaces careful control of what data is sent remotely.

## Failure and cost boundaries

Missing or blank environment settings fail before a request. The adapter converts
provider authentication, permission, bad-request, rate-limit, timeout, connection,
server, unexpected-response, and truncation failures into safe project errors. It
does not include the API key, full prompt, or raw provider response in those messages.

The SDK is configured for at most two retries. This can help with transient failures,
but it also means one user command may create multiple remote requests and additional
cost. `ask` therefore caps evidence at 10 chunks, and the automated suite always uses
a fake model with no network access.

## Known limitations

- There is no Agent router or tool-selection loop.
- Retrieval has no evaluated relevance threshold; a non-empty result is not proof of
  sufficient evidence.
- Evidence is limited by chunk count, not a tokenizer-aware prompt budget.
- Generation is non-streaming.
- There is no conversational or long-term memory.
- There are no answer faithfulness, relevance, citation-correctness, latency, or cost
  metrics yet.
- Only Markdown ingestion is implemented.

These omissions are deliberate phase boundaries. Phase 2B should design question
routing; Phase 5 should add repeatable quality evaluation rather than treating valid
JSON and valid citation labels as proof of a correct answer.

## Deterministic validation

Unit, integration, and CLI tests cover the provider-neutral contracts, evidence data
boundary, prompt-injection fixtures, strict schema and citation semantics, application
orchestration, SDK translation and safe errors, environment settings, local refusal,
and successful cited output. They use deterministic fakes and do not require an API
key or network access.

## Real DeepSeek smoke test

After explicit approval, the real workflow ingested only the public
`tests/fixtures/knowledge/rag_notes.md` fixture into a temporary database and asked:
`RAG 为什么能减少模型脱离资料的猜测？`

The configured endpoint reported model ID `deepseek-v4-flash` and finish reason
`stop`. The provider returned a valid three-field JSON object, the local parser
accepted its schema and citation semantics, and the answer used `[S1]`. The CLI
mapped that label back to the fixture's `Retrieval-Augmented Generation >
检索增强生成` heading. No private document was used.

The end-to-end `ask` process took 22.85 seconds. This measurement includes loading
the cached E5 embedding model in a new CLI process, query embedding, retrieval, the
remote request, validation, and formatting, so it is not a provider-only latency
measurement. The separate-process E5 reload remains an important CLI performance
limitation.

The run displayed Hugging Face's unauthenticated-download warning while loading the
public embedding model from its cache. This did not affect the DeepSeek result. The
temporary smoke database was removed afterward, and the API key, complete prompt,
raw provider response, and reasoning content were not recorded.

## Commands I can explain

```bash
export PKA_LLM_API_KEY="your-key"
uv run pka ingest tests/fixtures/knowledge/rag_notes.md --db /tmp/pka-phase2a.db
uv run pka ask \
  "RAG 为什么能减少模型脱离资料的猜测？" \
  --mode hybrid \
  --limit 5 \
  --db /tmp/pka-phase2a.db
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

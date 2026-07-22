# Phase 5B Learning Log: LLM Judge Evaluation

## What we built

Phase 5B extends `pka evaluate` with an explicit `--judge` mode. It keeps the
deterministic Phase 5A EM, token F1, and citation metrics, then adds three semantic
signals for each answered XQuAD-mini case: relevance, evidence faithfulness, and
completeness. Each Judge metric has a discrete score of 0, 1, or 2 plus a short
reason; report summaries normalize the scores to the 0.0 through 1.0 range.

## Why an LLM Judge helps

Extractive metrics are reproducible, but they may penalize a correct paraphrase or
an answer that uses different wording from the reference. A Judge can apply a written
rubric to semantic meaning, so it makes a useful additional regression signal for
changes to retrieval, prompts, or answer models. It does not make the evaluation
fully objective: model choice, prompt wording, provider changes, and sampling can
all affect a Judge verdict.

## Input isolation is a stronger boundary than a prompt instruction

Faithfulness asks whether an answer is supported by retrieved evidence. Completeness
asks whether it covers a reference answer. A single model call that sees both could
let its faithfulness verdict be influenced by reference-answer knowledge, even when
the prompt says not to use it. Phase 5B therefore makes two calls for every answered
case:

```text
question + answer + retrieved evidence -> faithfulness
question + answer + reference answers   -> relevance + completeness
```

The code represents these as different domain requests and different prompt builders.
Tests assert that a faithfulness prompt contains no reference answers and that a
reference-quality prompt contains no retrieved evidence. This is a practical prompt
engineering lesson: delimit untrusted data, state that it cannot change the rubric,
and enforce important boundaries in program structure instead of trusting natural
language alone.

## Structured output and failure handling

The Judge must return exact JSON shapes. The parser rejects extra or missing metrics,
non-integer and out-of-range scores, blank reasons, and malformed JSON. Provider and
schema failures propagate through the existing safe error path. The report is written
only after all answer generation and Judge calls succeed, so a failed run cannot be
mistaken for a zero-quality result.

## Cost, bias, and portfolio trade-offs

The 24-case suite makes up to 48 additional live Judge calls. `PKA_JUDGE_MODEL` can
separate the Judge model from `PKA_LLM_MODEL`, but it shares the configured provider
credentials and endpoint. Recording the Judge model ID and prompt versions in schema
v2 exposes those conditions in every report. Even a distinct Judge can share training
biases with the answer model, so compare Judge scores with deterministic metrics and
inspect individual reasons rather than optimizing one aggregate number.

## What remains out of scope

This phase still evaluates only document-RAG answers. It does not rate whether the
Agent retrieved at the right time, whether long-term Memory was safely recalled,
whether graph evidence was sufficient, or whether a refusal was appropriate. Those
need separate scenarios and rubrics rather than being folded into one score.

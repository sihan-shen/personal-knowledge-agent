# Phase 5A Learning Log: Bilingual Document-RAG Evaluation

## What we built

Phase 5A adds `pka evaluate`, a reproducible evaluation of the existing
document-RAG answer path. The packaged `xquad-mini v1` suite contains 24 aligned
English and Chinese cases derived from XQuAD. Each case supplies a question,
one-or-more gold extractive answers, a language label, and exactly one gold fixture
document ID. The fixture's attribution file pins the upstream source revision and
checksums, which makes its origin and reproduction conditions inspectable.

## Gold data and reproducibility

Gold data turns a subjective question—"did this answer work?"—into a repeatable
comparison against declared answer strings and a declared evidence document. The
suite is intentionally small and bilingual, not a claim of broad benchmark coverage.
Its packaged data, fixed case ordering, deterministic corpus ingestion, local metric
functions, and stable JSON serialization make a rerun comparable when the same
answer outputs are supplied.

The command rebuilds `.pka/evaluations/xquad-mini-v1/knowledge.db` from only the
public fixture and writes its report separately, by default to
`.pka/reports/xquad-mini-v1.json`. It has no `--db` option. This separation prevents
private knowledge-base documents from entering an evaluation case and avoids changing
the user's `.pka/knowledge.db` while measuring the suite.

## EM and extractive token F1

Answer exact match (EM) is one when a normalized model answer equals any normalized
gold answer, otherwise zero. Token F1 compares overlapping normalized tokens, then
uses the best score across the allowed gold answers. The tokenizer handles the suite's
English and Chinese text deterministically, so macro averages are stable local
scoring results.

These are extractive-answer metrics. They can under-score a correct paraphrase,
partial but useful answer, or a response that adds valid context; conversely they do
not test usefulness, safety, explanation quality, or factual completeness. Treat EM
and F1 as focused regression signals, not a complete measure of answer quality.

## Citation precision and recall

Each case identifies one gold fixture document. Citation precision is the fraction
of cited document IDs that equal that gold ID; citation recall is one when the gold ID
appears among the citations and zero otherwise. These metrics expose a common RAG
failure mode: an answer can overlap the reference while grounding itself in the wrong
or extra source. They still do not prove that every individual sentence is supported
by the cited document.

## Deterministic scoring, live generation

The evaluation's fixture loading, ingestion, answer validation, scoring, and report
writing are local and deterministic. The answer generation itself remains a live call
to the configured OpenAI-compatible provider, so provider/model changes and
nondeterministic generation can change the resulting report. Tests use deterministic
fake language models: they exercise the workflow and summary contract without a
network call, credential, or API cost.

## What this phase does not evaluate

This phase evaluates only the document-RAG path used by `pka ask`. It does not
evaluate Agent routing, long-term memory, graph answers, PDF/web/code ingestion, or
deployment behavior. Those broader evaluation questions remain later work rather
than being implied by one compact public suite.

# Phase 6B: GitHub Actions CI

## CI is not CD

Continuous integration answers whether a revision can be safely merged. This project
checks tests, static analysis, and the offline container contract. It does not publish
an image, deploy an application, or call an external LLM.

## Caches accelerate but do not validate

uv and BuildKit caches avoid repeated downloads and layer work. The lock-file check
and `uv sync --locked` still decide dependency correctness; a cache miss is slower,
not less correct.

## Least privilege and explicit supply chain inputs

The workflow requests only `contents: read`. Every Action is pinned by commit SHA, so
updating third-party workflow code is a visible source change rather than a mutable
tag update.

## Offline container smoke check

The CI image check runs the default `pka health` command. It catches Docker build and
entry-point regressions while avoiding private knowledge, model downloads, and API
credentials.

"""Regression tests for the checked-in Docker runtime contract."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_runs_the_offline_cli_as_a_non_root_user() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "uv.lock" in dockerfile
    assert "uv sync --locked --no-dev" in dockerfile
    assert "UV_PROJECT_ENVIRONMENT=/opt/venv" in dockerfile
    assert "COPY --from=builder --chown=pka:pka /opt/venv /opt/venv" in dockerfile
    assert "useradd --create-home --uid 10001 pka" in dockerfile
    assert "USER pka" in dockerfile
    assert 'ENTRYPOINT ["pka"]' in dockerfile
    assert 'CMD ["health"]' in dockerfile
    assert 'HEALTHCHECK CMD ["pka", "health"]' in dockerfile
    assert "HF_HOME=/home/pka/.cache/huggingface" in dockerfile
    assert "SENTENCE_TRANSFORMERS_HOME=/home/pka/.cache/sentence_transformers" in (
        dockerfile
    )


def test_dockerignore_excludes_sensitive_and_generated_runtime_state() -> None:
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {".env", ".pka/", ".venv/", ".git/"} <= ignored
    assert {"papers/", "references/", "research/"} <= ignored
    assert {
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
    } <= ignored


def test_documented_model_cache_stays_untracked_and_host_writable() -> None:
    ignored_by_git = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert ".pka-model-cache/" in ignored_by_git
    assert '--user "$(id -u):$(id -g)"' in readme

"""Regression tests for the GitHub Actions CI contract."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _workflow() -> str:
    return (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_ci_uses_approved_triggers_permissions_and_sha_pins() -> None:
    workflow = _workflow()

    assert "pull_request:" in workflow
    assert "branches: [main]" in workflow
    assert "contents: read" in workflow
    assert "cancel-in-progress: true" in workflow
    assert "@v" not in workflow
    assert "PKA_" not in workflow


def test_ci_runs_the_locked_python_quality_gate() -> None:
    workflow = _workflow()

    for command in (
        "uv sync --locked --all-groups",
        "uv run pytest",
        "uv run ruff format --check .",
        "uv run ruff check .",
        "uv run mypy src",
        "uv run pka health",
        "uv lock --check",
    ):
        assert command in workflow


def test_ci_builds_and_runs_the_offline_container_after_quality() -> None:
    workflow = _workflow()

    assert "needs: quality" in workflow
    assert "docker/setup-buildx-action@" in workflow
    assert (
        "docker/build-push-action@263435318d21b8e681c14492fe198d362a7d2c83" in workflow
    )
    assert "context: ." in workflow
    assert "load: true" in workflow
    assert "tags: pka-ci" in workflow
    assert "cache-from: type=gha" in workflow
    assert "cache-to: type=gha,mode=max" in workflow
    assert "docker run --rm pka-ci" in workflow
    assert "docker inspect --format" in workflow
    assert "docker push" not in workflow
    assert "docker login" not in workflow

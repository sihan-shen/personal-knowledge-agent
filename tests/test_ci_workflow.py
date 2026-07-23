"""Regression tests for the GitHub Actions CI contract."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APPROVED_ACTIONS = (
    "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
    "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
    "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b",
    "docker/setup-buildx-action@e468171a9de216ec08956ac3ada2f0791b6bd435",
    "docker/build-push-action@263435318d21b8e681c14492fe198d362a7d2c83",
)
ACTION_REFERENCE = re.compile(r"^\s*-\s+uses:\s+([^\s#]+)", re.MULTILINE)


def _workflow() -> str:
    return (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def _normalized_workflow() -> str:
    return " ".join(_workflow().split())


def test_ci_uses_approved_triggers_permissions_and_sha_pins() -> None:
    workflow = _workflow()

    assert "pull_request:" in workflow
    assert "branches: [main]" in workflow
    assert "contents: read" in workflow
    assert "cancel-in-progress: true" in workflow

    action_uses = ACTION_REFERENCE.findall(workflow)
    assert action_uses
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in action_uses)

    for action in APPROVED_ACTIONS:
        assert action in workflow

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
    assert "context: ." in workflow
    assert "load: true" in workflow
    assert "tags: pka-ci" in workflow
    assert "cache-from: type=gha" in workflow
    assert "cache-to: type=gha,mode=max" in workflow
    assert "docker run --rm pka-ci" in workflow
    expected_health_check = (
        "test \"$(docker inspect --format '{{json .Config.Healthcheck.Test}}' "
        'pka-ci)" = \'["CMD","pka","health"]\''
    )
    assert expected_health_check in _normalized_workflow()
    assert "docker push" not in workflow
    assert "docker login" not in workflow

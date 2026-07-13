"""Black-box tests for the command-line interface."""

from __future__ import annotations

import subprocess
import sys


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the package module as a user would invoke the CLI."""
    return subprocess.run(
        [sys.executable, "-m", "personal_knowledge_agent.cli", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_health_command_reports_healthy_project() -> None:
    result = run_cli("health")

    assert result.returncode == 0
    assert "Personal Knowledge Agent is healthy." in result.stdout


def test_unknown_command_returns_usage_error() -> None:
    result = run_cli("unknown")

    assert result.returncode != 0
    assert "usage:" in result.stderr.lower()

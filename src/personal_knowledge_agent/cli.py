"""Command-line interface for Personal Knowledge Agent."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the parser for supported user commands."""
    parser = argparse.ArgumentParser(prog="pka")
    parser.add_argument(
        "command",
        choices=("health",),
        help="Command to run. Currently supported: health.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the CLI and print the result of the requested command."""
    parsed_arguments = build_parser().parse_args(arguments)

    if parsed_arguments.command == "health":
        print("Personal Knowledge Agent is healthy.")


if __name__ == "__main__":
    main()

"""Entry point for the litpose command-line interface."""

from __future__ import annotations

import sys

from . import friendly
from .commands import COMMANDS


def _build_parser() -> friendly.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands registered.

    Returns:
        Configured ``ArgumentParser`` with all CLI subcommands attached.
    """
    parser = friendly.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Litpose command to run.",
        parser_class=friendly.ArgumentSubParser,
    )

    # Dynamically register all available commands
    for _name, module in COMMANDS.items():
        module.register_parser(subparsers)

    return parser


def main() -> None:
    """Entry point for the litpose CLI; parse arguments and dispatch to the appropriate command."""
    parser = _build_parser()

    # If no commands provided, display the help message.
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)  # type: ignore[arg-type]
        sys.exit(1)

    args = parser.parse_args()

    # Get the command handler dynamically
    command_handler = COMMANDS[args.command].handle

    # Run any migrations (from lightning_pose.migrations).
    from lightning_pose.migrations.migrations import run_migrations

    run_migrations()

    # Execute the command
    command_handler(args)

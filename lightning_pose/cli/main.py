"""Entry point for the litpose command-line interface."""

from __future__ import annotations

import logging
import sys
from importlib.metadata import version

from . import friendly
from .commands import COMMANDS


def _setup_logging(debug: bool = False) -> None:
    """Configure the lightning_pose package logger to emit to stdout.

    Args:
        debug: if True, set level to DEBUG; otherwise INFO.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    pkg_logger = logging.getLogger('lightning_pose')
    pkg_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    pkg_logger.addHandler(handler)
    pkg_logger.propagate = False


def _build_parser() -> friendly.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands registered.

    Returns:
        Configured ``ArgumentParser`` with all CLI subcommands attached.
    """
    parser = friendly.ArgumentParser()
    parser.add_argument(
        '--version',
        action='version',
        version=f'lightning-pose {version("lightning-pose")}',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='enable debug-level logging',
    )
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
    _setup_logging(debug=args.debug)

    # Get the command handler dynamically
    command_handler = COMMANDS[args.command].handle

    # Run any migrations (from lightning_pose.migrations).
    from lightning_pose.migrations.migrations import run_migrations

    run_migrations()

    # Execute the command
    command_handler(args)

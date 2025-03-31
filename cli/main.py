from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from . import friendly, types
from .commands import train, predict, crop, remap

def _build_parser():
    parser = friendly.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Litpose command to run.",
        parser_class=friendly.ArgumentSubParser,
    )

    # Import and register command parsers
    train.register_parser(subparsers)
    predict.register_parser(subparsers)
    crop.register_parser(subparsers)
    remap.register_parser(subparsers)

    return parser

def main():
    parser = _build_parser()

    # If no commands provided, display the help message.
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Map commands to their handlers
    command_handlers = {
        "train": train.handle,
        "predict": predict.handle,
        "crop": crop.handle,
        "remap": remap.handle,
    }

    # Execute the command
    command_handlers[args.command](args)

if __name__ == "__main__":
    main()

"""Convert command for the lightning-pose CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def register_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the convert command parser."""
    from textwrap import dedent

    description_text = dedent(
        """\
            Converts a labeled dataset from another pose estimation tool into a Lightning
            Pose project directory.

            ``dataset_path`` is dispatched by type:

            * a directory -- treated as a DeepLabCut project (``labeled-data/<video>/
              CollectedData*.{csv,h5}``).
            * a ``.slp`` file -- treated as a SLEAP ``.pkg.slp`` package. Export it from the
              SLEAP GUI via Predict -> Export Labels Package. Only single-view, single-animal
              SLEAP projects are supported.
            """
    )

    convert_parser = subparsers.add_parser(
        'convert',
        description=description_text,
        usage='litpose convert <dataset_path> --lp_dir LP_DIR',
    )
    convert_parser.add_argument(
        'dataset_path',
        type=Path,
        help='a DeepLabCut project directory, or a SLEAP .pkg.slp file',
    )
    convert_parser.add_argument(
        '--lp_dir',
        type=Path,
        required=True,
        help='destination Lightning Pose project directory; created if it does not exist',
    )
    return convert_parser


def get_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser for the `litpose convert` subcommand (for docs)."""
    parser = argparse.ArgumentParser(prog='litpose')
    subparsers = parser.add_subparsers(dest='command')
    return register_parser(subparsers)


def handle(args: argparse.Namespace) -> None:
    """Handle the convert command."""
    dataset_path: Path = args.dataset_path
    lp_dir: Path = args.lp_dir

    if dataset_path.is_dir():
        from lightning_pose.converters import dlc

        dlc.convert(dataset_path, lp_dir)
    elif dataset_path.suffix == '.slp':
        from lightning_pose.converters import sleap

        sleap.convert(dataset_path, lp_dir)
    else:
        raise ValueError(
            f'could not determine dataset type for {dataset_path}: expected a DeepLabCut '
            f'project directory or a SLEAP .slp file'
        )

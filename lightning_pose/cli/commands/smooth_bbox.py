"""Smooth-bbox command for the lightning-pose CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def register_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the smooth_bbox command parser."""
    import sys
    from textwrap import dedent

    is_building_docs = 'sphinx' in sys.modules
    _doc_link = (
        ':doc:`Cropzoom pipeline </source/user_guide_advanced/cropzoom_pipeline>`'
        if is_building_docs
        else (
            'https://lightning-pose.readthedocs.io/en/latest'
            '/source/user_guide_advanced/cropzoom_pipeline.html'
        )
    )

    description_text = dedent(
        f"""\
            Applies temporal smoothing to bounding-box CSV files produced by
            ``litpose create_bbox``.  Reads every ``*_bbox.csv`` file in the input
            directory, smooths the ``x``, ``y``, ``h``, and ``w`` columns, and writes
            the results to a new output directory together with a ``metadata.json``
            that records the smoothing parameters used.

            The smoothed bboxes can then be passed to ``litpose crop`` via
            ``--bbox_dir``.

            For an end-to-end usage example, see the user guide:
            {_doc_link}.
            """
    )

    parser = subparsers.add_parser(
        'smooth_bbox',
        description=description_text,
        usage='litpose smooth_bbox <bbox_dir> --output_dir <dir> [--method METHOD] [--window N]',
    )
    parser.add_argument(
        'bbox_dir',
        type=Path,
        help='directory containing raw *_bbox.csv files (output of litpose create_bbox)',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='directory where smoothed bbox files and metadata.json will be written',
    )
    parser.add_argument(
        '--method',
        type=str,
        default='median',
        help="smoothing method; currently only 'median' is supported (default: median)",
    )
    parser.add_argument(
        '--window',
        type=int,
        default=5,
        help='rolling window size in frames (default: 5)',
    )
    return parser


def get_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser for the ``litpose smooth_bbox`` subcommand (for docs)."""
    parser = argparse.ArgumentParser(prog='litpose')
    subparsers = parser.add_subparsers(dest='command')
    return register_parser(subparsers)


def handle(args: argparse.Namespace) -> None:
    """Handle the smooth_bbox command."""
    import lightning_pose.utils.cropzoom as cz

    cz.smooth_bbox(
        input_bbox_dir=args.bbox_dir,
        output_dir=args.output_dir,
        method=args.method,
        window=args.window,
    )

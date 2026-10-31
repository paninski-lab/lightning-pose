"""Create-bbox command for the lightning-pose CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from lightning_pose.cli import types
from lightning_pose.cli.bbox_args import add_bbox_sizing_args, resolve_bbox_sizing

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import lightning_pose.utils.cropzoom as cz  # noqa: F401
    from lightning_pose.api import Model  # noqa: F401


def register_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the create_bbox command parser."""
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
            Computes per-frame bounding boxes from detector predictions and saves them as
            CSV files.  Run ``litpose predict`` first to generate the predictions.

            For video inputs, bbox files are saved to::

                <model_dir>/
                └── video_preds/
                    └── <video_stem>_bbox.csv

            For labeled-frames inputs (CSV), bbox files are saved to::

                <model_dir>/
                └── image_preds/
                    └── <csv_file_name>/
                        └── bbox.csv

            The bounding boxes produced here can optionally be smoothed with
            ``litpose smooth_bbox`` before passing them to ``litpose crop``.

            For an end-to-end usage example, see the user guide:
            {_doc_link}.
            """
    )

    parser = subparsers.add_parser(
        'create_bbox',
        description=description_text,
        usage=(
            'litpose create_bbox <model_dir> <input_path:video|csv>...'
            ' [--crop_ratio=CROP_RATIO | --crop_size=CROP_SIZE]'
            ' [--anchor_keypoints=x,y,z]'
        ),
    )
    parser.add_argument(
        'model_dir', type=types.existing_model_dir, help='path to a detector model directory'
    )
    parser.add_argument(
        'input_path',
        type=Path,
        nargs='+',
        help='one or more video files, CSV files, or directories (directories are expanded to'
        ' their contained *.mp4 files)',
    )
    add_bbox_sizing_args(parser)
    return parser


def get_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser for the ``litpose create_bbox`` subcommand (for docs)."""
    parser = argparse.ArgumentParser(prog='litpose')
    subparsers = parser.add_subparsers(dest='command')
    return register_parser(subparsers)


def handle(args: argparse.Namespace) -> None:
    """Handle the create_bbox command."""
    import lightning_pose.utils.cropzoom as cz  # noqa: F811
    from lightning_pose.api import Model  # noqa: F811

    model = Model.from_dir(args.model_dir)

    sizing, anchor_keypoints = resolve_bbox_sizing(
        args.crop_ratio, args.crop_size, args.anchor_keypoints,
    )
    detector_cfg = OmegaConf.create({**sizing, 'anchor_keypoints': anchor_keypoints})

    input_paths: list[Path] = []
    for p in args.input_path:
        p = Path(p)
        if p.is_dir():
            logger.info(f'processing directory {p}')
            input_paths.extend(sorted(f for f in p.iterdir() if f.suffix == '.mp4'))
        else:
            input_paths.append(p)

    for input_path in input_paths:
        if input_path.suffix == '.mp4':
            input_preds_file = model.video_preds_dir() / (input_path.stem + '.csv')
            output_bbox_file = model.video_preds_dir() / (input_path.stem + '_bbox.csv')
        elif input_path.suffix == '.csv':
            preds_dir = model.image_preds_dir() / input_path.name
            input_preds_file = preds_dir / 'predictions.csv'
            output_bbox_file = preds_dir / 'bbox.csv'
        else:
            raise NotImplementedError('only mp4 and csv files are supported.')

        logger.info(f'creating bboxes for {input_path.name}')
        cz.generate_bbox(
            input_preds_file=input_preds_file,
            detector_cfg=detector_cfg,
            output_bbox_file=output_bbox_file,
        )

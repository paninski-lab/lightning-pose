"""Create-bbox command for the lightning-pose CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from .. import types

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
    parser.add_argument(
        '--crop_ratio',
        type=float,
        default=None,
        help=(
            'size the bounding box this many times larger than the animal keypoint span'
            ' (default 2.0 when neither --crop_ratio nor --crop_size is given).'
            ' Mutually exclusive with --crop_size.'
        ),
    )
    parser.add_argument(
        '--crop_size',
        type=int,
        default=None,
        help=(
            'fixed square bounding box side length in pixels, centred on the per-frame mean'
            ' of the anchor keypoints. Mutually exclusive with --crop_ratio.'
        ),
    )
    parser.add_argument(
        '--anchor_keypoints',
        type=str,
        default='',
        help='comma-separated list of anchor keypoint names; defaults to all keypoints',
    )
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

    crop_ratio = args.crop_ratio
    crop_size = args.crop_size

    if crop_ratio is not None and crop_size is not None:
        raise ValueError('--crop_ratio and --crop_size are mutually exclusive.')
    if crop_ratio is None and crop_size is None:
        crop_ratio = 2.0

    anchor_keypoints = args.anchor_keypoints.split(',') if args.anchor_keypoints else []

    if crop_size is not None:
        if crop_size <= 0:
            raise ValueError(f'--crop_size must be a positive integer, got {crop_size}.')
        detector_cfg = OmegaConf.create({
            'crop_height': crop_size,
            'crop_width': crop_size,
            'anchor_keypoints': anchor_keypoints,
        })
    else:
        assert crop_ratio is not None
        if crop_ratio <= 1:
            raise ValueError(f'--crop_ratio must be greater than 1, got {crop_ratio}.')
        detector_cfg = OmegaConf.create({
            'crop_ratio': crop_ratio,
            'anchor_keypoints': anchor_keypoints,
        })

    input_paths: list[Path] = []
    for p in args.input_path:
        p = Path(p)
        if p.is_dir():
            print(f'Processing directory {p}')
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

        print(f'Creating bboxes for {input_path.name}')
        cz.generate_bbox(
            input_preds_file=input_preds_file,
            detector_cfg=detector_cfg,
            output_bbox_file=output_bbox_file,
        )

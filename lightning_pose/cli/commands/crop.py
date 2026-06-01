"""Crop command for the lightning-pose CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import types

if TYPE_CHECKING:
    import lightning_pose.utils.cropzoom as cz  # noqa: F401
    from lightning_pose.api import Model  # noqa: F401


def register_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the crop command parser."""
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
            Crops a video or labeled frames using pre-computed bounding boxes.
            Run ``litpose create_bbox`` (and optionally ``litpose smooth_bbox``) first.

            Cropped videos are saved to::

                <model_dir>/
                └── cropped_videos/
                    └── cropped_<video_filename>.mp4

            Cropped images and a remapped labels CSV are saved to::

                <model_dir>/
                └── cropped_images/
                        └── a/b/c/<image_name>.png
                <model_dir>/
                └── image_preds/
                    └── <csv_file_name>/
                        └── cropped_<csv_file_name>.csv

            When ``--bbox_dir`` is omitted, bbox files are read from the default
            locations written by ``litpose create_bbox``.  Pass ``--bbox_dir`` to use
            bboxes from a different source (e.g. output of ``litpose smooth_bbox`` or
            bboxes produced by an external tool).

            For an end-to-end usage example of the Cropzoom workflow, see the user guide:
            {_doc_link}.
            """
    )

    crop_parser = subparsers.add_parser(
        'crop',
        description=description_text,
        usage=(
            'litpose crop <model_dir> <input_path:video|csv>...'
            ' [--bbox_dir=BBOX_DIR]'
        ),
    )
    crop_parser.add_argument(
        'model_dir', type=types.existing_model_dir, help='path to a model directory'
    )
    crop_parser.add_argument('input_path', type=Path, nargs='+', help='one or more files')
    crop_parser.add_argument(
        '--bbox_dir',
        type=Path,
        default=None,
        help=(
            'directory containing bbox CSV files to use for cropping. '
            'For videos, looks for <bbox_dir>/<video_stem>_bbox.csv; '
            'for CSV inputs, looks for <bbox_dir>/bbox.csv. '
            'Defaults to the location written by litpose create_bbox.'
        ),
    )
    return crop_parser


def get_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser for the `litpose crop` subcommand (for docs)."""
    parser = argparse.ArgumentParser(prog='litpose')
    subparsers = parser.add_subparsers(dest='command')
    return register_parser(subparsers)


def handle(args: argparse.Namespace) -> None:
    """Handle the crop command."""
    import lightning_pose.utils.cropzoom as cz  # noqa: F811
    from lightning_pose.api import Model  # noqa: F811

    model_dir = args.model_dir
    model = Model.from_dir(model_dir)

    # Make both cropped_images and cropped_videos dirs. Reason: After this, the user
    # will train a pose model, and current code in io utils checks that both
    # data_dir and videos_dir are present. if we just create one or the other,
    # the check will fail.
    model.cropped_data_dir().mkdir(parents=True, exist_ok=True)
    model.cropped_videos_dir().mkdir(parents=True, exist_ok=True)

    bbox_dir = args.bbox_dir

    for input_path in [Path(p) for p in args.input_path]:
        if input_path.suffix == '.mp4':
            if bbox_dir is not None:
                input_bbox_file = bbox_dir / (input_path.stem + '_bbox.csv')
            else:
                input_bbox_file = model.video_preds_dir() / (input_path.stem + '_bbox.csv')
            output_file = model.cropped_videos_dir() / ('cropped_' + input_path.name)

            cz.crop_video(
                input_video_file=input_path,
                input_bbox_file=input_bbox_file,
                output_file=output_file,
            )
        elif input_path.suffix == '.csv':
            preds_dir = model.image_preds_dir() / input_path.name
            input_data_dir = Path(model.config.cfg.data.data_dir)
            cropped_data_dir = model.cropped_data_dir()

            if bbox_dir is not None:
                input_bbox_file = bbox_dir / 'bbox.csv'
            else:
                input_bbox_file = preds_dir / 'bbox.csv'
            output_csv_file_path = preds_dir / ('cropped_' + input_path.name)

            cz.crop_labeled_frames(
                input_data_dir=input_data_dir,
                input_csv_file=input_path,
                input_bbox_file=input_bbox_file,
                output_data_dir=cropped_data_dir,
                output_csv_file=output_csv_file_path,
            )
        else:
            raise NotImplementedError('only mp4 and csv files are supported.')

"""Predict command for the lightning-pose CLI."""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

from .. import types

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lightning_pose.api import Model
    from lightning_pose.api.model import _Precision

# Friendly CLI names -> PyTorch Lightning precision strings used internally.
_PRECISION_CHOICES: dict[str, _Precision] = {
    "fp32": "32-true",
    "fp16": "16-mixed",
    "bf16": "bf16-mixed",
}


def register_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the predict command parser."""
    predict_parser = subparsers.add_parser(
        "predict",
        description=textwrap.dedent(
            """\
        Predicts keypoints on videos or images.

          Video predictions are saved to::

            <model_dir>/
            └── video_preds/
                ├── <video_filename>.csv              (predictions)
                ├── <video_filename>_<metric>.csv     (losses)
                └── labeled_videos/
                    └── <video_filename>_labeled.mp4

          Image predictions are saved to::

            <model_dir>/
            └── image_preds/
                └── <image_dirname | csv_filename | timestamp>/
                    ├── predictions.csv
                    ├── predictions_<metric>.csv      (losses)
                    └── <image_filename>_labeled.png
        """
        ),
        usage="litpose predict <model_dir> <input_path:video|image|dir|csv>...  [OPTIONS]",
    )
    predict_parser.add_argument(
        "model_dir", type=types.existing_model_dir, help="path to a model directory"
    )

    predict_parser.add_argument(
        "input_path",
        type=Path,
        nargs="+",
        help=textwrap.dedent(
            """\
            one or more  video files, image files, CSV files, or directories to run prediction on

            * directories: iterates over videos or images in the directory
            * CSV file: must be formatted as a label file. predicts on the frames and computes
                pixel error against keypoint labels
            """
        ),
    )
    predict_parser.add_argument(
        "--overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="overrides attributes of the config file. Uses hydra syntax:\n"
        "https://hydra.cc/docs/advanced/override_grammar/basic/",
    )

    predict_parser.add_argument(
        "--precision",
        choices=sorted(_PRECISION_CHOICES.keys()),
        default="fp32",
        help=(
            "precision to run inference at. Does not affect the checkpoint "
            "on disk -- weights stay fp32; this only controls precision during "
            "the forward pass. Default: fp32."
        ),
    )

    predict_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite videos that already have prediction files",
    )

    predict_parser.add_argument(
        "--bbox_dir",
        type=Path,
        default=None,
        help=(
            "directory containing bbox CSV files produced by ``litpose create_bbox`` or a "
            "compatible external source. For CSV inputs, looks for ``bbox.csv`` inside this "
            "directory. For video inputs, looks for ``<video_stem>_bbox.csv`` inside this "
            "directory. When provided, each frame is cropped to its bounding box before "
            "being passed to the model, and predictions are saved in the original coordinate "
            "space."
        ),
    )

    post_prediction_args = predict_parser.add_argument_group("post-prediction")
    post_prediction_args.add_argument(
        "--skip_viz",
        action="store_true",
        help="skip generating prediction-annotated images/videos",
    )

    # For app use only.
    post_prediction_args.add_argument(
        "--progress_file",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return predict_parser


def get_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser for the `litpose predict` subcommand (for docs)."""
    parser = argparse.ArgumentParser(prog="litpose")
    subparsers = parser.add_subparsers(dest="command")
    return register_parser(subparsers)


def handle(args: argparse.Namespace) -> None:
    """Handle the predict command."""
    # Delay this import because it's slow.
    from lightning_pose.api import Model

    model = Model.from_dir2(
        args.model_dir,
        hydra_overrides=args.overrides,
        precision=_PRECISION_CHOICES[args.precision],
    )
    input_paths = [Path(p) for p in args.input_path]

    if model.config.is_multi_view():
        _predict_multi_type_multi_view(
            model, input_paths, args.skip_viz, not args.overwrite, progress_file=args.progress_file
        )
    else:
        for p in input_paths:
            _predict_multi_type(
                model, p, args.skip_viz, not args.overwrite,
                progress_file=args.progress_file,
                bbox_dir=args.bbox_dir,
            )


def _predict_multi_type(
    model: Model,
    path: Path,
    skip_viz: bool,
    skip_existing: bool,
    progress_file: Path | None = None,
    bbox_dir: Path | None = None,
) -> None:
    """Run prediction on a single path, dispatching to video, CSV, or directory handling.

    Args:
        model: the model to run prediction with.
        path: input file or directory to predict on.
        skip_viz: if True, skip generating labeled visualization outputs.
        skip_existing: if True, skip predictions for which an output CSV already exists.
        progress_file: optional path to write prediction progress as JSON.
        bbox_dir: optional directory containing bbox CSV files. For CSV inputs, ``bbox.csv``
            inside this directory is used. For video inputs, ``<stem>_bbox.csv`` is used.
    """
    if path.is_dir():
        image_files = [p for p in path.iterdir() if p.is_file() and p.suffix in [".png", ".jpg"]]
        video_files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".mp4"]

        if len(image_files) > 0:
            raise NotImplementedError("Predicting on image dir.")

        logger.info(f'processing directory {path}')
        for p in video_files:
            _predict_multi_type(
                model, p, skip_viz, skip_existing,
                progress_file=progress_file,
                bbox_dir=bbox_dir,
            )

    elif path.suffix == ".mp4":
        # Check if prediction file already exists
        prediction_csv_file = model.video_preds_dir() / f"{path.stem}.csv"
        if skip_existing and prediction_csv_file.exists():
            logger.info(f'skipping {path} (prediction file already exists)')
            return

        model.predict_on_video_file(
            video_file=path,
            generate_labeled_video=(not skip_viz),
            progress_file=progress_file,
            bbox_file=bbox_dir / f'{path.stem}_bbox.csv' if bbox_dir is not None else None,
        )
    elif path.suffix == ".csv":
        # Check if prediction file already exists
        prediction_csv_file = model.image_preds_dir() / path.name / "predictions.csv"
        if skip_existing and prediction_csv_file.exists():
            logger.info(f'skipping {path} (prediction file already exists)')
            return

        model.predict_on_label_csv(
            csv_file=path,
            bbox_file=bbox_dir / "bbox.csv" if bbox_dir is not None else None,
        )
    elif path.suffix in [".png", ".jpg"]:
        raise NotImplementedError("Not yet implemented: predicting on image files.")
    else:
        pass


def _predict_multi_type_multi_view(
    model: Model,
    paths: list[Path],
    skip_viz: bool,
    skip_existing: bool,
    progress_file: Path | None = None,
) -> None:
    """Run multi-view prediction on a list of paths (videos or directories).

    Args:
        model: the multi-view model to run prediction with.
        paths: list of input video files or directories to predict on.
        skip_viz: if True, skip generating labeled visualization outputs.
        skip_existing: if True, skip sessions for which output CSVs already exist.
        progress_file: optional path to write prediction progress as JSON.
    """
    # delay this import because it's slow
    from lightning_pose.utils.io import (
        extract_session_name_from_video,
        split_video_files_by_view,
    )

    # if we pass in all videos, collect them into session batches and process
    if all(path.suffix == ".mp4" for path in paths):
        video_files_split = split_video_files_by_view(
            paths, model.config.cfg.data.view_names
        )
        session_names = [
            extract_session_name_from_video(
                video_file_per_view[0].name, model.config.cfg.data.view_names
            )
            for video_file_per_view in video_files_split
        ]
        logger.info(
            f'grouped {len(paths)} videos into {len(video_files_split)} sessions:\n'
            + pformat(session_names)
        )
        for video_file_per_view in video_files_split:
            if skip_existing and all(
                (model.video_preds_dir() / f"{Path(video).stem}.csv").exists()
                for video in video_file_per_view
            ):
                session_name = extract_session_name_from_video(
                    Path(video_file_per_view[0]).name, model.config.cfg.data.view_names
                )
                logger.info(f'skipping {session_name} (prediction file already exists)')
                continue

            model.predict_on_video_file_multiview(
                video_file_per_view,
                generate_labeled_video=not skip_viz,
                progress_file=progress_file,
            )
    # if we have a list of directories, we process the videos in each separately
    elif all(path.is_dir() for path in paths):
        for path in paths:
            video_files = [
                p for p in path.iterdir() if p.is_file() and p.suffix == ".mp4"
            ]
            if len(video_files) > 0:
                logger.info(f'processing directory {path}')

                _predict_multi_type_multi_view(
                    model, video_files, skip_viz, skip_existing, progress_file=progress_file
                )
            else:
                logger.info(f'skipping {path}: no videos found')
    else:
        raise NotImplementedError(
            "For multi view model predictions, either pass in multiple video views to be "
            "predicted, or a directory containing videos"
        )

"""Predict command for the lightning-pose CLI."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING

from .. import types

if TYPE_CHECKING:
    from lightning_pose.api.model import Model


def register_parser(subparsers):
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
            * CSV file: must be formatted as a label file. predicts on the frames and computes pixel error
                    against keypoint labels
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
        "--overwrite",
        action="store_true",
        help="overwrite videos that already have prediction files",
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


def get_parser():
    """Return an ArgumentParser for the `litpose predict` subcommand (for docs)."""
    import argparse

    parser = argparse.ArgumentParser(prog="litpose")
    subparsers = parser.add_subparsers(dest="command")
    return register_parser(subparsers)


def handle(args):
    """Handle the predict command."""
    # Delay this import because it's slow.
    from lightning_pose.api.model import Model

    model = Model.from_dir2(args.model_dir, hydra_overrides=args.overrides)
    input_paths = [Path(p) for p in args.input_path]

    if model.config.is_multi_view():
        _predict_multi_type_multi_view(
            model, input_paths, args.skip_viz, not args.overwrite, progress_file=args.progress_file
        )
    else:
        for p in input_paths:
            _predict_multi_type(
                model, p, args.skip_viz, not args.overwrite, progress_file=args.progress_file
            )


def _predict_multi_type(
    model: Model,
    path: Path,
    skip_viz: bool,
    skip_existing: bool,
    progress_file: Path | None = None,
):
    if path.is_dir():
        image_files = [p for p in path.iterdir() if p.is_file() and p.suffix in [".png", ".jpg"]]
        video_files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".mp4"]

        if len(image_files) > 0:
            raise NotImplementedError("Predicting on image dir.")

        print(f"Processing directory {path}")
        for p in video_files:
            _predict_multi_type(model, p, skip_viz, skip_existing, progress_file=progress_file)

    elif path.suffix == ".mp4":
        # Check if prediction file already exists
        prediction_csv_file = model.video_preds_dir() / f"{path.stem}.csv"
        if skip_existing and prediction_csv_file.exists():
            print(f"Skipping {path} (prediction file already exists)")
            return

        model.predict_on_video_file(
            video_file=path,
            generate_labeled_video=(not skip_viz),
            progress_file=progress_file,
        )
    elif path.suffix == ".csv":
        # Check if prediction file already exists
        prediction_csv_file = model.image_preds_dir() / path.name / "predictions.csv"
        if skip_existing and prediction_csv_file.exists():
            print(f"Skipping {path} (prediction file already exists)")
            return

        model.predict_on_label_csv(
            csv_file=path,
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
):
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
        print(f"Grouped {len(paths)} videos into {len(video_files_split)} sessions:")
        pprint(
            [
                extract_session_name_from_video(
                    video_file_per_view[0].name, model.config.cfg.data.view_names
                )
                for video_file_per_view in video_files_split
            ],
        )
        for video_file_per_view in video_files_split:
            if skip_existing and all(
                (model.video_preds_dir() / f"{Path(video).stem}.csv").exists()
                for video in video_file_per_view
            ):
                session_name = extract_session_name_from_video(
                    Path(video_file_per_view[0]).name, model.config.cfg.data.view_names
                )
                print(f"Skipping {session_name} (prediction file already exists)")
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
                print(f"Processing directory {path}")

                _predict_multi_type_multi_view(
                    model, video_files, skip_viz, skip_existing, progress_file=progress_file
                )
            else:
                print(f"Skipping {path}: no videos found.")
    else:
        raise NotImplementedError(
            "For multi view model predictions, either pass in multiple video views to be predicted, or a directory containing videos"
        )

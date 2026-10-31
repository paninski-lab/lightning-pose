"""Predict command for the lightning-pose CLI."""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, get_args

from lightning_pose.cli import types
from lightning_pose.cli.bbox_args import add_bbox_sizing_args, resolve_bbox_sizing
from lightning_pose.utils.inference_types import _OnnxPrecision, _Precision, _Reader, _Runtime

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lightning_pose.api import Model

# --precision, --runtime, --reader, and --onnx-precision choices are each
# derived from the corresponding lightning_pose.utils.inference_types alias
_PRECISION_CHOICES = get_args(_Precision)
_RUNTIME_CHOICES = get_args(_Runtime)
_READER_CHOICES = get_args(_Reader)
_ONNX_PRECISION_CHOICES = get_args(_OnnxPrecision)


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

          With ``--detector``, a detector model and this (pose) model run back
          to back in memory: the detector's keypoints define a per-frame
          bounding box, each frame is cropped to it, and predictions are
          returned in original frame coordinates. Outputs are saved next to
          the pose model's video predictions::

            <model_dir>/
            └── video_preds/
                ├── <video_filename>.csv         (original coordinates)
                └── <video_filename>_bbox.csv    (per-frame bounding boxes)

          Image predictions are saved to::

            <model_dir>/
            └── image_preds/
                └── <image_dirname | csv_filename | timestamp>/
                    ├── predictions.csv
                    ├── predictions_<metric>.csv      (losses)
                    └── <image_filename>_labeled.png
        """
        ),
        usage=(
            "litpose predict <model_dir> <input_path:video|image|dir|csv>...  [OPTIONS]\n"
            "       litpose predict <pose_model_dir> <input_path:video|dir>... "
            "--detector <detector_model_dir> [OPTIONS]"
        ),
    )
    predict_parser.add_argument(
        "model_dir",
        type=types.existing_model_dir,
        help="path to a model directory (the pose model, when --detector is given)",
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
        choices=sorted(_PRECISION_CHOICES),
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
        "--batch_size",
        type=int,
        default=None,
        help=(
            "override the batch size used during inference. For video inputs, sets "
            "dali.base.predict.sequence_length (or dali.context.predict.sequence_length "
            "for context models) -- the number of frames DALI loads per batch. For "
            "CSV/image inputs, sets training.val_batch_size. Lower this if inference "
            "runs out of GPU memory; raise it to speed up inference on a GPU with "
            "memory to spare. Equivalent to passing the corresponding --overrides "
            "KEY=VALUE by hand."
        ),
    )

    predict_parser.add_argument(
        "--bbox_dir",
        type=Path,
        default=None,
        help=(
            "directory containing bbox CSV files produced by ``litpose create_bbox`` or a "
            "compatible external source. For CSV inputs, looks for ``bbox.csv`` inside this "
            "directory. For video inputs, looks for ``<video_stem>_bbox.csv`` inside this "
            "directory; for multiview models, one ``<video_stem>_bbox.csv`` per view video. "
            "When provided, each frame is cropped to its bounding box before "
            "being passed to the model, and predictions are saved in the original coordinate "
            "space."
        ),
    )

    predict_parser.add_argument(
        "--runtime",
        choices=sorted(_RUNTIME_CHOICES),
        default="eager",
        help=(
            "inference backend. 'eager' (default) runs the trained checkpoint. "
            "'onnx' and 'tensorrt' run a session previously built with "
            "`litpose export --runtime onnx` / `--runtime tensorrt`. With "
            "--runtime onnx or --runtime tensorrt, --precision is ignored -- "
            "the exported file's own precision is what runs."
        ),
    )

    predict_parser.add_argument(
        "--onnx-precision",
        choices=sorted(_ONNX_PRECISION_CHOICES),
        default=None,
        help=(
            "which ONNX export to load (or, for --runtime tensorrt, which "
            "engine cache). Only used with --runtime onnx or --runtime "
            "tensorrt. If omitted and exactly one export exists, it is used "
            "automatically; if several exist, pass this to disambiguate."
        ),
    )

    predict_parser.add_argument(
        "--reader",
        choices=sorted(_READER_CHOICES),
        default=None,
        help=(
            "video-reading backend for video inputs. 'dali', 'pynvvc', or 'opencv'. If "
            "omitted (default), auto-selects 'pynvvc' when it's usable on this machine "
            "for the given video, else 'dali' when it's installed, else 'opencv' (the "
            "portable fallback, always available). Independent of --runtime/--compile "
            "-- this only controls video ingestion, not model execution. Ignored for "
            "CSV/image inputs."
        ),
    )

    predict_parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="compile the model with torch.compile() for faster inference. The first "
        "prediction after compiling is slower due to compilation overhead.",
    )

    cropzoom_args = predict_parser.add_argument_group(
        "cropzoom pipeline",
        "run a detector and this (pose) model back to back in memory; video inputs only",
    )
    cropzoom_args.add_argument(
        "--detector",
        type=types.existing_model_dir,
        default=None,
        help=(
            "path to a detector model directory. When given, each frame is passed through the "
            "detector, cropped to a bounding box computed from its keypoints, and passed "
            "through the pose model (model_dir). Incompatible with --bbox_dir, --overrides, "
            "--runtime onnx/tensorrt, and --compile. Per-frame boxes are not smoothed; frames "
            "without a usable detection reuse the previous frame's box."
        ),
    )
    add_bbox_sizing_args(cropzoom_args)
    cropzoom_args.add_argument(
        "--detector_confidence_thresh",
        type=float,
        default=None,
        help=(
            "a frame whose detector confidence is below this value for any anchor keypoint "
            "reuses the previous frame's box. 0 disables the check. Default: 0.5."
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

    if args.detector is not None:
        _validate_cropzoom_args(args)
        _predict_cropzoom(args)
        return
    _reject_cropzoom_only_args(args)

    # Checked before constructing the model so the user gets this message rather
    # than the equivalent RuntimeError from Model.compile(), which would surface
    # as a raw traceback well after the ONNX session had already been built.
    if args.compile and args.runtime != "eager":
        raise ValueError(
            f"--compile is only supported with --runtime eager, but --runtime "
            f"{args.runtime} was given. torch.compile() has no effect on an "
            f"ONNX Runtime or TensorRT session."
        )

    hydra_overrides = list(args.overrides) if args.overrides else []
    if args.batch_size is not None:
        hydra_overrides += [
            f"training.val_batch_size={args.batch_size}",
            f"dali.base.predict.sequence_length={args.batch_size}",
            f"dali.context.predict.sequence_length={args.batch_size}",
        ]

    model = Model.from_dir2(
        args.model_dir,
        hydra_overrides=hydra_overrides or None,
        precision=args.precision,
        runtime=args.runtime,
        onnx_precision=args.onnx_precision,
    )
    input_paths = [Path(p) for p in args.input_path]

    if args.compile:
        model.compile()

    if model.config.is_multi_view():
        _predict_multi_type_multi_view(
            model, input_paths, args.skip_viz, not args.overwrite,
            progress_file=args.progress_file, reader=args.reader,
            bbox_dir=args.bbox_dir,
        )
    else:
        for p in input_paths:
            _predict_multi_type(
                model, p, args.skip_viz, not args.overwrite,
                progress_file=args.progress_file,
                bbox_dir=args.bbox_dir,
                reader=args.reader,
            )


_CROPZOOM_ONLY_FLAGS = (
    ("crop_ratio", "--crop_ratio"),
    ("crop_size", "--crop_size"),
    ("detector_confidence_thresh", "--detector_confidence_thresh"),
)


def _reject_cropzoom_only_args(args: argparse.Namespace) -> None:
    """Raise if a cropzoom-pipeline option was given without ``--detector``.

    Args:
        args: parsed command-line arguments.

    Raises:
        ValueError: if any of the bbox sizing / detector options is set.
    """
    given = [flag for attr, flag in _CROPZOOM_ONLY_FLAGS if getattr(args, attr) is not None]
    if args.anchor_keypoints:
        given.append("--anchor_keypoints")
    if given:
        raise ValueError(f"{', '.join(given)} only take effect together with --detector.")


def _validate_cropzoom_args(args: argparse.Namespace) -> None:
    """Raise if ``--detector`` is combined with options the pipeline does not support.

    Checked before any model is built so the user gets this message up front.

    Args:
        args: parsed command-line arguments.

    Raises:
        ValueError: if an incompatible option is set.
    """
    incompatible = {
        "--bbox_dir": args.bbox_dir is not None,
        "--overrides": bool(args.overrides),
        "--compile": args.compile,
        "--runtime": args.runtime != "eager",
    }
    given = [flag for flag, is_set in incompatible.items() if is_set]
    if given:
        raise ValueError(
            f"{', '.join(given)} cannot be combined with --detector (the pipeline computes "
            f"its own bboxes and runs both models in eager mode)."
        )


def _predict_cropzoom(args: argparse.Namespace) -> None:
    """Run the detector -> crop -> pose estimator pipeline on the input videos.

    Args:
        args: parsed command-line arguments; ``args.detector`` is set.

    Raises:
        NotImplementedError: if an input is not a video or directory of videos.
        ValueError: if the bbox sizing options are invalid.
    """
    # delay this import because it's slow
    from lightning_pose.pipelines import CropZoomPipeline

    sizing, anchor_keypoints = resolve_bbox_sizing(
        args.crop_ratio, args.crop_size, args.anchor_keypoints,
    )
    thresh = args.detector_confidence_thresh
    pipeline = CropZoomPipeline.from_dirs(
        detector_dir=args.detector,
        pose_estimator_dir=args.model_dir,
        anchor_keypoints=anchor_keypoints,
        confidence_thresh=0.5 if thresh is None else thresh,
        precision=args.precision,
        **sizing,
    )

    video_files: list[Path] = []
    for p in (Path(p) for p in args.input_path):
        if p.is_dir():
            logger.info(f'processing directory {p}')
            video_files.extend(sorted(f for f in p.iterdir() if f.suffix == ".mp4"))
        elif p.suffix == ".mp4":
            video_files.append(p)
        else:
            raise NotImplementedError(
                f"--detector only supports video inputs (.mp4 files or directories of them), "
                f"got {p}."
            )

    output_dir = pipeline.pose_estimator.video_preds_dir()
    extra_kwargs = {} if args.batch_size is None else {"batch_size": args.batch_size}
    for video_file in video_files:
        if not args.overwrite and (output_dir / f"{video_file.stem}.csv").exists():
            logger.info(f'skipping {video_file} (prediction file already exists)')
            continue
        pipeline.predict_on_video_file(
            video_file,
            output_dir,
            reader=args.reader,
            progress_file=args.progress_file,
            generate_labeled_video=not args.skip_viz,
            **extra_kwargs,
        )


def _predict_multi_type(
    model: Model,
    path: Path,
    skip_viz: bool,
    skip_existing: bool,
    progress_file: Path | None = None,
    bbox_dir: Path | None = None,
    reader: _Reader | None = None,
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
                reader=reader,
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
            reader=reader,
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
    reader: _Reader | None = None,
    bbox_dir: Path | None = None,
) -> None:
    """Run multi-view prediction on a list of paths (videos or directories).

    Args:
        model: the multi-view model to run prediction with.
        paths: list of input video files or directories to predict on.
        skip_viz: if True, skip generating labeled visualization outputs.
        skip_existing: if True, skip sessions for which output CSVs already exist.
        progress_file: optional path to write prediction progress as JSON.
        reader: optional video-reading backend; None auto-selects.
        bbox_dir: optional directory containing a ``<video_stem>_bbox.csv`` for every view's
            video. When provided, each view is cropped to its bboxes before inference.
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

            bbox_file_per_view = None
            if bbox_dir is not None:
                bbox_file_per_view = [
                    bbox_dir / f'{Path(video).stem}_bbox.csv' for video in video_file_per_view
                ]
                missing = [f for f in bbox_file_per_view if not f.exists()]
                if missing:
                    raise FileNotFoundError(f'bbox file(s) not found: {missing}')

            model.predict_on_video_file_multiview(
                video_file_per_view,
                generate_labeled_video=not skip_viz,
                progress_file=progress_file,
                reader=reader,
                bbox_file_per_view=bbox_file_per_view,
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
                    model, video_files, skip_viz, skip_existing,
                    progress_file=progress_file, reader=reader, bbox_dir=bbox_dir,
                )
            else:
                logger.info(f'skipping {path}: no videos found')
    else:
        raise NotImplementedError(
            "For multi view model predictions, either pass in multiple video views to be "
            "predicted, or a directory containing videos"
        )

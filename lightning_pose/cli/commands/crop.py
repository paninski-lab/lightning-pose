"""Crop command for the lightning-pose CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from .. import types

if TYPE_CHECKING:
    import lightning_pose.utils.cropzoom as cz  # noqa: F401
    from lightning_pose.api.model import Model  # noqa: F401


def register_parser(subparsers):
    """Register the crop command parser."""
    from textwrap import dedent

    crop_parser = subparsers.add_parser(
        "crop",
        description=dedent(
            """\
            Crops a video or labeled frames based on model predictions.
            Requires model predictions to already have been generated using `litpose predict`.

            Cropped videos are saved to:
                <model_dir>/
                └── video_preds/
                    ├── <video_filename>.csv              (predictions)
                    ├── <video_filename>_bbox.csv         (bbox)
                    └── remapped_<video_filename>.csv     (TODO move to remap command)
                └── cropped_videos/
                    └── cropped_<video_filename>.mp4      (cropped video)

            Cropped images are saved to:
                <model_dir>/
                └── image_preds/
                    └── <csv_file_name>/
                        ├── predictions.csv
                        ├── bbox.csv                      (bbox)
                        └── cropped_<csv_file_name>.csv   (cropped labels)
                └── cropped_images/
                        └── a/b/c/<image_name>.png        (cropped images)\
            """
        ),
        usage="litpose crop <model_dir> <input_path:video|csv>... --crop_ratio=CROP_RATIO --anchor_keypoints=x,y,z",  # noqa
    )
    crop_parser.add_argument(
        "model_dir", type=types.existing_model_dir, help="path to a model directory"
    )

    crop_parser.add_argument(
        "input_path", type=Path, nargs="+", help="one or more files"
    )
    crop_parser.add_argument(
        "--crop_ratio",
        type=float,
        default=2.0,
        help="Crop a bounding box this much larger than the animal. Default is 2.",
    )
    crop_parser.add_argument(
        "--anchor_keypoints",
        type=str,
        default="",  # Or a reasonable default like "0,0,0" if appropriate
        help="Comma-separated list of anchor keypoint names, defaults to all keypoints",
    )


def handle(args):
    """Handle the crop command."""
    import lightning_pose.utils.cropzoom as cz  # noqa: F811
    from lightning_pose.api.model import Model  # noqa: F811

    model_dir = args.model_dir
    model = Model.from_dir(model_dir)

    # Make both cropped_images and cropped_videos dirs. Reason: After this, the user
    # will train a pose model, and current code in io utils checks that both
    # data_dir and videos_dir are present. if we just create one or the other,
    # the check will fail.
    model.cropped_data_dir().mkdir(parents=True, exist_ok=True)
    model.cropped_videos_dir().mkdir(parents=True, exist_ok=True)

    input_paths = [Path(p) for p in args.input_path]

    detector_cfg = OmegaConf.create(
        {
            "crop_ratio": args.crop_ratio,
            "anchor_keypoints": (
                args.anchor_keypoints.split(",") if args.anchor_keypoints else []
            ),
        }
    )
    assert detector_cfg.crop_ratio > 1

    for input_path in input_paths:
        if input_path.suffix == ".mp4":
            input_preds_file = model.video_preds_dir() / (input_path.stem + ".csv")
            output_bbox_file = model.video_preds_dir() / (input_path.stem + "_bbox.csv")
            output_file = model.cropped_videos_dir() / ("cropped_" + input_path.name)

            cz.generate_cropped_video(
                input_video_file=input_path,
                input_preds_file=input_preds_file,
                detector_cfg=detector_cfg,
                output_bbox_file=output_bbox_file,
                output_file=output_file,
            )
        elif input_path.suffix == ".csv":
            preds_dir = model.image_preds_dir() / input_path.name
            input_data_dir = Path(model.config.cfg.data.data_dir)
            cropped_data_dir = model.cropped_data_dir()

            output_bbox_file = preds_dir / "bbox.csv"
            output_csv_file_path = preds_dir / ("cropped_" + input_path.name)
            input_preds_file = preds_dir / "predictions.csv"
            cz.generate_cropped_labeled_frames(
                input_data_dir=input_data_dir,
                input_csv_file=input_path,
                input_preds_file=input_preds_file,
                detector_cfg=detector_cfg,
                output_data_dir=cropped_data_dir,
                output_bbox_file=output_bbox_file,
                output_csv_file=output_csv_file_path,
            )
        else:
            raise NotImplementedError("Only mp4 and csv files are supported.")

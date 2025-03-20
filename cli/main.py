from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

# Don't import anything from torch or lightning_pose until needed.
# These imports are slow and delay CLI help text outputs.
# if TYPE_CHECKING allows use of imports for type annotations, without
# actually invoking the import at runtime.
if TYPE_CHECKING:
    from lightning_pose.model import Model

from . import friendly, types


def _build_parser():
    parser = friendly.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Litpose command to run.",
        parser_class=friendly.ArgumentSubParser,
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        description="Train a lightning-pose model using the specified configuration file.",
        usage="litpose train <config_file> \\\n"
        "                      [--output_dir OUTPUT_DIR] \\\n"
        "                      [--overrides KEY=VALUE...]"
        "",
    )
    train_parser.add_argument(
        "config_file",
        type=types.config_file,
        help="path a config file.\n"
        "Download and modify the config template from: \n"
        "https://github.com/paninski-lab/lightning-pose/blob/main/scripts/configs/config_default.yaml",
    )
    train_parser.add_argument(
        "--output_dir",
        type=types.model_dir,
        help="explicitly specifies the output model directory.\n"
        "If not specified, defaults to "
        "./outputs/{YYYY-MM-DD}/{HH:MM:SS}/",
    )
    train_parser.add_argument(
        "--detector_model",
        type=types.existing_model_dir,
        help="If specified, uses cropped training data in the detector model's directory.",
    )
    train_parser.add_argument(
        "--overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="overrides attributes of the config file. Uses hydra syntax:\n"
        "https://hydra.cc/docs/advanced/override_grammar/basic/",
    )

    # Add arguments specific to the 'train' command here

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        description="Predicts keypoints on videos or images.\n"
        "\n"
        "  Video predictions are saved to:\n"
        "    <model_dir>/\n"
        "    └── video_preds/\n"
        "        ├── <video_filename>.csv              (predictions)\n"
        "        ├── <video_filename>_<metric>.csv     (losses)\n"
        "        └── labeled_videos/\n"
        "            └── <video_filename>_labeled.mp4\n"
        "\n"
        "  Image predictions are saved to:\n"
        "    <model_dir>/\n"
        "    └── image_preds/\n"
        "        └── <image_dirname | csv_filename | timestamp>/\n"
        "            ├── predictions.csv\n"
        "            ├── predictions_<metric>.csv      (losses)\n"
        "            └── <image_filename>_labeled.png\n",
        usage="litpose predict <model_dir> <input_path:video|image|dir|csv>...  [OPTIONS]",
    )
    predict_parser.add_argument(
        "model_dir", type=types.existing_model_dir, help="path to a model directory"
    )

    predict_parser.add_argument(
        "input_path",
        type=Path,
        nargs="+",
        help="one or more paths. They can be video files, image files, CSV files, or directories.\n"
        "    directory: predicts over videos or images in the directory.\n"
        "               saves image outputs to `image_preds/<directory_name>`\n"
        "    video file: predicts on the video\n"
        "    image file: predicts on the image. saves outputs to `image_preds/<timestamp>`\n"
        "    CSV file: predicts on the images specified in the file.\n"
        "              uses the labels to compute pixel error.\n"
        "              saves outputs to `image_preds/<csv_file_name>`\n",
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

    # Crop command
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
        usage="litpose crop <model_dir> <input_path:video|csv>... --crop_ratio=CROP_RATIO --anchor_keypoints=x,y,z",
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

    remap_parser = subparsers.add_parser(
        "remap",
        description=dedent(
            """\
            Remaps predictions from cropped to original coordinate space.
            Requires model predictions to already have been generated using `litpose predict`.

            Remapped predictions are saved as "remapped_{preds_file}" in the same folder as preds_file.
            """
        ),
        usage="litpose remap <preds_file> <bbox_file>",
    )
    remap_parser.add_argument("preds_file", type=Path, help="path to a prediction file")
    remap_parser.add_argument("bbox_file", type=Path, help="path to a bbox file")

    return parser


def main():
    parser = _build_parser()

    # If no commands provided, display the help message.
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "train":
        _train(args)

    elif args.command == "predict":
        _predict(args)

    elif args.command == "crop":
        _crop(args)

    elif args.command == "remap":
        _remap_preds(args)


def _crop(args: argparse.Namespace):
    import lightning_pose.utils.cropzoom as cz
    from lightning_pose.model import Model

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


def _remap_preds(args: argparse.Namespace):
    import lightning_pose.utils.cropzoom as cz

    output_file = args.preds_file.with_name("remapped_" + args.preds_file.name)

    cz.generate_cropped_csv_file(
        input_csv_file=args.preds_file,
        input_bbox_file=args.bbox_file,
        output_csv_file=output_file,
        mode="add",
    )


def _train(args: argparse.Namespace):
    import hydra

    if args.output_dir:
        output_dir = args.output_dir
    else:
        now = datetime.datetime.now()
        output_dir = (
            Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H:%M:%S")
        )

    print(f"Output directory: {output_dir.absolute()}")
    if args.overrides:
        print(f"Overrides: {args.overrides}")

    with hydra.initialize_config_dir(
        version_base="1.1", config_dir=str(args.config_file.parent.absolute())
    ):
        cfg = hydra.compose(config_name=args.config_file.stem, overrides=args.overrides)

        # Delay this import because it's slow.
        from lightning_pose.model import Model
        from lightning_pose.train import train

        # TODO: Move some aspects of directory mgmt to the train function.
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maintain legacy hydra chdir until downstream no longer depends on it.

        if args.detector_model:
            # create detector model object before chdir so that relative path is resolved correctly
            detector_model = Model.from_dir(args.detector_model)
            import copy

            cfg = copy.deepcopy(cfg)
            cfg.data.data_dir = str(detector_model.cropped_data_dir())
            cfg.data.video_dir = str(detector_model.cropped_videos_dir())
            if isinstance(cfg.data.csv_file, str):
                cfg.data.csv_file = str(
                    detector_model.cropped_csv_file_path(cfg.data.csv_file)
                )
            else:
                cfg.data.csv_file = [
                    str(detector_model.cropped_csv_file_path(f))
                    for f in cfg.data.csv_file
                ]
            cfg.eval.test_videos_directory = cfg.data.video_dir

        os.chdir(output_dir)
        train(cfg)


def _predict(args: argparse.Namespace):
    # Delay this import because it's slow.
    from lightning_pose.model import Model

    model = Model.from_dir2(args.model_dir, hydra_overrides=args.overrides)
    input_paths = [Path(p) for p in args.input_path]

    for p in input_paths:
        _predict_multi_type(model, p, args.skip_viz, not args.overwrite)


def _predict_multi_type(model: Model, path: Path, skip_viz: bool, skip_existing: bool):
    if path.is_dir():
        image_files = [
            p for p in path.iterdir() if p.is_file() and p.suffix in [".png", ".jpg"]
        ]
        video_files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".mp4"]

        if len(image_files) > 0:
            raise NotImplementedError("Predicting on image dir.")

        for p in video_files:
            _predict_multi_type(model, p, skip_viz, skip_existing)
    elif path.suffix == ".mp4":
        # Check if prediction file already exists
        prediction_csv_file = model.video_preds_dir() / f"{path.stem}.csv"
        if skip_existing and prediction_csv_file.exists():
            print(f"Skipping {path} (prediction file already exists)")
            return

        model.predict_on_video_file(
            video_file=path, generate_labeled_video=(not skip_viz)
        )
    elif path.suffix == ".csv":
        # Check if prediction file already exists
        prediction_csv_file = model.image_preds_dir() / path.name / "predictions.csv"
        if skip_existing and prediction_csv_file.exists():
            print(f"Skipping {path} (prediction file already exists)")
            return

        model.predict_on_label_csv(
            csv_file=path,
            generate_labeled_images=False,  # TODO: implement visualization
        )
    elif path.suffix in [".png", ".jpg"]:
        raise NotImplementedError("Not yet implemented: predicting on image files.")
    else:
        pass


if __name__ == "__main__":
    main()

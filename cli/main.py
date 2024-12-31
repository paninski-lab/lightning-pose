from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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

    post_prediction_args = predict_parser.add_argument_group("post-prediction")
    post_prediction_args.add_argument(
        "--skip_viz",
        action="store_true",
        help="skip generating prediction-annotated images/videos",
    )
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
        from lightning_pose.train import train

        # TODO: Move some aspects of directory mgmt to the train function.
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maintain legacy hydra chdir until downstream no longer depends on it.
        os.chdir(output_dir)
        train(cfg)


def _predict(args: argparse.Namespace):
    # Delay this import because it's slow.
    from lightning_pose.model import Model

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir.absolute()}")

    model = Model.from_dir(model_dir)
    input_paths = [Path(p) for p in args.input_path]

    for p in input_paths:
        _predict_multi_type(model, p, args.skip_viz)


def _predict_multi_type(model: Model, path: Path, skip_viz: bool):
    if path.is_dir():
        image_files = [
            p for p in path.iterdir() if p.is_file() and p.suffix in [".png", ".jpg"]
        ]
        video_files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".mp4"]

        if len(image_files) > 0:
            raise NotImplementedError("Predicting on image dir.")

        for p in video_files:
            _predict_multi_type(model, p, skip_viz)
    elif path.suffix == ".mp4":
        model.predict_on_video_file(
            video_file=path, generate_labeled_video=(not skip_viz)
        )
    elif path.suffix == ".csv":
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

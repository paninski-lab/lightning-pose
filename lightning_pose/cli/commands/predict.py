"""Predict command for the lightning-pose CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .. import types

if TYPE_CHECKING:
    from lightning_pose.api.model import Model


def register_parser(subparsers):
    """Register the predict command parser."""
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
        help="one or more paths; can be video files, image files, CSV files, or directories.\n"
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


def handle(args):
    """Handle the predict command."""
    # Delay this import because it's slow.
    from lightning_pose.api.model import Model

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
        )
    elif path.suffix in [".png", ".jpg"]:
        raise NotImplementedError("Not yet implemented: predicting on image files.")
    else:
        pass

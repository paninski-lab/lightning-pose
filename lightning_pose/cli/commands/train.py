"""Train command for the lightning-pose CLI."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .. import types

if TYPE_CHECKING:
    from lightning_pose.api.model import Model  # noqa: F401
    from lightning_pose.train import train  # noqa: F401


def register_parser(subparsers):
    """Register the train command parser."""
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
        "https://github.com/paninski-lab/lightning-pose/blob/main/scripts/configs/config_default.yaml",  # noqa
    )
    train_parser.add_argument(
        "--output_dir",
        type=types.model_dir,
        help="explicitly specifies the output model directory.\n"
        "If not specified, defaults to "
        "./outputs/{YYYY-MM-DD}/{HH-MM-SS}/",
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


def handle(args):
    """Handle the train command."""
    # Import lightning_pose modules only when needed
    import hydra

    if args.output_dir:
        output_dir = args.output_dir
    else:
        now = datetime.datetime.now()
        output_dir = (
            Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        )

    print(f"Output directory: {output_dir.absolute()}")
    if args.overrides:
        print(f"Overrides: {args.overrides}")

    with hydra.initialize_config_dir(
        version_base="1.1", config_dir=str(args.config_file.parent.absolute())
    ):
        cfg = hydra.compose(config_name=args.config_file.stem, overrides=args.overrides)

        # Delay this import because it's slow.
        from lightning_pose.api.model import Model  # noqa: F811
        from lightning_pose.train import train  # noqa: F811

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

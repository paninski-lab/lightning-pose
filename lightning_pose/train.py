"""Example model training function."""

import contextlib
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from typeguard import typechecked

import lightning_pose
from lightning_pose.api.model import Model
from lightning_pose.api.model_config import ModelConfig
from lightning_pose.utils import pretty_print_cfg, pretty_print_str
from lightning_pose.utils.io import (
    find_video_files_for_views,
)
from lightning_pose.utils.mega_factory_impl import ModelComponentContainerImpl

# to ignore imports for sphinx-autoapidoc
__all__ = ["train"]


# TODO: Replace with contextlib.chdir in python 3.11.
@contextlib.contextmanager
def chdir(dir: str | Path):
    pwd = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(pwd)


@typechecked
def train(
    cfg: DictConfig, model_dir: str | Path | None = None, skip_evaluation=False
) -> Model:
    """
    Trains a model using the configuration `cfg`. Saves model to `model_dir`
    (defaults to cwd if unspecified).
    """
    # Default to cwd for backwards compatibility. Future: make model_dir required.
    model_dir = Path(model_dir or os.getcwd())
    model_dir.mkdir(parents=True, exist_ok=True)
    with chdir(model_dir):
        model = _train(cfg)
    # Comment out the above, and uncomment the below to skip
    # training and go straight to post-training analysis:
    # model = Model.from_dir(os.getcwd())

    if not skip_evaluation:
        _evaluate_on_training_dataset(model)
        _evaluate_on_training_dataset(model, ood_mode=True)
        _predict_test_videos(model)

    return model


def _absolute_csv_file(csv_file, data_dir):
    csv_file = Path(csv_file)
    if not csv_file.is_absolute():
        return Path(data_dir) / csv_file
    return csv_file


def _evaluate_on_training_dataset(model: Model, ood_mode=False):
    """Arguments:
    ood_mode: look for "_new"-suffixed versions of the training csv file"""
    if model.config.is_single_view():
        csv_file = _absolute_csv_file(
            model.config.cfg.data.csv_file, model.config.cfg.data.data_dir
        )
        if ood_mode:
            csv_file = csv_file.with_stem(csv_file.stem + "_new")
        csv_files = [csv_file]
    else:
        csv_files = []
        for csv_file in model.config.cfg.data.csv_file:
            csv_file = _absolute_csv_file(csv_file, model.config.cfg.data.data_dir)
            if ood_mode:
                csv_file = csv_file.with_stem(csv_file.stem + "_new")
            csv_files.append(csv_file)
        if model.config.cfg.data.get("camera_params_file"):
            camera_params_file = _absolute_csv_file(
                model.config.cfg.data.camera_params_file,
                model.config.cfg.data.data_dir,
            )
            if ood_mode:
                camera_params_file = camera_params_file.with_stem(
                    camera_params_file.stem + "_new"
                )
        else:
            camera_params_file = None

        # NOTE: setting bbox_files = None here is a hacky way to get the model predictions
        # to be in the cropped image space; otherwise the bbox info would lead to
        # predictions in the original image space. This can be achieved post-hoc by using
        # the CLI remap command.
        bbox_files = None

        # This is how the code would look without the hack
        # if model.config.cfg.data.get("bbox_file"):
        #     bbox_files = []
        #     for bbox_file in model.config.cfg.data.bbox_file:
        #         bbox_file = _absolute_csv_file(bbox_file, model.config.cfg.data.data_dir)
        #         if ood_mode:
        #             bbox_file = bbox_file.with_stem(bbox_file.stem + "_new")
        #         bbox_files.append(bbox_file)
        # else:
        #     bbox_files = None

    # ood mode: skip prediction when _new files don't exist.
    if ood_mode and not csv_files[0].exists():
        return

    # Print a custom message when in OOD mode.
    if ood_mode:
        pretty_print_str("Predicting OOD images...")
    else:
        pretty_print_str("Predicting train/val/test images...")

    # Run prediction and metric computation.
    if model.config.is_multi_view():
        model.predict_on_label_csv_multiview(
            csv_file_per_view=csv_files,
            bbox_file_per_view=bbox_files,
            camera_params_file=camera_params_file,
            data_dir=model.config.cfg.data.data_dir,
            compute_metrics=True,
            add_train_val_test_set=(not ood_mode),
        )
    else:
        csv_file = csv_files[0]
        model.predict_on_label_csv(
            csv_file=csv_file,
            data_dir=model.config.cfg.data.data_dir,
            compute_metrics=True,
            add_train_val_test_set=(not ood_mode),
        )

    # Copy prediction files to legacy location in model dir.
    for i, csv_file in enumerate(csv_files):
        if len(csv_files) > 1:
            view_name = model.config.cfg.data.view_names[i]
        # Copy output files to model_dir for backward-compatibility.
        # New users should look up these files in image_preds.
        for p_file in (model.image_preds_dir() / csv_file.name).glob(
            "predictions*.csv"
        ):
            metric_suffix = re.match(r"predictions(.*)\.csv", p_file.name)[1]
            out_file = "predictions"
            if len(csv_files) > 1:
                out_file += "_" + view_name
            if metric_suffix:
                out_file += metric_suffix
            if ood_mode:
                out_file += "_new"
            out_file += ".csv"
            out_file = model.model_dir / out_file

            shutil.copy(p_file, out_file)


def _predict_test_videos(model: Model):
    if model.config.cfg.eval.predict_vids_after_training:
        pretty_print_str("Predicting videos in cfg.eval.test_videos_directory...")
        # dealing with multiview
        if model.config.is_multi_view():
            for video_file_per_view in find_video_files_for_views(
                video_dir=model.config.cfg.data.video_dir,
                view_names=model.config.cfg.data.view_names,
            ):
                model.predict_on_video_file_multiview(
                    video_file_per_view=video_file_per_view,
                    compute_metrics=True,
                    generate_labeled_video=model.config.cfg.eval.save_vids_after_training,
                )
        else:
            for video_file in model.config.test_video_files():
                pretty_print_str(f"Predicting video: {video_file}...")

                model.predict_on_video_file(
                    Path(video_file),
                    generate_labeled_video=model.config.cfg.eval.save_vids_after_training,
                )


def _train(cfg: DictConfig) -> Model:
    # reset all seeds
    seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # record lightning-pose version
    with open_dict(cfg):
        cfg.model.lightning_pose_version = lightning_pose.version

    print("Config file:")
    pretty_print_cfg(cfg)

    ModelConfig(cfg).validate()

    container = ModelComponentContainerImpl(cfg)

    # ----------------------------------------------------------------------------------
    # Save configuration in output directory
    # ----------------------------------------------------------------------------------
    # Done before training; files will exist even if script dies prematurely.
    hydra_output_directory = os.getcwd()
    print(f"Hydra output directory: {hydra_output_directory}")

    # save config file
    dest_config_file = Path(hydra_output_directory) / "config.yaml"
    OmegaConf.save(config=cfg, f=dest_config_file, resolve=False)

    # save labeled data file(s)
    if isinstance(cfg.data.csv_file, str):
        # single view
        csv_files = [cfg.data.csv_file]
    else:
        # multi view
        assert isinstance(cfg.data.csv_file, ListConfig)
        csv_files = cfg.data.csv_file
    for csv_file in csv_files:
        src_csv_file = Path(csv_file)
        if not src_csv_file.is_absolute():
            src_csv_file = Path(cfg.data.data_dir) / src_csv_file

        dest_csv_file = Path(hydra_output_directory) / src_csv_file.name
        shutil.copyfile(src_csv_file, dest_csv_file)

    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    # logger
    logger = container.get_logger()
    # Log hydra config to tensorboard as helpful metadata.
    for key, value in cfg.items():
        logger.experiment.add_text(
            "hydra_config_%s" % key, "```\n%s```" % OmegaConf.to_yaml(value)
        )

    # early stopping, learning rate monitoring, model checkpointing, backbone unfreezing

    # train model!
    trainer = container.get_trainer()
    trainer.fit(model=container.get_model(), datamodule=container.get_data_module())

    # When devices > 0, lightning creates a process per device.
    # Kill processes other than the main process, otherwise they all go forward.
    if not trainer.is_global_zero:
        sys.exit(0)

    return Model.from_dir(hydra_output_directory)

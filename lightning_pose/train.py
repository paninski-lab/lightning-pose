"""Example model training function."""

import contextlib
import math
import os
import random
import re
import shutil
import sys
import warnings
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from typeguard import typechecked

import lightning_pose
from lightning_pose.api.model import Model
from lightning_pose.api.model_config import ModelConfig
from lightning_pose.utils import pretty_print_cfg, pretty_print_str
from lightning_pose.utils.io import return_absolute_data_paths
from lightning_pose.utils.scripts import (
    calculate_steps_per_epoch,
    get_callbacks,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
)

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

    # ood mode: skip prediction when _new files don't exist.
    if ood_mode and not csv_files[0].exists():
        return

    # Print a custom message when in OOD mode.
    if ood_mode:
        pretty_print_str("Predicting OOD images...")
    else:
        pretty_print_str("Predicting train/val/test images...")

    for i, csv_file in enumerate(csv_files):
        model.predict_on_label_csv(
            csv_file=csv_file,
            data_dir=model.config.cfg.data.data_dir,
            compute_metrics=True,
            generate_labeled_images=False,
            add_train_val_test_set=(not ood_mode),
        )

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
        pretty_print_str(f"Predicting videos in cfg.eval.test_videos_directory...")
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

    print("Our Hydra config file:")
    pretty_print_cfg(cfg)

    ModelConfig(cfg).validate()

    # path handling for toy data
    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)

    # ----------------------------------------------------------------------------------
    # Set up data/model objects
    # ----------------------------------------------------------------------------------

    # imgaug transform
    imgaug_transform = get_imgaug_transform(cfg=cfg)

    # dataset
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)

    # datamodule; breaks up dataset into train/val/test
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

    steps_per_epoch = calculate_steps_per_epoch(data_module)

    # Convert milestone_steps to milestones if applicable (before `get_model`).
    if (
        "multisteplr" in cfg.training.lr_scheduler_params
        and "milestone_steps" in cfg.training.lr_scheduler_params.multisteplr
    ):
        milestone_steps = cfg.training.lr_scheduler_params.multisteplr.milestone_steps
        milestones = [math.ceil(s / steps_per_epoch) for s in milestone_steps]
        cfg.training.lr_scheduler_params.multisteplr.milestones = milestones

    # model
    model = get_model(cfg=cfg, data_module=data_module, loss_factories=loss_factories)

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
            src_csv_file = Path(data_dir) / src_csv_file

        dest_csv_file = Path(hydra_output_directory) / src_csv_file.name
        shutil.copyfile(src_csv_file, dest_csv_file)

    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    # logger
    logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)
    # Log hydra config to tensorboard as helpful metadata.
    for key, value in cfg.items():
        logger.experiment.add_text(
            "hydra_config_%s" % key, "```\n%s```" % OmegaConf.to_yaml(value)
        )

    # early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
    callbacks = get_callbacks(
        cfg,
        early_stopping=cfg.training.get("early_stopping", False),
        lr_monitor=True,
        ckpt_every_n_epochs=cfg.training.get("ckpt_every_n_epochs", None),
    )

    # set up trainer

    # Old configs may have num_gpus: 0. We will remove support in a future release.
    if cfg.training.num_gpus == 0:
        warnings.warn(
            "Config contains unsupported value num_gpus: 0. "
            "Update num_gpus to 1 in your config."
        )
    cfg.training.num_gpus = max(cfg.training.num_gpus, 1)

    # Initialize to Trainer defaults. Note max_steps defaults to -1.
    min_steps, max_steps, min_epochs, max_epochs = (None, -1, None, None)
    if "min_steps" in cfg.training:
        min_steps = cfg.training.min_steps
        max_steps = cfg.training.max_steps
    else:
        min_epochs = cfg.training.min_epochs
        max_epochs = cfg.training.max_epochs

    # Initialize to Trainer defaults. Note max_steps defaults to -1.

    # Unlike min_epoch/min_step, both of these are valid to specify.
    check_val_every_n_epoch = cfg.training.get(
        "check_val_every_n_epoch", 1
    )  # 1 is default for Trainer.
    val_check_interval = cfg.training.get("val_check_interval")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.training.num_gpus,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        max_steps=max_steps,
        min_steps=min_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        val_check_interval=val_check_interval,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        # To understand why we set this, see 'max_size_cycle' in UnlabeledDataModule.
        limit_train_batches=cfg.training.get("limit_train_batches") or steps_per_epoch,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        profiler=cfg.training.get("profiler", None),
        sync_batchnorm=True,
    )

    # train model!
    trainer.fit(model=model, datamodule=data_module)

    # When devices > 0, lightning creates a process per device.
    # Kill processes other than the main process, otherwise they all go forward.
    if not trainer.is_global_zero:
        sys.exit(0)

    return Model.from_dir(hydra_output_directory)

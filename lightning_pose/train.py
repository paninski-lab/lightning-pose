"""Example model training function."""

import os
import random
import shutil
import sys
import warnings
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from typeguard import typechecked

from lightning_pose.model import Model
from lightning_pose.utils import pretty_print_cfg, pretty_print_str
from lightning_pose.utils.io import return_absolute_data_paths
from lightning_pose.utils.scripts import (
    calculate_train_batches,
    get_callbacks,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
)

# to ignore imports for sphinx-autoapidoc
__all__ = ["train"]


@typechecked
def train(cfg: DictConfig) -> Model:
    """
    Trains a model using the configuration `cfg`. Saves model to current
    working directory (callers should `chdir` to the desired `model_dir` prior to calling).
    """
    model = _train(cfg)
    # Comment out the above, and uncomment the below to skip
    # training and go straight to post-training analysis:
    # import os
    # model = Model.from_dir(os.getcwd())

    _evaluate_on_training_dataset(model)
    _evaluate_on_ood_dataset(model)
    _predict_test_videos(model)

    return model


def _absolute_csv_file(csv_file, data_dir):
    csv_file = Path(csv_file)
    if not csv_file.is_absolute():
        return Path(data_dir) / csv_file
    return csv_file


def _evaluate_on_training_dataset(model: Model):
    pretty_print_str("Predicting train/val/test images...")

    if model.config.is_single_view():
        csv_file = _absolute_csv_file(
            model.config.cfg.data.csv_file, model.config.cfg.data.data_dir
        )
        csv_files = [csv_file]
        output_filename_stems = ["predictions"]
    else:
        csv_files = []
        output_filename_stems = []
        for csv_file, view_name in zip(
            model.config.cfg.data.csv_file, model.config.cfg.data.view_names
        ):
            csv_files.append(
                _absolute_csv_file(csv_file, model.config.cfg.data.data_dir)
            )
            output_filename_stems.append(f"predictions_{view_name}")

    for csv_file, output_filename_stem in zip(csv_files, output_filename_stems):
        model.predict_on_label_csv_internal(
            csv_file=csv_file,
            data_dir=model.config.cfg.data.data_dir,
            # TODO annotate with train/val/test split metadata.
            compute_metrics=True,
            generate_labeled_images=False,
            output_dir=model.model_dir,
            output_filename_stem=output_filename_stem,
            add_train_val_test_set=True,
        )


def _evaluate_on_ood_dataset(model: Model):
    if model.config.is_single_view():
        csv_file = _absolute_csv_file(
            model.config.cfg.data.csv_file, model.config.cfg.data.data_dir
        )
        ood_csv_file = csv_file.with_stem(csv_file.stem + "_new")
        ood_csv_files = [ood_csv_file]
        output_filename_stems = ["predictions_new"]
    else:
        ood_csv_files = []
        output_filename_stems = []
        for csv_file, view_name in zip(
            model.config.cfg.data.csv_file, model.config.cfg.data.view_names
        ):
            csv_file = _absolute_csv_file(csv_file, model.config.cfg.data.data_dir)
            ood_csv_file = csv_file.with_stem(csv_file.stem + "_new")
            ood_csv_files.append(ood_csv_file)
            output_filename_stems.append(f"predictions_new_{view_name}")

    if ood_csv_files[0].is_file():
        pretty_print_str("Predicting OOD images...")

        for ood_csv_file, output_filename_stem in zip(
            ood_csv_files, output_filename_stems
        ):
            model.predict_on_label_csv_internal(
                csv_file=ood_csv_file,
                data_dir=model.config.cfg.data.data_dir,
                compute_metrics=True,
                generate_labeled_images=False,
                output_dir=model.model_dir,
                output_filename_stem=output_filename_stem,
            )


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
    from lightning_pose import __version__ as lightning_pose_version

    with open_dict(cfg):
        cfg.model.lightning_pose_version = lightning_pose_version

    print("Our Hydra config file:")
    pretty_print_cfg(cfg)

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

    # calculate number of batches for both labeled and unlabeled data per epoch
    limit_train_batches = calculate_train_batches(cfg, dataset)

    # set up trainer

    # Old configs may have num_gpus: 0. We will remove support in a future release.
    if cfg.training.num_gpus == 0:
        warnings.warn(
            "Config contains unsupported value num_gpus: 0. "
            "Update num_gpus to 1 in your config."
        )
    cfg.training.num_gpus = max(cfg.training.num_gpus, 1)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.training.num_gpus,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        check_val_every_n_epoch=min(
            cfg.training.check_val_every_n_epoch,
            cfg.training.max_epochs,  # for debugging or otherwise training for a short time
        ),
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        limit_train_batches=limit_train_batches,
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

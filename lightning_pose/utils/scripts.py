"""Helper functions to build pipeline components from config dictionary."""

import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import copy
from typing import TYPE_CHECKING, Callable
import imgaug.augmenters as iaa
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ValidationError
from typeguard import typechecked

from lightning_pose.callbacks import AnnealWeight, PatchMasking, UnfreezeBackbone
from lightning_pose.data.augmentations import (
    expand_imgaug_str_to_dict,
    imgaug_transform,
)

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import (
    BaseTrackingDataset,
    HeatmapDataset,
    MultiviewHeatmapDataset,
)
from lightning_pose.data.datatypes import ComputeMetricsSingleResult
from lightning_pose.data.utils import (
    compute_num_train_frames,
    split_sizes_from_probabilities,
)
from lightning_pose.losses.factory import LossFactory
from lightning_pose.metrics import (
    pca_multiview_reprojection_error,
    pca_singleview_reprojection_error,
    pixel_error,
    temporal_norm,
)
from lightning_pose.models.base import (
    _apply_defaults_for_lr_scheduler_params,
    _apply_defaults_for_optimizer_params,
)
from lightning_pose.utils import io as io_utils
from lightning_pose.utils.pca import KeypointPCA
from torch.utils.data import Subset, random_split, DataLoader

# to ignore imports for sphix-autoapidoc
__all__ = [
    "get_imgaug_transform",
    "get_dataset",
    "get_data_module",
    "get_loss_factories",
    "get_model",
    "get_callbacks",
    "calculate_steps_per_epoch",
    "compute_metrics",
    "get_split_datasets",
]


@typechecked
def get_imgaug_transform(cfg: DictConfig) -> iaa.Sequential:
    """Create simple and flexible data transform pipeline that augments images and keypoints.

    Args:
        cfg: standard config file that carries around dataset info; relevant is the parameter
            - "cfg.training.imgaug" which can take on the following values:
                - default/none: resizing only
                - dlc: imgaug pipeline implemented in DLC 2.0 package
                (rotation, motion blur, dropout, salt/pepper noise, elastic transform,
                histogram equalization, emboss, crop)
                - dlc-lr: `dlc` pipeline plus 0° or 180° rotation (left-right flipping)
                - dlc-top-down: `dlc` pipeline plus 0°, 90°, 180°, or 270° rotation
                - dlc-mv: multiview-compatible `dlc` pipeline (excludes 2D geometric transforms
                 like rotation, elastic transform, and crop that would break 3D consistency)
                - dict/DictConfig: custom augmentation parameters where each key is
                an imgaug transform name and value contains probability, args, and kwargs.
            - "cfg.training.imgaug_3d":
                boolean flag to control 3D-compatible augmentations for multiview models;
                set to False to disable automatic "dlc-mv" enforcement;
                set to True to enable 3D augmentations for when camera params file exist.

    Returns:
        imgaug pipeline

    """

    params = cfg.training.get("imgaug", "default")
    if isinstance(params, str):
        # Check if user explicitly wants to use 3D augmentations for multiview models
        imagug_3d = cfg.training.get("imgaug_3d", None)

        # enforce "dlc-mv" imgaug pipeline for multiview models (no 2D geometric transforms)
        # only if explicitly requested or if no preference is set and camera params exist
        if (
            params not in ["default", "none"]
            and cfg.model.model_type.find("multiview") > -1
            and cfg.data.get("camera_params_file")
            and (imagug_3d is True or imagug_3d is None)
        ):
            params = "dlc-mv"
        params_dict = expand_imgaug_str_to_dict(params)
    elif isinstance(params, dict) or isinstance(params, DictConfig):
        if isinstance(params, DictConfig):
            # recursively convert Dict/ListConfigs to dicts/lists
            params_dict = OmegaConf.to_object(params)
        else:
            params_dict = params.copy()
        for transform, val in params_dict.items():
            assert getattr(
                iaa, transform
            ), f"{transform} is not a valid imgaug transform"
    else:
        raise TypeError(
            f"params is of type {type(params)}, must be str, dict, or DictConfig"
        )

    return imgaug_transform(params_dict)


@typechecked
def get_dataset(
    cfg: DictConfig,
    data_dir: str,
    imgaug_transform: iaa.Sequential,
) -> BaseTrackingDataset | HeatmapDataset | MultiviewHeatmapDataset:
    """Create a dataset that contains labeled data."""

    if cfg.model.model_type == "regression":
        if cfg.data.get("view_names", None) and len(cfg.data.view_names) > 1:
            raise NotImplementedError(
                "Multi-view support only available for heatmap-based models"
            )
        else:
            dataset = BaseTrackingDataset(
                root_directory=data_dir,
                csv_path=cfg.data.csv_file,
                image_resize_height=cfg.data.image_resize_dims.height,
                image_resize_width=cfg.data.image_resize_dims.width,
                imgaug_transform=imgaug_transform,
                do_context=False,  # no context for regression models
            )
    elif cfg.model.model_type.find("heatmap") > -1:
        if cfg.data.get("view_names", None) and len(cfg.data.view_names) > 1:
            UserWarning(
                "No precautions regarding the size of the images were considered here, "
                "images will be resized accordingly to configs!"
            )
            if cfg.training.imgaug in ["default", "none"] or not cfg.data.get(
                "camera_params_file"
            ):
                # we are either
                # 1. running inference on un-augmented data, and need to make sure to resize
                # 2. using a multiview model w/o camera params, and need to take care of resizing
                resize = True
            else:
                resize = False
            dataset = MultiviewHeatmapDataset(
                root_directory=data_dir,
                csv_paths=cfg.data.csv_file,
                view_names=list(cfg.data.view_names),
                image_resize_height=cfg.data.image_resize_dims.height,
                image_resize_width=cfg.data.image_resize_dims.width,
                imgaug_transform=imgaug_transform,
                downsample_factor=cfg.data.get("downsample_factor", 2),
                do_context=cfg.model.model_type
                == "heatmap_mhcrnn",  # context only for mhcrnn
                resize=resize,
                uniform_heatmaps=cfg.training.get(
                    "uniform_heatmaps_for_nan_keypoints", False
                ),
                camera_params_path=cfg.data.get("camera_params_file", None),
                bbox_paths=cfg.data.get("bbox_file", None),
            )
        else:
            dataset = HeatmapDataset(
                root_directory=data_dir,
                csv_path=cfg.data.csv_file,
                image_resize_height=cfg.data.image_resize_dims.height,
                image_resize_width=cfg.data.image_resize_dims.width,
                imgaug_transform=imgaug_transform,
                downsample_factor=cfg.data.get("downsample_factor", 2),
                do_context=cfg.model.model_type
                == "heatmap_mhcrnn",  # context only for mhcrnn
                uniform_heatmaps=cfg.training.get(
                    "uniform_heatmaps_for_nan_keypoints", False
                ),
            )

    else:
        raise NotImplementedError(
            "%s is an invalid cfg.model.model_type" % cfg.model.model_type
        )

    return dataset


def get_train_val_batches(cfg: DictConfig) -> tuple[int, int]:
    """Determine the number of batches to use for training and validation."""
    # Divide config batch_size by num_gpus to maintain the same effective batch
    # size in a multi-gpu setting.
    train_batch_size = int(
        np.ceil(cfg.training.train_batch_size / cfg.training.num_gpus)
    )
    val_batch_size = int(np.ceil(cfg.training.val_batch_size / cfg.training.num_gpus))

    return train_batch_size, val_batch_size


@typechecked
def get_data_module(
    cfg: DictConfig,
    dataset: BaseTrackingDataset | HeatmapDataset | MultiviewHeatmapDataset,
    video_dir: str | None = None,
    dataloader_factory: Callable[[str], DataLoader] | None = None,
) -> BaseDataModule | UnlabeledDataModule:
    """Create a data module using provided dataloader factory (preferred).

    If `dataloader_factory` is None, this function will derive splits and create a
    default labeled dataloader factory from cfg for backward compatibility.
    """

    # Old configs may have num_gpus: 0. We will remove support in a future release.
    if cfg.training.num_gpus == 0:
        warnings.warn(
            "Config contains unsupported value num_gpus: 0. "
            "Update num_gpus to 1 in your config."
        )
    cfg.training.num_gpus = max(cfg.training.num_gpus, 1)

    semi_supervised = io_utils.check_if_semi_supervised(cfg.model.losses_to_use)

    # Build splits and default factory if not provided
    splits = get_split_datasets(cfg=cfg, dataset=dataset)
    if dataloader_factory is None:
        dataloader_factory = get_dataloader_factory(
            cfg=cfg, dataset=dataset, splits=splits
        )

    if not semi_supervised:
        data_module = BaseDataModule(
            dataset=dataset,
            splits=splits,
            dataloader_factory=dataloader_factory,
        )
    else:
        # Divide config batch_size by num_gpus to maintain the same effective batch
        # size in a multi-gpu setting.
        base_sequence_length = int(
            np.ceil(cfg.dali.base.train.sequence_length / cfg.training.num_gpus)
        )
        # Maintain effective context batch size in num_gpus adjustment,
        # otherwise the effective context batch size will be too small due to the
        # 2 context frames on each side of center.
        _effective_context_batch_size = max(cfg.dali.context.train.batch_size - 4, 0)
        # Each GPU should get the effective batch size / num_gpus, + 4 for context frames.
        context_batch_size = int(
            np.ceil(_effective_context_batch_size / cfg.training.num_gpus + 4)
        )

        if cfg.model.model_type == "heatmap_mhcrnn" and context_batch_size < 5:
            raise ValidationError(
                "dali.context.train.batch_size must be >= 5 * num_gpus for "
                "semi-supervised context models. "
                "Found {cfg.dali.context.train.batch_size}"
            )

        dali_config = OmegaConf.merge(
            cfg.dali,
            {
                "base": {"train": {"sequence_length": base_sequence_length}},
                "context": {"train": {"batch_size": context_batch_size}},
            },
        )

        view_names = cfg.data.get("view_names", None)
        view_names = list(view_names) if view_names is not None else None
        data_module = UnlabeledDataModule(
            dataset=dataset,
            splits=splits,
            dataloader_factory=dataloader_factory,
            video_paths_list=video_dir,
            view_names=view_names,
            dali_config=dali_config,
            imgaug=cfg.training.get("imgaug", "default"),
        )
    return data_module


@typechecked
def get_loss_factories(
    cfg: DictConfig,
    data_module: BaseDataModule | UnlabeledDataModule,
) -> dict:
    """Create loss factory that orchestrates different losses during training."""

    cfg_loss_dict = OmegaConf.to_object(cfg.losses)

    loss_params_dict = {"supervised": {}, "unsupervised": {}}

    # collect all supervised losses in a dict; no extra params needed
    # set "log_weight = 0.0" so that weight = 1 and effective weight is (1 / 2)
    if cfg.model.model_type.find("heatmap") > -1:
        loss_name = "heatmap_" + cfg.model.heatmap_loss_type
        loss_params_dict["supervised"][loss_name] = {"log_weight": 0.0}
        if cfg.model.model_type.find("multiview") > -1 and cfg.data.get(
            "camera_params_file"
        ):
            log_weight = cfg.losses.get("supervised_pairwise_projections", {}).get(
                "log_weight"
            )
            if log_weight is not None:
                print("adding supervised pairwise projection loss")
                loss_params_dict["supervised"]["supervised_pairwise_projections"] = {
                    "log_weight": log_weight
                }
    else:
        loss_params_dict["supervised"][cfg.model.model_type] = {"log_weight": 0.0}

    # collect all unsupervised losses and their params in a dict
    if cfg.model.losses_to_use is not None:
        for loss_name in cfg.model.losses_to_use:
            # general parameters
            loss_params_dict["unsupervised"][loss_name] = cfg_loss_dict[loss_name]
            loss_params_dict["unsupervised"][loss_name]["loss_name"] = loss_name
            # loss-specific parameters
            if loss_name[:8] == "unimodal" or loss_name[:15] == "temporal_heatmap":
                if cfg.model.model_type == "regression":
                    raise NotImplementedError(
                        "unimodal loss can only be used with classes inheriting from "
                        "HeatmapTracker. \nYou specified a RegressionTracker model."
                    )
                # record original image dims (after initial resizing)
                raise Exception(
                    "need to update unimodal and/or temporal heatmap loss to not use "
                    "cfg.data.image_resize_dims, which has been deprecated."
                )
                height_og = cfg.data.image_resize_dims.height
                width_og = cfg.data.image_resize_dims.width
                loss_params_dict["unsupervised"][loss_name][
                    "original_image_height"
                ] = height_og
                loss_params_dict["unsupervised"][loss_name][
                    "original_image_width"
                ] = width_og
                # record downsampled image dims
                height_ds = int(
                    height_og // (2 ** cfg.data.get("downsample_factor", 2))
                )
                width_ds = int(width_og // (2 ** cfg.data.get("downsample_factor", 2)))
                loss_params_dict["unsupervised"][loss_name][
                    "downsampled_image_height"
                ] = height_ds
                loss_params_dict["unsupervised"][loss_name][
                    "downsampled_image_width"
                ] = width_ds
                if loss_name[:8] == "unimodal":
                    loss_params_dict["unsupervised"][loss_name]["uniform_heatmaps"] = (
                        cfg.training.get("uniform_heatmaps_for_nan_keypoints", False)
                    )
            elif loss_name == "pca_multiview":
                if cfg.data.get("view_names", None) and len(cfg.data.view_names) > 1:
                    # assume user has provided a set of columns that are present in each view
                    num_keypoints = cfg.data.num_keypoints
                    num_views = len(cfg.data.view_names)
                    if isinstance(cfg.data.mirrored_column_matches[0], int):
                        loss_params_dict["unsupervised"][loss_name][
                            "mirrored_column_matches"
                        ] = [
                            (
                                v * num_keypoints
                                + np.array(cfg.data.mirrored_column_matches, dtype=int)
                            ).tolist()
                            for v in range(num_views)
                        ]
                    else:
                        # allow user to force specific mapping in multiview case
                        loss_params_dict["unsupervised"][loss_name][
                            "mirrored_column_matches"
                        ] = cfg.data.mirrored_column_matches
                else:
                    # user must provide all matching columns
                    loss_params_dict["unsupervised"][loss_name][
                        "mirrored_column_matches"
                    ] = cfg.data.mirrored_column_matches
            elif loss_name == "pca_singleview":
                if cfg.data.get("view_names", None) and len(cfg.data.view_names) > 1:
                    raise NotImplementedError(
                        "The Pose PCA loss is currently not implemented for multiview data."
                    )
                else:
                    loss_params_dict["unsupervised"][loss_name][
                        "columns_for_singleview_pca"
                    ] = cfg.data.get("columns_for_singleview_pca", None)

    # build supervised loss factory, which orchestrates all supervised losses
    loss_factory_sup = LossFactory(
        losses_params_dict=loss_params_dict["supervised"],
        data_module=data_module,
    )
    # build unsupervised loss factory, which orchestrates all unsupervised losses
    loss_factory_unsup = LossFactory(
        losses_params_dict=loss_params_dict["unsupervised"],
        data_module=data_module,
    )

    return {"supervised": loss_factory_sup, "unsupervised": loss_factory_unsup}


@typechecked
def get_model(
    cfg: DictConfig,
    data_module: BaseDataModule | UnlabeledDataModule | None,
    loss_factories: dict[str, LossFactory] | dict[str, None],
) -> pl.LightningModule:
    """Create model: regression or heatmap based, supervised or semi-supervised."""

    ## BEGIN: Hack for model training only (not needed for inference)
    steps_per_epoch = calculate_steps_per_epoch(data_module)

    # convert milestone_steps to milestones if applicable (before `get_model`).
    if (
        "multisteplr" in cfg.training.lr_scheduler_params
        and "milestone_steps" in cfg.training.lr_scheduler_params.multisteplr
    ):
        milestone_steps = cfg.training.lr_scheduler_params.multisteplr.milestone_steps
        milestones = [math.ceil(s / steps_per_epoch) for s in milestone_steps]
        cfg.training.lr_scheduler_params.multisteplr.milestones = milestones

    # convert patch masking epochs if applicable (before `get_callbacks`)
    if "patch_mask" in cfg.training and "init_epoch" in cfg.training.patch_mask:
        init_step = math.ceil(cfg.training.patch_mask.init_epoch * steps_per_epoch)
        final_step = math.ceil(cfg.training.patch_mask.final_epoch * steps_per_epoch)
        with open_dict(cfg):
            cfg.training.patch_mask.init_step = init_step
            cfg.training.patch_mask.final_step = final_step
    ## END: Hack for model training only (not needed for inference)

    optimizer = cfg.training.get("optimizer", "Adam")
    optimizer_params = _apply_defaults_for_optimizer_params(
        optimizer,
        cfg.training.get("optimizer_params"),
    )

    lr_scheduler = cfg.training.get("lr_scheduler", "multisteplr")
    lr_scheduler_params = _apply_defaults_for_lr_scheduler_params(
        lr_scheduler, cfg.training.get("lr_scheduler_params", {}).get(f"{lr_scheduler}")
    )

    semi_supervised = io_utils.check_if_semi_supervised(cfg.model.losses_to_use)
    image_h = cfg.data.image_resize_dims.height
    image_w = cfg.data.image_resize_dims.width
    if "vit" in cfg.model.backbone:
        if image_h != image_w:
            raise RuntimeError(
                "ViT model requires resized height and width to be equal"
            )

    backbone_pretrained = cfg.model.get("backbone_pretrained", True)
    if not semi_supervised:
        if cfg.model.model_type == "regression":
            from lightning_pose.models import RegressionTracker

            model = RegressionTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
            )
        elif cfg.model.model_type == "heatmap":
            if data_module:
                num_targets = data_module.dataset.num_targets
            else:
                num_targets = None
            from lightning_pose.models import HeatmapTracker

            model = HeatmapTracker(
                num_keypoints=cfg.data.num_keypoints,
                num_targets=num_targets,
                loss_factory=loss_factories["supervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.get("downsample_factor", 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
                backbone_checkpoint=cfg.model.get(
                    "backbone_checkpoint"
                ),  # only used by ViTMAE
            )
        elif cfg.model.model_type == "heatmap_mhcrnn":
            from lightning_pose.models import HeatmapTrackerMHCRNN

            model = HeatmapTrackerMHCRNN(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.get("downsample_factor", 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
                backbone_checkpoint=cfg.model.get(
                    "backbone_checkpoint"
                ),  # only used by ViTMAE
            )
        elif cfg.model.model_type == "heatmap_multiview_transformer":
            from lightning_pose.models import HeatmapTrackerMultiviewTransformer

            model = HeatmapTrackerMultiviewTransformer(
                num_keypoints=cfg.data.num_keypoints,
                num_views=len(cfg.data.view_names),
                loss_factory=loss_factories["supervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                head=cfg.model.get("head", "heatmap_cnn"),
                downsample_factor=cfg.data.get("downsample_factor", 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
                backbone_checkpoint=cfg.model.get("backbone_checkpoint"),  # only used by ViTMAE
            )
        else:
            raise NotImplementedError(
                f"{cfg.model.model_type} is an invalid cfg.model.model_type for a fully "
                f"supervised model"
            )

    else:
        if cfg.model.model_type == "regression":
            from lightning_pose.models import SemiSupervisedRegressionTracker

            model = SemiSupervisedRegressionTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                loss_factory_unsupervised=loss_factories["unsupervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
            )

        elif cfg.model.model_type == "heatmap":
            from lightning_pose.models import SemiSupervisedHeatmapTracker

            model = SemiSupervisedHeatmapTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                loss_factory_unsupervised=loss_factories["unsupervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.get("downsample_factor", 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
                backbone_checkpoint=cfg.model.get(
                    "backbone_checkpoint"
                ),  # only used by ViTMAE
            )
        elif cfg.model.model_type == "heatmap_mhcrnn":
            from lightning_pose.models import SemiSupervisedHeatmapTrackerMHCRNN

            model = SemiSupervisedHeatmapTrackerMHCRNN(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                loss_factory_unsupervised=loss_factories["unsupervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.get("downsample_factor", 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
                backbone_checkpoint=cfg.model.get(
                    "backbone_checkpoint"
                ),  # only used by ViTMAE
            )
        elif cfg.model.model_type == "heatmap_multiview_transformer":
            from lightning_pose.models import (
                SemiSupervisedHeatmapTrackerMultiviewTransformer,
            )

            model = SemiSupervisedHeatmapTrackerMultiviewTransformer(
                num_keypoints=cfg.data.num_keypoints,
                num_views=len(cfg.data.view_names),
                loss_factory=loss_factories["supervised"],
                loss_factory_unsupervised=loss_factories["unsupervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                head=cfg.model.get("head", "heatmap_cnn"),
                downsample_factor=cfg.data.get("downsample_factor", 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
                patch_mask_config=cfg.training.get("patch_mask", {}),
            )
        else:
            raise NotImplementedError(
                f"{cfg.model.model_type} invalid cfg.model.model_type for a semi-supervised model"
            )

    # load weights from user-provided checkpoint path
    if cfg.model.get("checkpoint", None):
        ckpt = cfg.model.checkpoint
        print(f"Loading weights from {ckpt}")
        if not ckpt.endswith(".ckpt"):
            import glob

            ckpt = glob.glob(os.path.join(ckpt, "**", "*.ckpt"), recursive=True)[0]
        # Try loading with default settings first, fallback to weights_only=False if needed
        try:
            state_dict = torch.load(ckpt)["state_dict"]
        except Exception as e:
            print(f"Warning: Failed to load checkpoint with default settings: {e}")
            print("Attempting to load with weights_only=False...")
            state_dict = torch.load(ckpt, weights_only=False)["state_dict"]
        # try loading all weights
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            # assume heads don't match; just load backbone weights
            new_state_dict = OrderedDict()
            for key, val in state_dict.items():
                if "backbone" in key:
                    new_state_dict[key] = val
            model.load_state_dict(new_state_dict, strict=False)

    return model


def get_training_logger(cfg):
    return pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)


@typechecked
def get_callbacks(
    cfg: DictConfig,
    checkpointing=True,
    backbone_unfreeze=True,
) -> list:
    # Param extraction from train.py. May be overridden for testing.
    early_stopping = cfg.training.get("early_stopping", False)
    lr_monitor = True
    ckpt_every_n_epochs = cfg.training.get("ckpt_every_n_epochs", None)

    callbacks = []

    if early_stopping:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_supervised_loss",
            patience=cfg.training.early_stop_patience,
            mode="min",
        )
        callbacks.append(early_stopping)

    if backbone_unfreeze:
        unfreeze_step = cfg.training.get("unfreezing_step")
        unfreeze_epoch = cfg.training.get("unfreezing_epoch")
        unfreeze_backbone_callback = UnfreezeBackbone(
            unfreeze_step=unfreeze_step, unfreeze_epoch=unfreeze_epoch
        )
        callbacks.append(unfreeze_backbone_callback)

    if lr_monitor:
        # this callback should be added after UnfreezeBackbone in order to log its learning rate
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    # always save out best model
    if checkpointing:
        ckpt_best_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val_supervised_loss",
            mode="min",
            filename="{epoch}-{step}-best",
        )
        callbacks.append(ckpt_best_callback)

    if ckpt_every_n_epochs:
        # if ckpt_every_n_epochs is not None, save separate checkpoint files
        ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor=None,
            every_n_epochs=ckpt_every_n_epochs,
            save_top_k=-1,
        )
        callbacks.append(ckpt_callback)

    # we just need this callback for unsupervised losses or multiview models with 3d loss
    if (
        (cfg.model.losses_to_use != []) and (cfg.model.losses_to_use is not None)
    ) or cfg.losses.get("supervised_pairwise_projections", {}).get(
        "log_weight"
    ) is not None:
        anneal_weight_callback = AnnealWeight(**cfg.callbacks.anneal_weight)
        callbacks.append(anneal_weight_callback)

    # add patch masking callback for multiview transformer models if patch masking is enabled
    if (
        cfg.model.model_type == "heatmap_multiview_transformer"
        and cfg.training.get("patch_mask", {}).get("final_ratio", 0.0) > 0.0
    ):

        patch_masking_callback = PatchMasking(
            patch_mask_config=cfg.training.get("patch_mask", {}),
            patch_seed=cfg.training.rng_seed_model_pt,
        )
        callbacks.append(patch_masking_callback)

    return callbacks


def calculate_steps_per_epoch(data_module: BaseDataModule):
    """Infer steps per epoch from the training dataloader.

    For semi-supervised (CombinedLoader), we still enforce a minimum of 10 steps
    to encourage more unlabeled exposure when labeled data is scarce.
    """
    train_loader = data_module.train_dataloader()
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        # Fallback: compute from dataset length and an inferred batch size where possible
        train_dataset_length = len(data_module.train_dataset)
        # Try to get batch_size attribute if available
        batch_size = getattr(train_loader, "batch_size", None)
        if not batch_size or batch_size == 0:
            # conservative default to avoid division by zero
            batch_size = 1
        steps_per_epoch = math.ceil(train_dataset_length / batch_size)

    is_unsupervised = isinstance(data_module, UnlabeledDataModule)

    # To understand why we do this, see 'max_size_cycle' in UnlabeledDataModule.
    if is_unsupervised:
        steps_per_epoch = max(10, steps_per_epoch)
    return steps_per_epoch


@typechecked
def compute_metrics(
    cfg: DictConfig,
    preds_file: str | list[str] | Path | list[Path],
    data_module: BaseDataModule | UnlabeledDataModule | None = None,
) -> None:
    """Compute various metrics on predictions csv file, potentially for multiple views.
    Saves metrics to files next to predictions file, in the convention of:
        {prediction_file_stem}_{metric_name}.csv

    Args:
        cfg: the config used to determine whether single or multiview and which metrics
            to compute
        preds_file: Path to model predictions used to compute metrics.
            For multiview, a list of prediction files corresponding to the csv_files.
        data_module: for computing PCA metrics

    """
    if not isinstance(cfg.data.csv_file, str):
        assert isinstance(preds_file, list)
        assert len(preds_file) == len(cfg.data.csv_file)
        for csv_file, preds_file_ in zip(cfg.data.csv_file, preds_file):
            labels_file = Path(csv_file)
            if not labels_file.is_absolute():
                labels_file = Path(cfg.data.data_dir) / labels_file
            labels_file = io_utils.return_absolute_path(str(labels_file))
            compute_metrics_single(
                cfg=cfg,
                labels_file=labels_file,
                preds_file=preds_file_,
                data_module=data_module,
            )
    else:
        assert isinstance(cfg.data.csv_file, str)
        labels_file = Path(cfg.data.csv_file)
        if not labels_file.is_absolute():
            labels_file = Path(cfg.data.data_dir) / labels_file
        labels_file = io_utils.return_absolute_path(str(labels_file))
        compute_metrics_single(
            cfg=cfg,
            labels_file=labels_file,
            preds_file=preds_file,
            data_module=data_module,
        )


@typechecked
def compute_metrics_single(
    cfg: DictConfig,
    labels_file: str | Path | None,
    preds_file: str | Path,
    data_module: BaseDataModule | UnlabeledDataModule | None = None,
) -> ComputeMetricsSingleResult:
    """Compute various metrics on a predictions csv file from a single view."""
    # load predictions
    pred_df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)
    keypoint_names = io_utils.get_keypoint_names(
        cfg, csv_file=str(preds_file), header_rows=[0, 1, 2]
    )
    xyl_mask = pred_df.columns.get_level_values("coords").isin(["x", "y", "likelihood"])
    tmp = pred_df.loc[:, xyl_mask].to_numpy().reshape(pred_df.shape[0], -1, 3)

    index = pred_df.index
    if pred_df.keys()[-1][0] == "set":
        # these are predictions on labeled data
        # get rid of last column that contains info about train/val/test set
        is_video = False
        set = pred_df.iloc[:, -1].to_numpy()
    else:
        # these are predictions on video data
        is_video = True
        set = None

    keypoints_pred = tmp[:, :, :2]  # shape (samples, n_keypoints, 2)
    # confidences = tmp[:, :, -1]  # shape (samples, n_keypoints)

    # hard-code metrics for now
    if is_video:
        metrics_to_compute = ["temporal"]
    else:  # labeled data
        assert labels_file is not None
        metrics_to_compute = ["pixel_error"]
    # for either labeled and unlabeled data, if a pca loss is specified in config, we compute the
    # associated metric

    if (
        data_module is not None
        and cfg.data.get("columns_for_singleview_pca", None) is not None
        and len(cfg.data.columns_for_singleview_pca) != 0
        and not isinstance(
            data_module.dataset, MultiviewHeatmapDataset
        )  # mirrored-only for now
    ):
        metrics_to_compute += ["pca_singleview"]
    if (
        data_module is not None
        and cfg.data.get("mirrored_column_matches", None) is not None
        and len(cfg.data.mirrored_column_matches) != 0
        and not isinstance(
            data_module.dataset, MultiviewHeatmapDataset
        )  # mirrored-only for now
    ):
        metrics_to_compute += ["pca_multiview"]

    result = ComputeMetricsSingleResult()
    preds_file_path = Path(preds_file)
    # compute metrics; csv files will be saved to the same directory the prdictions are stored in
    if "pixel_error" in metrics_to_compute:
        # Read labeled data
        labels_df = pd.read_csv(labels_file, header=[0, 1, 2], index_col=0)
        labels_df = io_utils.fix_empty_first_row(labels_df)
        assert labels_df.index.equals(index)

        keypoints_true = labels_df.to_numpy().reshape(labels_df.shape[0], -1, 2)
        error_per_keypoint = pixel_error(keypoints_true, keypoints_pred)
        error_df = pd.DataFrame(error_per_keypoint, index=index, columns=keypoint_names)
        # add train/val/test split
        if set is not None:
            error_df["set"] = set

        save_file = preds_file_path.with_name(preds_file_path.stem + "_pixel_error.csv")
        error_df.to_csv(save_file)
        result.pixel_error_df = error_df

    if "temporal" in metrics_to_compute:
        temporal_norm_per_keypoint = temporal_norm(keypoints_pred)
        temporal_norm_df = pd.DataFrame(
            temporal_norm_per_keypoint, index=index, columns=keypoint_names
        )
        # add train/val/test split
        if set is not None:
            temporal_norm_df["set"] = set
        save_file = preds_file_path.with_name(
            preds_file_path.stem + "_temporal_norm.csv"
        )
        temporal_norm_df.to_csv(save_file)
        result.temporal_norm_df = temporal_norm_df

    if "pca_singleview" in metrics_to_compute:
        try:
            # build pca object
            pca = KeypointPCA(
                loss_type="pca_singleview",
                data_module=data_module,
                components_to_keep=cfg.losses.pca_singleview.components_to_keep,
                empirical_epsilon_percentile=cfg.losses.pca_singleview.get(
                    "empirical_epsilon_percentile", 1.0
                ),
                columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
                centering_method=cfg.losses.pca_singleview.get(
                    "centering_method", None
                ),
            )
            # re-fit pca on the labeled data to get params
            pca()
            # compute reprojection error
            pcasv_error_per_keypoint = pca_singleview_reprojection_error(
                keypoints_pred, pca
            )
            pcasv_df = pd.DataFrame(
                pcasv_error_per_keypoint, index=index, columns=keypoint_names
            )
            # add train/val/test split
            if set is not None:
                pcasv_df["set"] = set
            save_file = preds_file_path.with_name(
                preds_file_path.stem + "_pca_singleview_error.csv"
            )
            pcasv_df.to_csv(save_file)
            result.pca_sv_df = pcasv_df

        except ValueError as e:
            # PCA will fail if not enough train frames.
            # skip pca metric in this case.
            # re-raise if this is not the PCA error this try is intended to swallow
            if "cannot fit PCA" not in str(e):
                raise e

    if "pca_multiview" in metrics_to_compute:
        # build pca object
        pca = KeypointPCA(
            loss_type="pca_multiview",
            data_module=data_module,
            components_to_keep=cfg.losses.pca_singleview.components_to_keep,
            empirical_epsilon_percentile=cfg.losses.pca_singleview.get(
                "empirical_epsilon_percentile", 1.0
            ),
            mirrored_column_matches=cfg.data.mirrored_column_matches,
        )
        # re-fit pca on the labeled data to get params
        pca()
        # compute reprojection error
        pcamv_error_per_keypoint = pca_multiview_reprojection_error(keypoints_pred, pca)
        pcamv_df = pd.DataFrame(
            pcamv_error_per_keypoint, index=index, columns=keypoint_names
        )
        # add train/val/test split
        if set is not None:
            pcamv_df["set"] = set
        save_file = preds_file_path.with_name(
            preds_file_path.stem + "_pca_multiview_error.csv"
        )
        pcamv_df.to_csv(save_file)
        result.pca_mv_df = pcamv_df

    return result


@typechecked
def get_split_datasets(
    cfg: DictConfig,
    dataset: torch.utils.data.Dataset,
) -> tuple[Subset, Subset, Subset]:
    """Split a dataset into train/val/test subsets with augmentation-aware handling.

    This mirrors the logic previously implemented in BaseDataModule._setup.

    Args:
        cfg: Full config; split-related parameters are read from `cfg.training`:
            - `train_prob`, optional `val_prob`, optional `test_prob`
            - `train_frames` (int or float)
            - `rng_seed_data_pt` (int)
        dataset: The full dataset to split.

    Returns:
        Tuple of (train_subset, val_subset, test_subset).
    """
    datalen = len(dataset)
    print(f"Number of labeled images in the full dataset (train+val+test): {datalen}")

    # derive split parameters from cfg
    train_probability = cfg.training.get("train_prob", 0.8)
    val_probability = cfg.training.get("val_prob", None)
    test_probability = cfg.training.get("test_prob", None)
    train_frames = cfg.training.get("train_frames", None)
    torch_seed = cfg.training.get("rng_seed_data_pt", 42)

    # split data based on provided probabilities
    data_splits_list = split_sizes_from_probabilities(
        datalen,
        train_probability=train_probability,
        val_probability=val_probability,
        test_probability=test_probability,
    )

    if (
        getattr(dataset, "imgaug_transform", None) is not None
        and len(dataset.imgaug_transform) == 1
    ):
        # no augmentations in the pipeline; subsets can share same underlying dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            data_splits_list,
            generator=torch.Generator().manual_seed(torch_seed),
        )
    else:
        # augmentations in the pipeline; we want validation and test datasets that only resize
        # we can't simply change the imgaug pipeline in the datasets after they've been split
        # because the subsets actually point to the same underlying dataset, so we create
        # separate datasets here
        train_idxs, val_idxs, test_idxs = random_split(
            range(len(dataset)),
            data_splits_list,
            generator=torch.Generator().manual_seed(torch_seed),
        )

        train_dataset = Subset(copy.deepcopy(dataset), indices=list(train_idxs))
        val_dataset = Subset(copy.deepcopy(dataset), indices=list(val_idxs))
        test_dataset = Subset(copy.deepcopy(dataset), indices=list(test_idxs))

        # only use the final resize transform for the validation and test datasets
        # try to pull the final transform; if unavailable (e.g., multiview that doesn't resize
        # by default), enforce resizing to dataset.height/width
        if (
            getattr(dataset, "imgaug_transform", None) is not None
            and len(dataset.imgaug_transform) > 0
            and dataset.imgaug_transform[-1].__str__().find("Resize") == 0
        ):
            final_transform = iaa.Sequential([dataset.imgaug_transform[-1]])
        else:
            height = getattr(dataset, "height", None)
            width = getattr(dataset, "width", None)
            if height is None or width is None:
                raise AttributeError(
                    "Dataset must have 'height' and 'width' attributes when no final Resize transform is present."
                )
            final_transform = iaa.Sequential(
                [iaa.Resize({"height": height, "width": width})]
            )

        val_dataset.dataset.imgaug_transform = final_transform
        if hasattr(val_dataset.dataset, "dataset"):
            # this will get triggered for multiview datasets
            print("val: updating children datasets with resize imgaug pipeline")
            for _, dset in val_dataset.dataset.dataset.items():
                dset.imgaug_transform = final_transform

        test_dataset.dataset.imgaug_transform = final_transform
        if hasattr(test_dataset.dataset, "dataset"):
            # this will get triggered for multiview datasets
            print("test: updating children datasets with resize imgaug pipeline")
            for _, dset in test_dataset.dataset.dataset.items():
                dset.imgaug_transform = final_transform

    # further subsample training data if desired
    if train_frames is not None:
        n_frames = compute_num_train_frames(len(train_dataset), train_frames)
        if n_frames < len(train_dataset):
            # reflect further subsampling from train_frames
            train_dataset.indices = train_dataset.indices[:n_frames]

    print(
        f"Dataset splits -- "
        f"train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, "
        f"test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloader_factory(
    cfg: DictConfig,
    dataset: torch.utils.data.Dataset,
    splits: tuple[Subset, Subset, Subset],
) -> Callable[[str], DataLoader]:
    """Returns stage -> dataloader for labeled data (train/val/test/full)."""
    train_batch_size, val_batch_size = get_train_val_batches(cfg)
    train_dataset, val_dataset, test_dataset = splits

    def get_dataloader(stage: str) -> DataLoader:
        num_workers = cfg.training.get("num_workers")
        if num_workers is None:
            slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
            if slurm_cpus:
                num_workers = int(slurm_cpus)
            else:
                # Fallback to os.cpu_count()
                num_workers = os.cpu_count()
        if stage == "train":
            return DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
                shuffle=True,
                generator=torch.Generator().manual_seed(cfg.training.rng_seed_data_pt),
            )
        if stage == "val":
            return DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
        if stage == "test":
            return DataLoader(
                test_dataset,
                batch_size=cfg.training.test_batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
        if stage == "full":
            return DataLoader(
                dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
            )
        raise NotImplementedError(f"Unknown stage: {stage}")

    return get_dataloader


def get_training_trainer(
    cfg: DictConfig, logger, callbacks, steps_per_epoch
) -> pl.Trainer:
    """Get trainer for training."""
    # set up trainer

    cfg.training.num_gpus = max(cfg.training.num_gpus, 1)

    # initialize to Trainer defaults. Note max_steps defaults to -1.
    min_steps, max_steps, min_epochs, max_epochs = (None, -1, None, None)
    if "min_steps" in cfg.training:
        min_steps = cfg.training.min_steps
        max_steps = cfg.training.max_steps
    else:
        min_epochs = cfg.training.min_epochs
        max_epochs = cfg.training.max_epochs

    # Unlike min_epoch/min_step, both of these are valid to specify.
    check_val_every_n_epoch = cfg.training.get("check_val_every_n_epoch", 1)
    val_check_interval = cfg.training.get("val_check_interval")

    return pl.Trainer(
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

"""Helper functions to build pipeline components from config dictionary."""

import os
import warnings
from collections import OrderedDict
from pathlib import Path

import imgaug.augmenters as iaa
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import ValidationError

from lightning_pose.callbacks import (
    AnnealWeight,
    JSONTrainingProgressTracker,
    PatchMasking,
    UnfreezeBackbone,
)
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
from lightning_pose.losses.factory import LossFactory
from lightning_pose.models import ALLOWED_MODELS, check_if_semi_supervised
from lightning_pose.models.base import (
    _apply_defaults_for_lr_scheduler_params,
    _apply_defaults_for_optimizer_params,
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "get_imgaug_transform",
    "get_dataset",
    "get_data_module",
    "get_model",
    "get_callbacks",
]


def get_imgaug_transform(cfg: DictConfig | ListConfig) -> iaa.Sequential:
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
            assert isinstance(params_dict, dict)
        else:
            params_dict = params.copy()
        for transform, _val in params_dict.items():
            assert getattr(iaa, str(transform)), f"{transform} is not a valid imgaug transform"
    else:
        raise TypeError(f"params is of type {type(params)}, must be str, dict, or DictConfig")

    return imgaug_transform(params_dict)  # type: ignore[arg-type]


def get_dataset(
    cfg: DictConfig | ListConfig,
    data_dir: str,
    imgaug_transform: iaa.Sequential,
) -> BaseTrackingDataset | HeatmapDataset | MultiviewHeatmapDataset:
    """Create a dataset that contains labeled data."""

    if cfg.model.model_type == "regression":
        if cfg.data.get("view_names", None) and len(cfg.data.view_names) > 1:
            raise NotImplementedError("Multi-view support only available for heatmap-based models")
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
            if (
                cfg.training.imgaug in ["default", "none"]
                or not cfg.data.get("camera_params_file")
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
                do_context=cfg.model.model_type == "heatmap_mhcrnn",  # context only for mhcrnn
                resize=resize,
                uniform_heatmaps=cfg.training.get("uniform_heatmaps_for_nan_keypoints", False),
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
                do_context=cfg.model.model_type == "heatmap_mhcrnn",  # context only for mhcrnn
                uniform_heatmaps=cfg.training.get("uniform_heatmaps_for_nan_keypoints", False),
            )

    else:
        raise NotImplementedError(f"{cfg.model.model_type} is an invalid cfg.model.model_type")

    return dataset


def get_data_module(
    cfg: DictConfig | ListConfig,
    dataset: BaseTrackingDataset | HeatmapDataset | MultiviewHeatmapDataset,
    video_dir: str | None = None,
) -> BaseDataModule | UnlabeledDataModule:
    """Create a data module that splits a dataset into train/val/test iterators."""

    # Old configs may have num_gpus: 0. We will remove support in a future release.
    if cfg.training.num_gpus == 0:
        warnings.warn(
            "Config contains unsupported value num_gpus: 0. "
            "Update num_gpus to 1 in your config.",
            stacklevel=2,
        )
    cfg.training.num_gpus = max(cfg.training.num_gpus, 1)

    # Divide config batch_size by num_gpus to maintain the same effective batch
    # size in a multi-gpu setting.
    train_batch_size = int(
        np.ceil(cfg.training.train_batch_size / cfg.training.num_gpus)
    )
    val_batch_size = int(np.ceil(cfg.training.val_batch_size / cfg.training.num_gpus))

    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    if not semi_supervised:
        data_module = BaseDataModule(
            dataset=dataset,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.get("num_workers"),
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
            torch_seed=cfg.training.rng_seed_data_pt,
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

        assert video_dir is not None, 'video_dir must be provided for semi-supervised training'
        view_names = cfg.data.get("view_names", None)
        view_names = list(view_names) if view_names is not None else None
        data_module = UnlabeledDataModule(
            dataset=dataset,
            video_paths_list=video_dir,
            view_names=view_names,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.get("num_workers"),
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
            dali_config=dali_config,
            torch_seed=cfg.training.rng_seed_data_pt,
            imgaug=cfg.training.get("imgaug", "default"),
        )
    return data_module


def get_model(
    cfg: DictConfig | ListConfig,
    data_module: BaseDataModule | UnlabeledDataModule | None,
    loss_factories: dict[str, LossFactory] | dict[str, None],
) -> ALLOWED_MODELS:
    """Create model: regression or heatmap based, supervised or semi-supervised."""

    optimizer = cfg.training.get("optimizer", "Adam")
    optimizer_params = _apply_defaults_for_optimizer_params(
        optimizer,
        cfg.training.get("optimizer_params"),
    )

    lr_scheduler = cfg.training.get("lr_scheduler", "multisteplr")
    lr_scheduler_params = _apply_defaults_for_lr_scheduler_params(
        lr_scheduler,
        cfg.training.get("lr_scheduler_params", {}).get(f"{lr_scheduler}")
    )

    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    image_h = cfg.data.image_resize_dims.height
    image_w = cfg.data.image_resize_dims.width
    if "vit" in cfg.model.backbone:
        if image_h != image_w:
            raise RuntimeError("ViT model requires resized height and width to be equal")

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
                backbone_checkpoint=cfg.model.get("backbone_checkpoint"),  # only used by ViTMAE
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
                backbone_checkpoint=cfg.model.get("backbone_checkpoint"),  # only used by ViTMAE
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
                backbone_checkpoint=cfg.model.get("backbone_checkpoint"),  # only used by ViTMAE
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
                backbone_checkpoint=cfg.model.get("backbone_checkpoint"),  # only used by ViTMAE
            )
        elif cfg.model.model_type == "heatmap_multiview_transformer":
            from lightning_pose.models import SemiSupervisedHeatmapTrackerMultiviewTransformer
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


def get_callbacks(
    cfg: DictConfig | ListConfig,
    early_stopping: bool = False,
    checkpointing: bool = True,
    lr_monitor: bool = True,
    ckpt_every_n_epochs: int | None = None,
    backbone_unfreeze: bool = True,
    status_file: Path | None = None,
) -> list:
    """Build and return the list of training callbacks based on the config.

    Args:
        cfg: hydra config containing training and callback parameters.
        early_stopping: if True, add an ``EarlyStopping`` callback.
        checkpointing: if True, add a ``ModelCheckpoint`` callback that saves the best model.
        lr_monitor: if True, add a ``LearningRateMonitor`` callback.
        ckpt_every_n_epochs: if not None, also save a checkpoint every this many epochs.
        backbone_unfreeze: if True, add the ``UnfreezeBackbone`` callback.
        status_file: if not None, add a ``JSONTrainingProgressTracker`` callback writing to this
            path.

    Returns:
        List of callback objects ready to pass to a ``pl.Trainer``.
    """
    callbacks = []

    if early_stopping:
        early_stopping_cb = EarlyStopping(
            monitor="val_supervised_loss",
            patience=cfg.training.early_stop_patience,
            mode="min",
        )
        callbacks.append(early_stopping_cb)

    if backbone_unfreeze:
        unfreeze_step = cfg.training.get("unfreezing_step")
        unfreeze_epoch = cfg.training.get("unfreezing_epoch")
        unfreeze_backbone_callback = UnfreezeBackbone(
            unfreeze_step=unfreeze_step, unfreeze_epoch=unfreeze_epoch
        )
        callbacks.append(unfreeze_backbone_callback)

    if lr_monitor:
        # this callback should be added after UnfreezeBackbone in order to log its learning rate
        lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor_cb)

    # always save out best model
    if checkpointing:
        ckpt_best_callback = ModelCheckpoint(
            monitor="val_supervised_loss",
            mode="min",
            filename="{epoch}-{step}-best",
        )
        callbacks.append(ckpt_best_callback)

    if ckpt_every_n_epochs:
        # if ckpt_every_n_epochs is not None, save separate checkpoint files
        ckpt_callback = ModelCheckpoint(
            monitor=None,
            every_n_epochs=ckpt_every_n_epochs,
            save_top_k=-1,
        )
        callbacks.append(ckpt_callback)

    # we need this callback for both supervised and unsupervised losses
    has_supervised_loss = any(
        loss_config.get("log_weight") is not None
        for loss_name, loss_config in cfg.losses.items() if loss_name.startswith("supervised_")
    )
    if (
        ((cfg.model.losses_to_use != []) and (cfg.model.losses_to_use is not None))
        or has_supervised_loss
    ):
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

    if status_file is not None:
        callbacks.append(JSONTrainingProgressTracker(status_file))
    return callbacks

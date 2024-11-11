"""Helper functions to build pipeline components from config dictionary."""

import os
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import imgaug.augmenters as iaa
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ValidationError
from typeguard import typechecked

from lightning_pose.callbacks import AnnealWeight, UnfreezeBackbone
from lightning_pose.data.augmentations import imgaug_transform
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import (
    BaseTrackingDataset,
    HeatmapDataset,
    MultiviewHeatmapDataset,
)
from lightning_pose.data.utils import compute_num_train_frames, split_sizes_from_probabilities
from lightning_pose.losses.factory import LossFactory
from lightning_pose.metrics import (
    pca_multiview_reprojection_error,
    pca_singleview_reprojection_error,
    pixel_error,
    temporal_norm,
)
from lightning_pose.models import (
    ALLOWED_MODELS,
    HeatmapTracker,
    HeatmapTrackerMHCRNN,
    RegressionTracker,
    SemiSupervisedHeatmapTracker,
    SemiSupervisedHeatmapTrackerMHCRNN,
    SemiSupervisedRegressionTracker,
)
from lightning_pose.utils.io import (
    check_if_semi_supervised,
    get_keypoint_names,
    return_absolute_path,
)
from lightning_pose.utils.pca import KeypointPCA
from lightning_pose.utils.predictions import create_labeled_video, predict_single_video

# to ignore imports for sphix-autoapidoc
__all__ = [
    "get_imgaug_transform",
    "get_dataset",
    "get_data_module",
    "get_loss_factories",
    "get_model",
    "get_callbacks",
    "calculate_train_batches",
    "compute_metrics",
    "export_predictions_and_labeled_video",
]


@typechecked
def get_imgaug_transform(cfg: DictConfig) -> iaa.Sequential:
    """Create simple data transform pipeline that augments images.

    Args:
        cfg: standard config file that carries around dataset info; relevant is the parameter
            "cfg.training.imgaug" which can take on the following values:
            - default: resizing only
            - dlc: imgaug pipeline implemented in DLC 2.0 package
            - dlc-top-down: `dlc` pipeline plus random flipping along both horizontal and vertical
                axes

    """
    return imgaug_transform(cfg)


@typechecked
def get_dataset(
    cfg: DictConfig,
    data_dir: str,
    imgaug_transform: iaa.Sequential,
) -> Union[BaseTrackingDataset, HeatmapDataset, MultiviewHeatmapDataset]:
    """Create a dataset that contains labeled data."""

    if cfg.model.model_type == "regression":
        if cfg.data.get("view_names", None) and len(cfg.data.view_names) > 1:
            raise NotImplementedError("Multi-view support only available for heatmap-based models")
        else:
            dataset = BaseTrackingDataset(
                root_directory=data_dir,
                csv_path=cfg.data.csv_file,
                imgaug_transform=imgaug_transform,
                do_context=False,  # no context for regression models
            )
    elif cfg.model.model_type == "heatmap" or cfg.model.model_type == "heatmap_mhcrnn":
        if cfg.data.get("view_names", None) and len(cfg.data.view_names) > 1:
            UserWarning(
                "No precautions regarding the size of the images were considered here, "
                "images will be resized accordingly to configs!"
            )
            dataset = MultiviewHeatmapDataset(
                root_directory=data_dir,
                csv_paths=cfg.data.csv_file,
                view_names=list(cfg.data.view_names),
                downsample_factor=cfg.data.downsample_factor,
                imgaug_transform=imgaug_transform,
                uniform_heatmaps=cfg.training.get("uniform_heatmaps_for_nan_keypoints", False),
                do_context=cfg.model.model_type == "heatmap_mhcrnn",  # context only for mhcrnn
            )
        else:
            dataset = HeatmapDataset(
                root_directory=data_dir,
                csv_path=cfg.data.csv_file,
                imgaug_transform=imgaug_transform,
                downsample_factor=cfg.data.downsample_factor,
                do_context=cfg.model.model_type == "heatmap_mhcrnn",  # context only for mhcrnn
                uniform_heatmaps=cfg.training.get("uniform_heatmaps_for_nan_keypoints", False),
            )

    else:
        raise NotImplementedError("%s is an invalid cfg.model.model_type" % cfg.model.model_type)

    return dataset


@typechecked
def get_data_module(
    cfg: DictConfig,
    dataset: Union[BaseTrackingDataset, HeatmapDataset, MultiviewHeatmapDataset],
    video_dir: Optional[str] = None,
) -> Union[BaseDataModule, UnlabeledDataModule]:
    """Create a data module that splits a dataset into train/val/test iterators."""

    # Old configs may have num_gpus: 0. We will remove support in a future release.
    if cfg.training.num_gpus == 0:
        warnings.warn(
            "Config contains unsupported value num_gpus: 0. "
            "Update num_gpus to 1 in your config."
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
            num_workers=cfg.training.num_workers,
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
                f"dali.context.train.batch_size must be >= 5 * num_gpus for "
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
            video_paths_list=video_dir,
            view_names=view_names,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.num_workers,
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
            dali_config=dali_config,
            torch_seed=cfg.training.rng_seed_data_pt,
            imgaug=cfg.training.get("imgaug", "default"),
        )
    return data_module


@typechecked
def get_loss_factories(
    cfg: DictConfig,
    data_module: Union[BaseDataModule, UnlabeledDataModule],
) -> dict:
    """Create loss factory that orchestrates different losses during training."""

    cfg_loss_dict = OmegaConf.to_object(cfg.losses)

    loss_params_dict = {"supervised": {}, "unsupervised": {}}

    # collect all supervised losses in a dict; no extra params needed
    # set "log_weight = 0.0" so that weight = 1 and effective weight is (1 / 2)
    if cfg.model.model_type == "heatmap" or cfg.model.model_type == "heatmap_mhcrnn":
        loss_name = "heatmap_" + cfg.model.heatmap_loss_type
        loss_params_dict["supervised"][loss_name] = {"log_weight": 0.0}
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
                height_ds = int(height_og // (2 ** cfg.data.downsample_factor))
                width_ds = int(width_og // (2 ** cfg.data.downsample_factor))
                loss_params_dict["unsupervised"][loss_name][
                    "downsampled_image_height"
                ] = height_ds
                loss_params_dict["unsupervised"][loss_name][
                    "downsampled_image_width"
                ] = width_ds
                if loss_name[:8] == "unimodal":
                    loss_params_dict["unsupervised"][loss_name][
                        "uniform_heatmaps"
                    ] = cfg.training.get("uniform_heatmaps_for_nan_keypoints", False)
            elif loss_name == "pca_multiview":
                if cfg.data.get("view_names", None) and len(cfg.data.view_names) > 1:
                    # assume user has provided a set of columns that are present in each view
                    num_keypoints = cfg.data.num_keypoints
                    num_views = len(cfg.data.view_names)
                    if isinstance(cfg.data.mirrored_column_matches[0], int):
                        loss_params_dict["unsupervised"][loss_name][
                            "mirrored_column_matches"
                        ] = [
                            (v * num_keypoints
                             + np.array(cfg.data.mirrored_column_matches, dtype=int)).tolist()
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
                    ] = cfg.data.get('columns_for_singleview_pca', None)

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
    data_module: Union[BaseDataModule, UnlabeledDataModule],
    loss_factories: Dict[str, LossFactory],
) -> pl.LightningModule:
    """Create model: regression or heatmap based, supervised or semi-supervised."""

    lr_scheduler = cfg.training.lr_scheduler

    lr_scheduler_params = OmegaConf.to_object(
        cfg.training.lr_scheduler_params[lr_scheduler]
    )
    lr_scheduler_params["unfreeze_backbone_at_epoch"] = cfg.training.unfreezing_epoch

    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    image_h = cfg.data.image_resize_dims.height
    image_w = cfg.data.image_resize_dims.width
    if "vit" in cfg.model.backbone:
        if image_h != image_w:
            raise RuntimeError("ViT model requires resized height and width to be equal")

    backbone_pretrained = cfg.model.get("backbone_pretrained", True)
    if not semi_supervised:
        if cfg.model.model_type == "regression":
            model = RegressionTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
            )
        elif cfg.model.model_type == "heatmap":

            model = HeatmapTracker(
                num_keypoints=cfg.data.num_keypoints,
                num_targets=data_module.dataset.num_targets,
                loss_factory=loss_factories["supervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=data_module.dataset.output_shape,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
            )
        elif cfg.model.model_type == "heatmap_mhcrnn":
            model = HeatmapTrackerMHCRNN(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=data_module.dataset.output_shape,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
            )
        else:
            raise NotImplementedError(
                f"{cfg.model.model_type} is an invalid cfg.model.model_type for a fully "
                f"supervised model"
            )

    else:
        if cfg.model.model_type == "regression":
            model = SemiSupervisedRegressionTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                loss_factory_unsupervised=loss_factories["unsupervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
            )

        elif cfg.model.model_type == "heatmap":
            model = SemiSupervisedHeatmapTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                loss_factory_unsupervised=loss_factories["unsupervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=data_module.dataset.output_shape,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
            )
        elif cfg.model.model_type == "heatmap_mhcrnn":
            model = SemiSupervisedHeatmapTrackerMHCRNN(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                loss_factory_unsupervised=loss_factories["unsupervised"],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=data_module.dataset.output_shape,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
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
        state_dict = torch.load(ckpt)["state_dict"]
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


@typechecked
def get_callbacks(
    cfg: DictConfig,
    early_stopping=False,
    lr_monitor=True,
    ckpt_every_n_epochs=None,
    backbone_unfreeze=True,
) -> List:

    callbacks = []

    if early_stopping:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_supervised_loss",
            patience=cfg.training.early_stop_patience,
            mode="min",
        )
        callbacks.append(early_stopping)

    if backbone_unfreeze:
        unfreeze_backbone_callback = UnfreezeBackbone(cfg.training.unfreezing_epoch)
        callbacks.append(unfreeze_backbone_callback)

    if lr_monitor:
        # this callback should be added after UnfreezeBackbone in order to log its learning rate
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    # always save out best model
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

    # we just need this callback for unsupervised models
    if (cfg.model.losses_to_use != []) and (cfg.model.losses_to_use is not None):
        anneal_weight_callback = AnnealWeight(**cfg.callbacks.anneal_weight)
        callbacks.append(anneal_weight_callback)

    return callbacks


@typechecked
def calculate_train_batches(
    cfg: DictConfig,
    dataset: Optional[Union[BaseTrackingDataset, HeatmapDataset, MultiviewHeatmapDataset]] = None,
) -> int:
    """
    For semi-supervised models, this tells us how many batches to take from each dataloader
    (labeled and unlabeled) during a given epoch.
    The default set here is to exhaust all batches from the labeled data loader, often leaving
    many video frames untouched.
    But the unlabeled data loader will be randomly reset for the next epoch.
    We also enforce a minimum value of 10 so that models with a small number of labeled frames will
    cycle through the dataset multiple times per epoch, which we have found to be useful
    empirically.

    """
    if cfg.training.get("limit_train_batches", None) is None:
        # NOTE: small bit of redundant code from datamodule
        datalen = dataset.__len__()
        data_splits_list = split_sizes_from_probabilities(
            datalen,
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
        )
        num_train_frames = compute_num_train_frames(
            data_splits_list[0], cfg.training.get("train_frames", None)
        )
        # For multi-GPU, the computation is unchanged.
        #   num_train_frames is divided by num_gpus to get num_train_frames per gpu
        #   train_batch_size is also divided by num_gpus to get the mini-batch size
        #   so num_gpus cancels out of the numerator and denominator.
        num_labeled_batches = int(np.ceil(num_train_frames / cfg.training.train_batch_size))
        limit_train_batches = np.max([num_labeled_batches, 10])  # 10 is minimum
    else:
        limit_train_batches = cfg.training.limit_train_batches

    return int(limit_train_batches)


@typechecked
def compute_metrics(
    cfg: DictConfig,
    preds_file: Union[str, List[str]],
    data_module: Optional[Union[BaseDataModule, UnlabeledDataModule]] = None,
) -> None:
    """Compute various metrics on predictions csv file, potentially for multiple views."""
    if (
        cfg.data.get("view_names", None)
        and len(cfg.data.view_names) > 1
        and isinstance(preds_file, list)
    ):
        preds_file = sorted(preds_file)
        for view_name, csv_file, preds_file_ in zip(
                sorted(cfg.data.view_names),
                sorted(cfg.data.csv_file),
                preds_file
        ):
            assert view_name in preds_file_
            labels_file = return_absolute_path(os.path.join(cfg.data.data_dir, csv_file))
            # preds_file_ = preds_file.replace(".csv", f"_{view_name}.csv")
            compute_metrics_single(
                cfg=cfg,
                labels_file=labels_file,
                preds_file=preds_file_,
                data_module=data_module,
            )
    else:
        if isinstance(cfg.data.csv_file, str):
            labels_file = return_absolute_path(
                os.path.join(cfg.data.data_dir, cfg.data.csv_file)
            )
        else:
            labels_file = return_absolute_path(
                os.path.join(cfg.data.data_dir, cfg.data.csv_file[0])
            )
        compute_metrics_single(
            cfg=cfg, labels_file=labels_file, preds_file=preds_file, data_module=data_module,
        )


@typechecked
def compute_metrics_single(
    cfg: DictConfig,
    labels_file: str,
    preds_file: str,
    data_module: Optional[Union[BaseDataModule, UnlabeledDataModule]] = None,
) -> None:
    """Compute various metrics on a predictions csv file from a single view."""

    # get keypoint names
    labels_df = pd.read_csv(labels_file, header=[0, 1, 2], index_col=0)
    keypoint_names = get_keypoint_names(
        cfg, csv_file=labels_file, header_rows=[0, 1, 2])
    # load predictions
    pred_df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)
    xyl_mask = pred_df.columns.get_level_values("coords").isin(["x", "y", "likelihood"])
    tmp = pred_df.loc[:, xyl_mask].to_numpy().reshape(pred_df.shape[0], -1, 3)

    if pred_df.keys()[-1][0] == "set":
        # these are predictions on labeled data
        # get rid of last column that contains info about train/val/test set
        is_video = False
        index = labels_df.index
        set = pred_df.iloc[:, -1].to_numpy()
    else:
        # these are predictions on video data
        is_video = True
        index = pred_df.index
        set = None

    keypoints_pred = tmp[:, :, :2]  # shape (samples, n_keypoints, 2)
    # confidences = tmp[:, :, -1]  # shape (samples, n_keypoints)

    # hard-code metrics for now
    if is_video:
        metrics_to_compute = ["temporal"]
    else:  # labeled data
        metrics_to_compute = ["pixel_error"]
    # for either labeled and unlabeled data, if a pca loss is specified in config, we compute the
    # associated metric
    if (
        data_module is not None
        and cfg.data.get("columns_for_singleview_pca", None) is not None
        and len(cfg.data.columns_for_singleview_pca) != 0
        and not isinstance(data_module.dataset, MultiviewHeatmapDataset)  # mirrored-only for now
    ):
        metrics_to_compute += ["pca_singleview"]
    if (
        data_module is not None
        and cfg.data.get("mirrored_column_matches", None) is not None
        and len(cfg.data.mirrored_column_matches) != 0
        and not isinstance(data_module.dataset, MultiviewHeatmapDataset)  # mirrored-only for now
    ):
        metrics_to_compute += ["pca_multiview"]

    # compute metrics; csv files will be saved to the same directory the prdictions are stored in
    if "pixel_error" in metrics_to_compute:
        keypoints_true = labels_df.to_numpy().reshape(labels_df.shape[0], -1, 2)
        error_per_keypoint = pixel_error(keypoints_true, keypoints_pred)
        error_df = pd.DataFrame(error_per_keypoint, index=index, columns=keypoint_names)
        # add train/val/test split
        if set is not None:
            error_df["set"] = set
        save_file = preds_file.replace(".csv", "_pixel_error.csv")
        error_df.to_csv(save_file)

    if "temporal" in metrics_to_compute:
        temporal_norm_per_keypoint = temporal_norm(keypoints_pred)
        temporal_norm_df = pd.DataFrame(
            temporal_norm_per_keypoint, index=index, columns=keypoint_names
        )
        # add train/val/test split
        if set is not None:
            temporal_norm_df["set"] = set
        save_file = preds_file.replace(".csv", "_temporal_norm.csv")
        temporal_norm_df.to_csv(save_file)

    if "pca_singleview" in metrics_to_compute:
        # build pca object
        pca = KeypointPCA(
            loss_type="pca_singleview",
            data_module=data_module,
            components_to_keep=cfg.losses.pca_singleview.components_to_keep,
            empirical_epsilon_percentile=cfg.losses.pca_singleview.get(
                "empirical_epsilon_percentile", 1.0),
            columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
            centering_method=cfg.losses.pca_singleview.get("centering_method", None),
        )
        # re-fit pca on the labeled data to get params
        pca()
        # compute reprojection error
        pcasv_error_per_keypoint = pca_singleview_reprojection_error(keypoints_pred, pca)
        pcasv_df = pd.DataFrame(pcasv_error_per_keypoint, index=index, columns=keypoint_names)
        # add train/val/test split
        if set is not None:
            pcasv_df["set"] = set
        save_file = preds_file.replace(".csv", "_pca_singleview_error.csv")
        pcasv_df.to_csv(save_file)

    if "pca_multiview" in metrics_to_compute:
        # build pca object
        pca = KeypointPCA(
            loss_type="pca_multiview",
            data_module=data_module,
            components_to_keep=cfg.losses.pca_singleview.components_to_keep,
            empirical_epsilon_percentile=cfg.losses.pca_singleview.get(
                "empirical_epsilon_percentile", 1.0),
            mirrored_column_matches=cfg.data.mirrored_column_matches,
        )
        # re-fit pca on the labeled data to get params
        pca()
        # compute reprojection error
        pcamv_error_per_keypoint = pca_multiview_reprojection_error(keypoints_pred, pca)
        pcamv_df = pd.DataFrame(pcamv_error_per_keypoint, index=index, columns=keypoint_names)
        # add train/val/test split
        if set is not None:
            pcamv_df["set"] = set
        save_file = preds_file.replace(".csv", "_pca_multiview_error.csv")
        pcamv_df.to_csv(save_file)


@typechecked
def export_predictions_and_labeled_video(
    video_file: str,
    cfg: DictConfig,
    prediction_csv_file: str,
    ckpt_file: Optional[str] = None,
    trainer: Optional[pl.Trainer] = None,
    model: Optional[ALLOWED_MODELS] = None,
    data_module: Optional[Union[BaseDataModule, UnlabeledDataModule]] = None,
    labeled_mp4_file: Optional[str] = None,
    save_heatmaps: Optional[bool] = False,
) -> None:
    """Export predictions csv and a labeled video for a single video file."""

    if ckpt_file is None and model is None:
        raise ValueError("either 'ckpt_file' or 'model' must be passed")

    # compute predictions
    preds_df = predict_single_video(
        video_file=video_file,
        ckpt_file=ckpt_file,
        cfg_file=cfg,
        preds_file=prediction_csv_file,
        trainer=trainer,
        model=model,
        data_module=data_module,
        save_heatmaps=save_heatmaps,
    )

    # create labeled video
    if labeled_mp4_file is not None:
        os.makedirs(os.path.dirname(labeled_mp4_file), exist_ok=True)
        # transform df to numpy array
        keypoints_arr = np.reshape(preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
        xs_arr = keypoints_arr[:, :, 0]
        ys_arr = keypoints_arr[:, :, 1]
        mask_array = keypoints_arr[:, :, 2] > cfg.eval.confidence_thresh_for_vid
        # video generation
        video_clip = VideoFileClip(video_file)
        create_labeled_video(
            clip=video_clip,
            xs_arr=xs_arr,
            ys_arr=ys_arr,
            mask_array=mask_array,
            filename=labeled_mp4_file,
            colormap=cfg.eval.get("colormap", "cool")
        )

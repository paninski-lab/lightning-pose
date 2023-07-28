"""Helper functions to build pipeline components from config dictionary."""

import os
from typing import Dict, List, Optional, Union

import imgaug.augmenters as iaa
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig, OmegaConf
from typeguard import typechecked

from lightning_pose.callbacks import AnnealWeight
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import BaseTrackingDataset, HeatmapDataset
from lightning_pose.data.utils import compute_num_train_frames, split_sizes_from_probabilities
from lightning_pose.losses.factory import LossFactory
from lightning_pose.metrics import (
    pca_multiview_reprojection_error,
    pca_singleview_reprojection_error,
    pixel_error,
    temporal_norm,
)
from lightning_pose.models.heatmap_tracker import HeatmapTracker, SemiSupervisedHeatmapTracker
from lightning_pose.models.heatmap_tracker_mhcrnn import (
    HeatmapTrackerMHCRNN,
    SemiSupervisedHeatmapTrackerMHCRNN,
)
from lightning_pose.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)
from lightning_pose.utils.io import (
    check_if_semi_supervised,
    get_keypoint_names,
    return_absolute_path,
)
from lightning_pose.utils.pca import KeypointPCA
from lightning_pose.utils.predictions import create_labeled_video, predict_single_video


@typechecked
def get_imgaug_transform(cfg: DictConfig) -> iaa.Sequential:
    """Create simple data transform pipeline that augments images.

    Args:
        cfg: standard config file that carries around dataset info; relevant is the
            parameter "cfg.training.imgaug" which can take on the following values:
                default: resizing only
                dlc: imgaug pipeline implemented in DLC 2.0 package
                dlc-light: curated subset of dlc augmentations with more conservative
                    parameter settings

    """

    kind = cfg.training.get("imgaug", "default")

    data_transform = []

    if kind == "default":
        # resizing happens below
        print("using default image augmentation pipeline (resizing only)")
    elif kind == "dlc":

        print("using dlc image augmentation pipeline")

        apply_prob_0 = 0.5

        # rotate
        rotation = 25  # rotation uniformly sampled from (-rotation, +rotation)
        data_transform.append(iaa.Sometimes(
            0.4,
            iaa.Affine(rotate=(-rotation, rotation))
        ))
        # motion blur
        k = 5  # kernel size of blur
        angle = 90  # blur direction uniformly sampled from (-angle, +angle)
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.MotionBlur(k=k, angle=(-angle, angle))
        ))
        # coarse dropout
        prct = 0.02  # drop `prct` of all pixels by converting them to black pixels
        size_prct = 0.3  # drop pix on a low-res version of img that's `size_prct` of og
        per_channel = 0.5  # per_channel transformations on `per_channel` frac of images
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.CoarseDropout(p=prct, size_percent=size_prct, per_channel=per_channel)
        ))
        # elastic transform
        alpha = (0, 10)  # controls strength of displacement
        sigma = 5  # cotnrols smoothness of displacement
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
        ))
        # hist eq
        data_transform.append(iaa.Sometimes(
            0.1,
            iaa.AllChannelsHistogramEqualization()
        ))
        # clahe (contrast limited adaptive histogram equalization) -
        # hist eq over image patches
        data_transform.append(iaa.Sometimes(
            0.1,
            iaa.AllChannelsCLAHE()
        ))
        # emboss
        alpha = (0, 0.5)  # overlay embossed image on original with alpha in this range
        strength = (0.5, 1.5)  # strength of embossing lies in this range
        data_transform.append(iaa.Sometimes(
            0.1,
            iaa.Emboss(alpha=alpha, strength=strength)
        ))
        # crop
        crop_by = 0.15  # number of pix to crop on each side of img given as a fraction
        data_transform.append(iaa.Sometimes(
            0.4,
            iaa.CropAndPad(percent=(-crop_by, crop_by), keep_size=False)
        ))

    elif kind == "dlc-light":

        print("using dlc-light image augmentation pipeline")

        apply_prob_0 = 0.25
        apply_prob_1 = 0.10

        # rotate
        rotation = 10  # rotation uniformly sampled from (-rotation, +rotation)
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.Affine(rotate=(-rotation, rotation))
        ))
        # motion blur
        k = 3  # kernel size of blur
        angle = 90  # blur direction uniformly sampled from (-angle, +angle)
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.MotionBlur(k=k, angle=(-angle, angle))
        ))
        # elastic transform
        alpha = (0, 10)  # controls strength of displacement
        sigma = 5  # cotnrols smoothness of displacement
        data_transform.append(iaa.Sometimes(
            apply_prob_0,
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
        ))
        # clahe (contrast limited adaptive histogram equalization) -
        # hist eq over image patches
        data_transform.append(iaa.Sometimes(
            apply_prob_1,
            iaa.AllChannelsCLAHE()
        ))
        # crop
        crop_by = 0.10  # number of pix to crop on each side of img given as a fraction
        data_transform.append(iaa.Sometimes(
            apply_prob_1,
            iaa.CropAndPad(percent=(-crop_by, crop_by), keep_size=False)
        ))
        # resize
        data_transform.append(
            iaa.Resize({
                "height": cfg.data.image_resize_dims.height,
                "width": cfg.data.image_resize_dims.width}
            ))

    else:
        raise NotImplementedError(
            "must choose imgaug kind from 'default', 'dlc', 'dlc-light'"
        )

    # always apply resizing transform
    data_transform.append(
        iaa.Resize({
            "height": cfg.data.image_resize_dims.height,
            "width": cfg.data.image_resize_dims.width}
        ))

    return iaa.Sequential(data_transform)


@typechecked
def get_dataset(
    cfg: DictConfig, data_dir: str, imgaug_transform: iaa.Sequential
) -> Union[BaseTrackingDataset, HeatmapDataset]:
    """Create a dataset that contains labeled data."""
    import os

    from PIL import Image

    if cfg.model.model_type == "regression":
        dataset = BaseTrackingDataset(
            root_directory=data_dir,
            csv_path=cfg.data.csv_file,
            imgaug_transform=imgaug_transform,
            do_context=cfg.model.do_context,
        )
    elif cfg.model.model_type == "heatmap" or cfg.model.model_type == "heatmap_mhcrnn":
        dataset = HeatmapDataset(
            root_directory=data_dir,
            csv_path=cfg.data.csv_file,
            imgaug_transform=imgaug_transform,
            downsample_factor=cfg.data.downsample_factor,
            do_context=cfg.model.model_type == "heatmap_mhcrnn" or cfg.model.do_context,
            uniform_heatmaps=cfg.training.get("uniform_heatmaps_for_nan_keypoints", False),
        )
    else:
        raise NotImplementedError(
            "%s is an invalid cfg.model.model_type" % cfg.model.model_type
        )
    image = Image.open(os.path.join(dataset.root_directory, dataset.image_names[0])).convert("RGB")
    if image.size != (
        cfg.data.image_orig_dims.width,
        cfg.data.image_orig_dims.height,
    ):
        raise ValueError(
            f"image_orig_dims in data configuration file is "
            f"(width={cfg.data.image_orig_dims.width}, height={cfg.data.image_orig_dims.height}) "
            f"but your image size is (width={image.size[0]}, height={image.size[1]}). "
            f"Please update the data configuration file"
        )
    return dataset


@typechecked
def get_data_module(
    cfg: DictConfig,
    dataset: Union[BaseTrackingDataset, HeatmapDataset],
    video_dir: Optional[str] = None,
) -> Union[BaseDataModule, UnlabeledDataModule]:
    """Create a data module that splits a dataset into train/val/test iterators."""

    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    if not semi_supervised:
        if not (cfg.training.gpu_id, int):
            raise NotImplementedError(
                "Cannot currently fit fully supervised model on multiple gpus"
            )
        data_module = BaseDataModule(
            dataset=dataset,
            train_batch_size=cfg.training.train_batch_size,
            val_batch_size=cfg.training.val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.num_workers,
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
            torch_seed=cfg.training.rng_seed_data_pt,
        )
    else:
        if not (cfg.training.gpu_id, int):
            raise NotImplementedError(
                "Cannot currently fit semi-supervised model on multiple gpus"
            )
        data_module = UnlabeledDataModule(
            dataset=dataset,
            video_paths_list=video_dir,
            train_batch_size=cfg.training.train_batch_size,
            val_batch_size=cfg.training.val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.num_workers,
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
            dali_config=cfg.dali,
            torch_seed=cfg.training.rng_seed_data_pt,
            imgaug=cfg.training.get("imgaug", "default"),
        )
    return data_module


@typechecked
def get_loss_factories(
    cfg: DictConfig, data_module: Union[BaseDataModule, UnlabeledDataModule]
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
                loss_params_dict["unsupervised"][loss_name][
                    "mirrored_column_matches"
                ] = cfg.data.mirrored_column_matches
            elif loss_name == "pca_singleview":
                loss_params_dict["unsupervised"][loss_name][
                    "columns_for_singleview_pca"
                ] = cfg.data.columns_for_singleview_pca

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

    lr_scheduler = cfg.training["lr_scheduler"]
    lr_scheduler_params = cfg.training["lr_scheduler_params"][lr_scheduler]
    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    image_h = cfg.data.image_resize_dims.height
    image_w = cfg.data.image_resize_dims.width
    if "vit" in cfg.model.backbone:
        if image_h != image_w:
            raise RuntimeError("ViT model requires resized height and width to be equal")

    if not semi_supervised:
        if cfg.model.model_type == "regression":
            model = RegressionTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                backbone=cfg.model.backbone,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                do_context=cfg.model.do_context,
                image_size=image_h,  # only used by ViT
            )
        elif cfg.model.model_type == "heatmap":
            model = HeatmapTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                backbone=cfg.model.backbone,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=data_module.dataset.output_shape,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                do_context=cfg.model.do_context,
                image_size=image_h,  # only used by ViT
            )
        elif cfg.model.model_type == "heatmap_mhcrnn":
            model = HeatmapTrackerMHCRNN(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                backbone=cfg.model.backbone,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=data_module.dataset.output_shape,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
            )
        else:
            raise NotImplementedError(
                "%s is an invalid cfg.model.model_type for a fully supervised model"
                % cfg.model.model_type
            )

    else:
        if cfg.model.model_type == "regression":
            model = SemiSupervisedRegressionTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                loss_factory_unsupervised=loss_factories["unsupervised"],
                backbone=cfg.model.backbone,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                do_context=cfg.model.do_context,
                image_size=image_h,  # only used by ViT
            )

        elif cfg.model.model_type == "heatmap":
            model = SemiSupervisedHeatmapTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                loss_factory_unsupervised=loss_factories["unsupervised"],
                backbone=cfg.model.backbone,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=data_module.dataset.output_shape,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                do_context=cfg.model.do_context,
                image_size=image_h,  # only used by ViT
            )
        elif cfg.model.model_type == "heatmap_mhcrnn":
            model = SemiSupervisedHeatmapTrackerMHCRNN(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories["supervised"],
                loss_factory_unsupervised=loss_factories["unsupervised"],
                backbone=cfg.model.backbone,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=data_module.dataset.output_shape,
                torch_seed=cfg.training.rng_seed_model_pt,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,  # only used by ViT
            )
        else:
            raise NotImplementedError(
                "%s is an invalid cfg.model.model_type for a semi-supervised model"
                % cfg.model.model_type
            )

    return model


@typechecked
def get_callbacks(
    cfg: DictConfig,
    early_stopping=True,
    lr_monitor=True,
    ckpt_model=True,
    backbone_unfreeze=True,
) -> list:

    callbacks = []

    if early_stopping:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_supervised_loss",
            patience=cfg.training.early_stop_patience,
            mode="min",
        )
        callbacks.append(early_stopping)
    if lr_monitor:
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    if ckpt_model:
        ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val_supervised_loss"
        )
        callbacks.append(ckpt_callback)
    if backbone_unfreeze:
        transfer_unfreeze_callback = pl.callbacks.BackboneFinetuning(
            unfreeze_backbone_at_epoch=cfg.training.unfreezing_epoch,
            lambda_func=lambda epoch: 1.5,
            backbone_initial_ratio_lr=0.1,
            should_align=True,
            train_bn=True,
        )
        callbacks.append(transfer_unfreeze_callback)
    # we just need this callback for unsupervised models
    if (cfg.model.losses_to_use != []) and (cfg.model.losses_to_use is not None):
        anneal_weight_callback = AnnealWeight(**cfg.callbacks.anneal_weight)
        callbacks.append(anneal_weight_callback)

    return callbacks


@typechecked
def calculate_train_batches(
    cfg: DictConfig,
    dataset: Optional[Union[BaseTrackingDataset, HeatmapDataset]] = None
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
        # TODO: small bit of redundant code from datamodule
        datalen = dataset.__len__()
        data_splits_list = split_sizes_from_probabilities(
            datalen,
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
        )
        num_train_frames = compute_num_train_frames(
            data_splits_list[0], cfg.training.get("train_frames", None)
        )
        num_labeled_batches = int(np.ceil(num_train_frames / cfg.training.train_batch_size))
        limit_train_batches = np.max([num_labeled_batches, 10])  # 10 is minimum
    else:
        limit_train_batches = cfg.training.limit_train_batches

    return int(limit_train_batches)


@typechecked
def compute_metrics(
    cfg: DictConfig,
    preds_file: str,
    data_module: Optional[Union[BaseDataModule, UnlabeledDataModule]] = None,
) -> None:
    """Compute various metrics on predictions csv file."""

    # get keypoint names
    labels_file = return_absolute_path(
        os.path.join(cfg["data"]["data_dir"], cfg["data"]["csv_file"]))
    labels_df = pd.read_csv(labels_file, header=[0, 1, 2], index_col=0)
    keypoint_names = get_keypoint_names(
        cfg, csv_file=labels_file, header_rows=[0, 1, 2])

    # load predictions
    pred_df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)
    if pred_df.keys()[-1][0] == "set":
        # these are predictions on labeled data
        # get rid of last column that contains info about train/val/test set
        is_video = False
        tmp = pred_df.iloc[:, :-1].to_numpy().reshape(pred_df.shape[0], -1, 3)
        index = labels_df.index
        set = pred_df.iloc[:, -1].to_numpy()
    else:
        # these are predictions on video data
        is_video = True
        tmp = pred_df.to_numpy().reshape(pred_df.shape[0], -1, 3)
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
        cfg.data.get("columns_for_singleview_pca", None) is not None
        and len(cfg.data.columns_for_singleview_pca) != 0
    ):
        metrics_to_compute += ["pca_singleview"]
    if (
        cfg.data.get("mirrored_column_matches", None) is not None
        and len(cfg.data.mirrored_column_matches) != 0
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
            empirical_epsilon_percentile=cfg.losses.pca_singleview.empirical_epsilon_percentile,
            columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
        )
        # re-fit pca on the labeled data to get params
        pca()
        # compute reprojection error
        pcasv_error_per_keypoint = pca_singleview_reprojection_error(keypoints_pred, pca, cfg)
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
            empirical_epsilon_percentile=cfg.losses.pca_singleview.empirical_epsilon_percentile,
            mirrored_column_matches=cfg.data.mirrored_column_matches,
        )
        # re-fit pca on the labeled data to get params
        pca()
        # compute reprojection error
        pcamv_error_per_keypoint = pca_multiview_reprojection_error(keypoints_pred, pca, cfg)
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
    ckpt_file: str,
    trainer: Optional[pl.Trainer] = None,
    model: Optional[
        Union[
            RegressionTracker,
            HeatmapTracker,
            HeatmapTrackerMHCRNN,
            SemiSupervisedRegressionTracker,
            SemiSupervisedHeatmapTracker,
            SemiSupervisedHeatmapTrackerMHCRNN,
        ]
    ] = None,
    data_module: Optional[Union[BaseDataModule, UnlabeledDataModule]] = None,
    labeled_mp4_file: Optional[str] = None,
    save_heatmaps: Optional[bool] = False,
) -> None:
    """Export predictions csv and a labeled video for a single video file."""

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
        video_clip = VideoFileClip(video_file)
        # transform df to numpy array
        keypoints_arr = np.reshape(preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
        xs_arr = keypoints_arr[:, :, 0]
        ys_arr = keypoints_arr[:, :, 1]
        mask_array = keypoints_arr[:, :, 2] > cfg.eval.confidence_thresh_for_vid
        # video generation
        create_labeled_video(
            clip=video_clip,
            xs_arr=xs_arr,
            ys_arr=ys_arr,
            mask_array=mask_array,
            filename=labeled_mp4_file,
        )

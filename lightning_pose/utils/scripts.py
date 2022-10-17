"""Helper functions to build pipeline components from config dictionary."""

import imgaug.augmenters as iaa
from moviepy.editor import VideoFileClip
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import pytorch_lightning as pl
from typeguard import typechecked
from typing import Dict, Optional, Union

from lightning_pose.data.dali import PrepareDALI
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import BaseTrackingDataset, HeatmapDataset
from lightning_pose.losses.factory import LossFactory
from lightning_pose.models.heatmap_tracker import (
    HeatmapTracker,
    SemiSupervisedHeatmapTracker,
)
from lightning_pose.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)
from lightning_pose.utils import get_gpu_list_from_cfg, pretty_print_str
from lightning_pose.utils.io import check_if_semi_supervised, return_absolute_data_paths
from lightning_pose.utils.predictions import load_model_from_checkpoint, create_labeled_video, \
    PredictionHandler, predict_single_video


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
    from PIL import Image
    import os

    if cfg.model.model_type == "regression":
        dataset = BaseTrackingDataset(
            root_directory=data_dir,
            csv_path=cfg.data.csv_file,
            header_rows=OmegaConf.to_object(cfg.data.header_rows),
            imgaug_transform=imgaug_transform,
            do_context=cfg.model.do_context,
        )
    elif cfg.model.model_type == "heatmap":
        dataset = HeatmapDataset(
            root_directory=data_dir,
            csv_path=cfg.data.csv_file,
            header_rows=OmegaConf.to_object(cfg.data.header_rows),
            imgaug_transform=imgaug_transform,
            downsample_factor=cfg.data.downsample_factor,
            do_context=cfg.model.do_context,
        )
    else:
        raise NotImplementedError(
            "%s is an invalid cfg.model.model_type" % cfg.model.model_type
        )
    image = Image.open(
        os.path.join(dataset.root_directory, dataset.image_names[0])
    ).convert("RGB")
    if image.size != (
        cfg.data.image_orig_dims.width,
        cfg.data.image_orig_dims.height,
    ):
        raise ValueError(
            f"image_orig_dims in data configuration file is (width=%i, height=%i) but"
            f" your image size is (width=%i, height=%i). Please update the data "
            f"configuration file"
            % (
                cfg.data.image_orig_dims.width,
                cfg.data.image_orig_dims.height,
                image.size[0],
                image.size[1],
            )
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
    if cfg.model.model_type == "heatmap":
        loss_params_dict["supervised"]["heatmap_" + cfg.model.heatmap_loss_type] = {
            "log_weight": 0.0
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
            if loss_name[:8] == "unimodal":
                if cfg.model.model_type == "regression":
                    raise NotImplementedError(
                        f"unimodal loss can only be used with classes inheriting from "
                        f"HeatmapTracker. \nYou specified a RegressionTracker model."
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
                height_ds = int(height_og // (2**cfg.data.downsample_factor))
                width_ds = int(width_og // (2**cfg.data.downsample_factor))
                loss_params_dict["unsupervised"][loss_name][
                    "downsampled_image_height"
                ] = height_ds
                loss_params_dict["unsupervised"][loss_name][
                    "downsampled_image_width"
                ] = width_ds
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
        learn_weights=cfg.model.learn_weights,
    )

    return {"supervised": loss_factory_sup, "unsupervised": loss_factory_unsup}


@typechecked
def get_model(
    cfg: DictConfig,
    data_module: Union[BaseDataModule, UnlabeledDataModule],
    loss_factories: Dict[str, LossFactory],
) -> Union[
    RegressionTracker,
    HeatmapTracker,
    SemiSupervisedRegressionTracker,
    SemiSupervisedHeatmapTracker,
]:
    """Create model: regression or heatmap based, supervised or semi-supervised."""

    lr_scheduler = cfg.training["lr_scheduler"]
    lr_scheduler_params = cfg.training["lr_scheduler_params"][lr_scheduler]
    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
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
            )
        else:
            raise NotImplementedError(
                "%s is an invalid cfg.model.model_type for a fully supervised model"
                % cfg.model.model_type
            )
        # add losses onto initialized model
        # model.add_loss_factory(loss_factories["supervised"])

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
            )
        else:
            raise NotImplementedError(
                "%s is an invalid cfg.model.model_type for a semi-supervised model"
                % cfg.model.model_type
            )
    return model


@typechecked
def export_predictions_and_labeled_video(
    video_file: str,
    cfg: DictConfig,
    prediction_csv_file: str,
    ckpt_file: str,
    trainer: Optional[pl.Trainer] = None,
    model: Optional[Union[
        RegressionTracker,
        HeatmapTracker,
        SemiSupervisedRegressionTracker,
        SemiSupervisedHeatmapTracker,
    ]] = None,
    data_module: Optional[Union[BaseDataModule, UnlabeledDataModule]] = None,
    gpu_id: Optional[int] = None,
    labeled_mp4_file: Optional[str] = None,
) -> None:
    """Export predictions csv and a labeled video for a single video file."""

    # compute predictions
    preds_df = predict_single_video(
        video_file=video_file,
        ckpt_file=ckpt_file,
        cfg_file=cfg,
        preds_file=prediction_csv_file,
        gpu_id=gpu_id,
        trainer=trainer,
        model=model,
        data_module=data_module,
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

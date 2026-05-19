"""Factory function for creating pose estimation models from config."""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig, ListConfig

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.models.base import (
    _apply_defaults_for_lr_scheduler_params,
    _apply_defaults_for_optimizer_params,
    check_if_semi_supervised,
)

if TYPE_CHECKING:
    from lightning_pose.losses.factory import LossFactory
    from lightning_pose.models import ALLOWED_MODELS

__all__ = ['get_model']


def get_model(
    cfg: DictConfig | ListConfig,
    data_module: BaseDataModule | UnlabeledDataModule | None,
    loss_factories: dict[str, LossFactory] | dict[str, None],
) -> ALLOWED_MODELS:
    """Create model: regression or heatmap based, supervised or semi-supervised."""

    optimizer = cfg.training.get('optimizer', 'Adam')
    optimizer_params = _apply_defaults_for_optimizer_params(
        optimizer,
        cfg.training.get('optimizer_params'),
    )

    lr_scheduler = cfg.training.get('lr_scheduler', 'multisteplr')
    lr_scheduler_params = _apply_defaults_for_lr_scheduler_params(
        lr_scheduler,
        cfg.training.get('lr_scheduler_params', {}).get(f'{lr_scheduler}'),
    )

    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    image_h = cfg.data.image_resize_dims.height
    image_w = cfg.data.image_resize_dims.width
    if 'vit' in cfg.model.backbone:
        if image_h != image_w:
            raise RuntimeError('ViT model requires resized height and width to be equal')

    backbone_pretrained = cfg.model.get('backbone_pretrained', True)
    if not semi_supervised:
        if cfg.model.model_type == 'regression':
            from lightning_pose.models import RegressionTracker
            model = RegressionTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories['supervised'],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,
            )
        elif cfg.model.model_type == 'heatmap':
            num_targets = data_module.dataset.num_targets if data_module else None
            from lightning_pose.models import HeatmapTracker
            model = HeatmapTracker(
                num_keypoints=cfg.data.num_keypoints,
                num_targets=num_targets,
                loss_factory=loss_factories['supervised'],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.get('downsample_factor', 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,
                backbone_checkpoint=cfg.model.get('backbone_checkpoint'),
            )
        elif cfg.model.model_type == 'heatmap_mhcrnn':
            from lightning_pose.models import HeatmapTrackerMHCRNN
            model = HeatmapTrackerMHCRNN(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories['supervised'],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.get('downsample_factor', 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,
                backbone_checkpoint=cfg.model.get('backbone_checkpoint'),
            )
        elif cfg.model.model_type == 'heatmap_multiview_transformer':
            from lightning_pose.models import HeatmapTrackerMultiviewTransformer
            model = HeatmapTrackerMultiviewTransformer(
                num_keypoints=cfg.data.num_keypoints,
                num_views=len(cfg.data.view_names),
                loss_factory=loss_factories['supervised'],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                head=cfg.model.get('head', 'heatmap_cnn'),
                downsample_factor=cfg.data.get('downsample_factor', 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,
                backbone_checkpoint=cfg.model.get('backbone_checkpoint'),
            )
        else:
            raise NotImplementedError(
                f'{cfg.model.model_type} is an invalid cfg.model.model_type for a fully '
                f'supervised model'
            )

    else:
        if cfg.model.model_type == 'regression':
            from lightning_pose.models import SemiSupervisedRegressionTracker
            model = SemiSupervisedRegressionTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories['supervised'],
                loss_factory_unsupervised=loss_factories['unsupervised'],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,
            )
        elif cfg.model.model_type == 'heatmap':
            from lightning_pose.models import SemiSupervisedHeatmapTracker
            model = SemiSupervisedHeatmapTracker(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories['supervised'],
                loss_factory_unsupervised=loss_factories['unsupervised'],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.get('downsample_factor', 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,
                backbone_checkpoint=cfg.model.get('backbone_checkpoint'),
            )
        elif cfg.model.model_type == 'heatmap_mhcrnn':
            from lightning_pose.models import SemiSupervisedHeatmapTrackerMHCRNN
            model = SemiSupervisedHeatmapTrackerMHCRNN(
                num_keypoints=cfg.data.num_keypoints,
                loss_factory=loss_factories['supervised'],
                loss_factory_unsupervised=loss_factories['unsupervised'],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                downsample_factor=cfg.data.get('downsample_factor', 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,
                backbone_checkpoint=cfg.model.get('backbone_checkpoint'),
            )
        elif cfg.model.model_type == 'heatmap_multiview_transformer':
            from lightning_pose.models import SemiSupervisedHeatmapTrackerMultiviewTransformer
            model = SemiSupervisedHeatmapTrackerMultiviewTransformer(
                num_keypoints=cfg.data.num_keypoints,
                num_views=len(cfg.data.view_names),
                loss_factory=loss_factories['supervised'],
                loss_factory_unsupervised=loss_factories['unsupervised'],
                backbone=cfg.model.backbone,
                pretrained=backbone_pretrained,
                head=cfg.model.get('head', 'heatmap_cnn'),
                downsample_factor=cfg.data.get('downsample_factor', 2),
                torch_seed=cfg.training.rng_seed_model_pt,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params,
                image_size=image_h,
                patch_mask_config=cfg.training.get('patch_mask', {}),
            )
        else:
            raise NotImplementedError(
                f'{cfg.model.model_type} invalid cfg.model.model_type for a semi-supervised model'
            )

    if cfg.model.get('checkpoint', None):
        ckpt = cfg.model.checkpoint
        print(f'Loading weights from {ckpt}')
        if not ckpt.endswith('.ckpt'):
            ckpt = glob.glob(os.path.join(ckpt, '**', '*.ckpt'), recursive=True)[0]
        try:
            state_dict = torch.load(ckpt)['state_dict']
        except Exception as e:
            print(f'Warning: Failed to load checkpoint with default settings: {e}')
            print('Attempting to load with weights_only=False...')
            state_dict = torch.load(ckpt, weights_only=False)['state_dict']
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for key, val in state_dict.items():
                if 'backbone' in key:
                    new_state_dict[key] = val
            model.load_state_dict(new_state_dict, strict=False)

    return model

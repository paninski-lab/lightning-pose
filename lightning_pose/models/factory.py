"""Factory functions for building pose estimation models from a Hydra config.

Public entry points:

- :func:`get_model_class` — pure dispatch: returns the model *class* for a given
  ``(model_type, semi_supervised)`` pair without instantiating anything.
- :func:`get_model` — full construction: resolves optimizer/scheduler defaults,
  instantiates the appropriate model class, and optionally loads weights from a
  checkpoint.

All model class imports are deferred inside the function bodies to avoid circular
imports (this module is loaded early in the call stack, before the model classes are
fully defined).

**Supported model types**: ``regression``, ``heatmap``, ``heatmap_mhcrnn``,
``heatmap_multiview_transformer``.

**Adding a new model type**: add its string to :data:`ALLOWED_MODEL_TYPES`, add a
branch in :func:`get_model_class` (two lines, one per supervision mode), add an
``elif`` block in :func:`get_model` for its constructor kwargs, and create the model
file(s) under ``lightning_pose/models/``.
"""

from __future__ import annotations

import copy
import glob
import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

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

logger = logging.getLogger(__name__)

ALLOWED_MODEL_TYPES = Literal[
    'regression',
    'heatmap',
    'heatmap_mhcrnn',
    'heatmap_multiview_transformer',
]

__all__: list[str] = []


def get_model_class(
    model_type: ALLOWED_MODEL_TYPES,
    semi_supervised: bool,
) -> type[ALLOWED_MODELS]:
    """Return the model class for the given model type and supervision mode.

    Args:
        model_type: one of ``'regression'``, ``'heatmap'``, ``'heatmap_mhcrnn'``,
            ``'heatmap_multiview_transformer'``.
        semi_supervised: True to return the semi-supervised variant.

    Returns:
        model class (not an instance).

    Raises:
        NotImplementedError: if ``model_type`` is not recognised.

    """
    if not semi_supervised:
        if model_type == 'regression':
            from lightning_pose.models import RegressionTracker as ModelClass
        elif model_type == 'heatmap':
            from lightning_pose.models import HeatmapTracker as ModelClass
        elif model_type == 'heatmap_mhcrnn':
            from lightning_pose.models import HeatmapTrackerMHCRNN as ModelClass
        elif model_type == 'heatmap_multiview_transformer':
            from lightning_pose.models import HeatmapTrackerMultiviewTransformer as ModelClass
        else:
            raise NotImplementedError(
                f'{model_type} is an invalid model_type for a fully supervised model'
            )
    else:
        if model_type == 'regression':
            from lightning_pose.models import SemiSupervisedRegressionTracker as ModelClass
        elif model_type == 'heatmap':
            from lightning_pose.models import SemiSupervisedHeatmapTracker as ModelClass
        elif model_type == 'heatmap_mhcrnn':
            from lightning_pose.models import SemiSupervisedHeatmapTrackerMHCRNN as ModelClass
        elif model_type == 'heatmap_multiview_transformer':
            from lightning_pose.models import (
                SemiSupervisedHeatmapTrackerMultiviewTransformer as ModelClass,
            )
        else:
            raise NotImplementedError(
                f'{model_type} is an invalid model_type for a semi-supervised model'
            )
    return ModelClass


def get_model(
    cfg: DictConfig | ListConfig,
    data_module: BaseDataModule | UnlabeledDataModule | None,
    loss_factories: dict[str, LossFactory] | dict[str, None],
) -> ALLOWED_MODELS:
    """Build a pose estimation model from a Hydra config.

    Resolves optimizer and lr-scheduler defaults, then dispatches on
    ``cfg.model.model_type`` and whether unsupervised losses are present to instantiate
    the appropriate model class. Optionally loads weights from ``cfg.model.checkpoint``
    after construction (supports both ``.ckpt`` files and directories containing one).

    Args:
        cfg: Hydra config. Relevant fields:
            - ``cfg.model.model_type``: one of ``'regression'``, ``'heatmap'``,
              ``'heatmap_mhcrnn'``, ``'heatmap_multiview_transformer'``.
            - ``cfg.model.backbone``: backbone identifier (see ``ALLOWED_BACKBONES``).
            - ``cfg.model.losses_to_use``: list of unsupervised loss names; empty/None
              selects the fully supervised branch.
            - ``cfg.model.checkpoint``: optional path to a ``.ckpt`` file or directory
              from which to load weights after construction.
            - ``cfg.data.image_resize_dims``: ViT backbones require height == width.
        data_module: data module used to infer ``num_targets`` for heatmap models;
            may be ``None`` when building a model without a dataset (e.g. inference only).
        loss_factories: dict with keys ``'supervised'`` and ``'unsupervised'``, each
            mapping to a :class:`~lightning_pose.losses.factory.LossFactory` instance
            (or ``None`` for stub construction in tests).

    Returns:
        instantiated model ready for training or inference.

    Raises:
        RuntimeError: if a ViT backbone is selected with non-square image dimensions.
        NotImplementedError: if ``cfg.model.model_type`` is not a recognised value.
    """
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
    ModelClass = get_model_class(cfg.model.model_type, semi_supervised)

    # args shared by every model type
    common = dict(
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
    if semi_supervised:
        common['loss_factory_unsupervised'] = loss_factories['unsupervised']

    # model-type-specific constructor args
    if cfg.model.model_type == 'regression':
        extra: dict = {}
    elif cfg.model.model_type == 'heatmap':
        num_targets = data_module.dataset.num_targets if data_module else None
        extra = dict(
            num_targets=num_targets,
            downsample_factor=cfg.data.get('downsample_factor', 2),
            backbone_checkpoint=cfg.model.get('backbone_checkpoint'),
        )
        freeze_names = cfg.model.get('head_freeze_keypoints')
        if freeze_names:
            if cfg.model.get('head_mode', 'shared') != 'shared':
                raise ValueError('model.head_freeze_keypoints requires head_mode=shared')
            if data_module is None:
                logger.info('head_freeze_keypoints given without a data module (inference); ignored')
            else:
                names = list(data_module.dataset.keypoint_names)
                unknown = [n for n in freeze_names if n not in names]
                if unknown:
                    raise ValueError(
                        f'model.head_freeze_keypoints not in data.keypoint_names: {unknown}'
                    )
                extra['head_freeze_keypoints'] = [names.index(n) for n in freeze_names]
        head_mode = cfg.model.get('head_mode', 'shared')
        if head_mode not in ('shared', 'per_dataset', 'dataset_token'):
            raise ValueError(
                f"model.head_mode must be 'shared', 'per_dataset', or 'dataset_token', "
                f"got '{head_mode}'"
            )
        if head_mode in ('per_dataset', 'dataset_token'):
            if semi_supervised:
                raise NotImplementedError(
                    f'model.head_mode={head_mode} is not supported with unsupervised '
                    'losses: unlabeled video frames carry no dataset id'
                )
            dataset_names = cfg.data.get('dataset_names', None)
            if not dataset_names:
                raise ValueError(
                    f'model.head_mode={head_mode} requires data.dataset_names so '
                    'batches carry per-example dataset ids'
                )
            if head_mode == 'per_dataset':
                from lightning_pose.models import MultiHeadHeatmapTracker
                ModelClass = MultiHeadHeatmapTracker
            else:
                from lightning_pose.models import TokenConditionedHeatmapTracker
                ModelClass = TokenConditionedHeatmapTracker
                extra['token_lr'] = float(cfg.model.get('token_lr', 1e-2))
            extra['dataset_names'] = list(dataset_names)
    elif cfg.model.model_type == 'heatmap_mhcrnn':
        extra = dict(
            downsample_factor=cfg.data.get('downsample_factor', 2),
            backbone_checkpoint=cfg.model.get('backbone_checkpoint'),
        )
    elif cfg.model.model_type == 'heatmap_multiview_transformer':
        extra = dict(
            num_views=len(cfg.data.view_names),
            head=cfg.model.get('head', 'heatmap_cnn'),
            downsample_factor=cfg.data.get('downsample_factor', 2),
            backbone_checkpoint=cfg.model.get('backbone_checkpoint'),
        )
        if semi_supervised:
            extra['patch_mask_config'] = cfg.training.get('patch_mask', {})
    else:
        raise NotImplementedError(
            f'{cfg.model.model_type} is an invalid cfg.model.model_type'
        )

    model = ModelClass(**common, **extra)

    # LoRA on the backbone: base weights frozen, low-rank adapters trainable. Applied before
    # checkpoint loading — LoRALinear keeps the weight/bias keys, so the trunk loads unchanged
    # and the (zero-initialised) adapters are the only new tensors.
    lora_cfg = cfg.model.get('lora', None)
    if lora_cfg:
        from lightning_pose.models.backbones.lora import apply_lora
        apply_lora(
            model.backbone,
            targets=list(lora_cfg.get('targets', ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'])),
            rank=int(lora_cfg.get('rank', 16)),
            alpha=float(lora_cfg.get('alpha', 2 * int(lora_cfg.get('rank', 16)))),
        )
        model.lora_lr = float(lora_cfg['lr']) if lora_cfg.get('lr') is not None else None

    # fill the multi-head supporting-set mask from training-data visibility; the buffer
    # is non-persistent, so this runs at every construction (training and inference both
    # have a data module carrying the labeled dataset)
    if cfg.model.get('head_mode', 'shared') == 'per_dataset' and data_module is not None:
        dataset = data_module.dataset
        if getattr(dataset, 'visibility', None) is not None and dataset.dataset_ids is not None:
            model.set_head_keypoint_mask(
                visibility=dataset.visibility,
                dataset_ids=dataset.dataset_ids,
                keypoint_names=dataset.keypoint_names,
                hflip=bool(cfg.training.get('imgaug_hflip', False)),
            )
        model.blind_gamma = float(cfg.model.get('blind_gamma', 2.0))
        model.blind_conf_floor = float(cfg.model.get('blind_conf_floor', 0.0))

    if cfg.model.get('checkpoint', None):
        ckpt = cfg.model.checkpoint
        logger.info(f'loading weights from {ckpt}')
        if not ckpt.endswith('.ckpt'):
            ckpt = glob.glob(os.path.join(ckpt, '**', '*.ckpt'), recursive=True)[0]
        try:
            state_dict = torch.load(ckpt)['state_dict']
        except Exception as e:
            logger.warning(f'failed to load checkpoint with default settings: {e}')
            logger.warning('attempting to load with weights_only=False...')
            state_dict = torch.load(ckpt, weights_only=False)['state_dict']
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for key, val in state_dict.items():
                if 'backbone' in key:
                    new_state_dict[key] = val
            model.load_state_dict(new_state_dict, strict=False)

    # anchor teacher: a frozen copy of the model as loaded (LoRA adapters are zero at this
    # point, so the copy computes exactly the trunk). Kept outside the module tree so it is
    # not a submodule: absent from state_dict/checkpoints, parameters(), and .to(); the
    # student moves it to its own device lazily in get_loss_inputs_labeled.
    anchor_cfg = cfg.model.get('anchor', None)
    if anchor_cfg and data_module is not None:
        if not cfg.model.get('checkpoint', None):
            raise ValueError('model.anchor requires model.checkpoint (the teacher weights)')
        if cfg.model.model_type != 'heatmap' or cfg.model.get('head_mode', 'shared') != 'shared':
            raise ValueError('model.anchor is implemented for shared-head heatmap models only')
        # the loss factories hold the data module (and, for semi-supervised models, a DALI
        # pipeline that cannot be copied); the teacher only needs weights, so detach them
        factories = {
            n: model.__dict__['_modules'].pop(n)
            for n in ('loss_factory', 'loss_factory_unsup') if n in model.__dict__['_modules']
        }
        try:
            teacher = copy.deepcopy(model)
        finally:
            model.__dict__['_modules'].update(factories)
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        object.__setattr__(model, '_anchor_teacher', teacher)
        logger.info(
            f'anchor: frozen teacher attached (weight {float(anchor_cfg.get("weight", 1.0)):g}, '
            f'mode {anchor_cfg.get("mode", "unlabeled")})'
        )

    return model

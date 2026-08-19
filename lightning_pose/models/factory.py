"""Factory functions for building pose estimation models from a Hydra config.

Public entry points:

- :func:`get_model_class` — pure dispatch: returns the model *class* for a given
  ``(model_type, semi_supervised)`` pair without instantiating anything.
- :func:`get_model` — full construction: resolves optimizer/scheduler defaults,
  instantiates the appropriate model class, and optionally loads weights from a
  checkpoint. Before construction, validates that every configured loss's required
  inputs are actually produced by the chosen model type (see
  :func:`_validate_loss_model_compatibility`).

All model class imports are deferred inside the function bodies to avoid circular
imports (this module is loaded early in the call stack, before the model classes are
fully defined).

**Supported model types**: ``regression``, ``heatmap``, ``heatmap_mhcrnn``,
``heatmap_multiview_transformer``.

**Adding a new model type**: add its string to :data:`ALLOWED_MODEL_TYPES`, add a
branch in :func:`get_model_class` (two lines, one per supervision mode), add an
``elif`` block in :func:`get_model` for its constructor kwargs, and create the model
file(s) under ``lightning_pose/models/``. No change is needed for loss compatibility
checking as long as the new tracker's ``get_loss_inputs_labeled``/``_unlabeled``
methods are annotated with an ``OutputsDict`` TypedDict from
:mod:`lightning_pose.models.datatypes` — :func:`_validate_loss_model_compatibility`
reads the produced keys from that annotation and the required keys from each loss's
own ``__call__`` signature, so neither list needs manual upkeep.
"""

from __future__ import annotations

import glob
import inspect
import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal, get_type_hints

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
    from lightning_pose.losses.losses import Loss
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


def _loss_required_keys(loss_cls: type[Loss]) -> set[str]:
    """Return the data-dict keys ``loss_cls.__call__`` requires.

    Derived from the signature itself rather than a hand-maintained list, so it can
    never drift from the loss's actual implementation. A parameter counts as required
    when it has no default and isn't ``self``, ``stage``, or a ``*args``/``**kwargs``
    catch-all.

    Args:
        loss_cls: loss class to inspect (not an instance).

    Returns:
        set of required keyword argument names.
    """
    sig = inspect.signature(loss_cls.__call__)
    return {
        name for name, param in sig.parameters.items()
        if name not in ('self', 'stage')
        and param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL)
        and param.default is param.empty
    }


def _tracker_output_keys(model_cls: type[ALLOWED_MODELS], method_name: str) -> set[str]:
    """Return the data-dict keys ``model_cls.<method_name>`` produces.

    Derived from the method's ``OutputsDict`` return-type annotation (see
    :mod:`lightning_pose.models.datatypes`) rather than a hand-maintained list.
    ``getattr`` resolves through the MRO to the concrete subclass override, so this
    always sees a single ``TypedDict``, never the abstract base class's ``Union``.

    Args:
        model_cls: tracker class to inspect (not an instance).
        method_name: ``'get_loss_inputs_labeled'`` or ``'get_loss_inputs_unlabeled'``.

    Returns:
        set of keys the method's declared return type contains.
    """
    return_type = get_type_hints(getattr(model_cls, method_name))['return']
    return set(return_type.__required_keys__) | set(return_type.__optional_keys__)


def _validate_loss_model_compatibility(
    model_cls: type[ALLOWED_MODELS],
    loss_factories: dict[str, LossFactory] | dict[str, None],
    semi_supervised: bool,
) -> None:
    """Raise if a configured loss needs data the chosen model type doesn't produce.

    Catches an invalid loss/model pairing at construction time with an actionable
    message, instead of a ``TypeError`` surfacing deep inside the first training step.

    Args:
        model_cls: tracker class about to be instantiated.
        loss_factories: dict with ``'supervised'`` and ``'unsupervised'`` LossFactory
            instances (or ``None``, in which case that stage is skipped).
        semi_supervised: whether the unsupervised loss factory should also be checked.

    Raises:
        ValueError: if any configured loss requires a key the model does not produce.
    """
    stages = [('get_loss_inputs_labeled', 'supervised')]
    if semi_supervised:
        stages.append(('get_loss_inputs_unlabeled', 'unsupervised'))

    for method_name, stage_key in stages:
        loss_factory = loss_factories[stage_key]
        if loss_factory is None:
            continue
        produced = _tracker_output_keys(model_cls, method_name)
        for loss_name, loss_instance in loss_factory.loss_instance_dict.items():
            missing = _loss_required_keys(type(loss_instance)) - produced
            if missing:
                raise ValueError(
                    f"loss '{loss_name}' requires {sorted(missing)}, but "
                    f'{model_cls.__name__}.{method_name}() produces {sorted(produced)}'
                )


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
        ValueError: if a configured loss requires a key the model type does not produce.
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
    _validate_loss_model_compatibility(ModelClass, loss_factories, semi_supervised)

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
    else:
        raise NotImplementedError(
            f'{cfg.model.model_type} is an invalid cfg.model.model_type'
        )

    model = ModelClass(**common, **extra)

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

    return model

"""Factory functions to build data pipeline components from a Hydra config.

Three public functions, typically called in order:

1. :func:`get_imgaug_transform` — builds an imgaug augmentation pipeline from
   ``cfg.training.imgaug``.
2. :func:`get_dataset` — wraps the labeled CSV data in the appropriate dataset class
   (regression, single-view heatmap, or multiview heatmap).
3. :func:`get_data_module` — wraps a dataset in a data module that handles train/val/test
   splitting; selects :class:`~lightning_pose.data.datamodules.UnlabeledDataModule` for
   semi-supervised training (adds DALI video loader) or
   :class:`~lightning_pose.data.datamodules.BaseDataModule` for supervised-only training.
"""

import warnings

import imgaug.augmenters as iaa
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import ValidationError

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

# to ignore imports for sphinx-autoapidoc
__all__ = [
    'get_imgaug_transform',
    'get_dataset',
    'get_data_module',
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

    params = cfg.training.get('imgaug', 'default')
    if isinstance(params, str):
        # Check if user explicitly wants to use 3D augmentations for multiview models
        imagug_3d = cfg.training.get('imgaug_3d', None)

        # enforce "dlc-mv" imgaug pipeline for multiview models (no 2D geometric transforms)
        # only if explicitly requested or if no preference is set and camera params exist
        if (
            params not in ['default', 'none']
            and cfg.model.model_type.find('multiview') > -1
            and cfg.data.get('camera_params_file')
            and (imagug_3d is True or imagug_3d is None)
        ):
            params = 'dlc-mv'
        params_dict = expand_imgaug_str_to_dict(params)
    elif isinstance(params, dict) or isinstance(params, DictConfig):
        if isinstance(params, DictConfig):
            # recursively convert Dict/ListConfigs to dicts/lists
            params_dict = OmegaConf.to_object(params)
            assert isinstance(params_dict, dict)
        else:
            params_dict = params.copy()
        for transform, _val in params_dict.items():
            assert getattr(iaa, str(transform)), f'{transform} is not a valid imgaug transform'
    else:
        raise TypeError(f'params is of type {type(params)}, must be str, dict, or DictConfig')

    return imgaug_transform(params_dict)  # type: ignore[arg-type]


def get_dataset(
    cfg: DictConfig | ListConfig,
    data_dir: str,
    imgaug_transform: iaa.Sequential,
) -> BaseTrackingDataset | HeatmapDataset | MultiviewHeatmapDataset:
    """Build a labeled dataset from a Hydra config.

    Dispatches on ``cfg.model.model_type``:
    - ``'regression'``: returns a :class:`~lightning_pose.data.datasets.BaseTrackingDataset`.
    - ``'heatmap*'`` with multiple views: returns a
      :class:`~lightning_pose.data.datasets.MultiviewHeatmapDataset`; ``resize`` is set to
      ``False`` only when imgaug is active *and* a camera-params file is provided (in that
      case the augmentation pipeline already handles resizing).
    - ``'heatmap*'`` single-view: returns a
      :class:`~lightning_pose.data.datasets.HeatmapDataset`.

    Args:
        cfg: Hydra config. Relevant fields: ``cfg.model.model_type``,
            ``cfg.data.csv_file``, ``cfg.data.image_resize_dims``,
            ``cfg.data.view_names``, ``cfg.data.downsample_factor``,
            ``cfg.data.camera_params_file``, ``cfg.data.bbox_file``.
        data_dir: root directory that ``csv_path`` is resolved relative to.
        imgaug_transform: augmentation pipeline produced by :func:`get_imgaug_transform`.

    Returns:
        dataset instance appropriate for the configured model type.

    Raises:
        NotImplementedError: if ``cfg.model.model_type`` is not a recognised value, or if
            a multi-view regression model is requested.
    """

    if cfg.model.model_type == 'regression':
        if cfg.data.get('view_names', None) and len(cfg.data.view_names) > 1:
            raise NotImplementedError('Multi-view support only available for heatmap-based models')
        else:
            dataset = BaseTrackingDataset(
                root_directory=data_dir,
                csv_path=cfg.data.csv_file,
                image_resize_height=cfg.data.image_resize_dims.height,
                image_resize_width=cfg.data.image_resize_dims.width,
                imgaug_transform=imgaug_transform,
                do_context=False,  # no context for regression models
                bbox_path=cfg.data.get('bbox_file', None),
            )
    elif cfg.model.model_type.find('heatmap') > -1:
        if cfg.data.get('view_names', None) and len(cfg.data.view_names) > 1:
            UserWarning(
                'No precautions regarding the size of the images were considered here, '
                'images will be resized accordingly to configs!'
            )
            if (
                cfg.training.imgaug in ['default', 'none']
                or not cfg.data.get('camera_params_file')
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
                downsample_factor=cfg.data.get('downsample_factor', 2),
                do_context=cfg.model.model_type == 'heatmap_mhcrnn',  # context only for mhcrnn
                resize=resize,
                uniform_heatmaps=cfg.training.get('uniform_heatmaps_for_nan_keypoints', False),
                camera_params_path=cfg.data.get('camera_params_file', None),
                bbox_paths=cfg.data.get('bbox_file', None),
            )
        else:
            dataset = HeatmapDataset(
                root_directory=data_dir,
                csv_path=cfg.data.csv_file,
                image_resize_height=cfg.data.image_resize_dims.height,
                image_resize_width=cfg.data.image_resize_dims.width,
                imgaug_transform=imgaug_transform,
                downsample_factor=cfg.data.get('downsample_factor', 2),
                do_context=cfg.model.model_type == 'heatmap_mhcrnn',  # context only for mhcrnn
                uniform_heatmaps=cfg.training.get('uniform_heatmaps_for_nan_keypoints', False),
                bbox_path=cfg.data.get('bbox_file', None),
            )

    else:
        raise NotImplementedError(f'{cfg.model.model_type} is an invalid cfg.model.model_type')

    return dataset


def get_data_module(
    cfg: DictConfig | ListConfig,
    dataset: BaseTrackingDataset | HeatmapDataset | MultiviewHeatmapDataset,
    video_dir: str | None = None,
) -> BaseDataModule | UnlabeledDataModule:
    """Build a data module that wraps a dataset with train/val/test splitting.

    For supervised models, returns a :class:`~lightning_pose.data.datamodules.BaseDataModule`.
    For semi-supervised models, returns an
    :class:`~lightning_pose.data.datamodules.UnlabeledDataModule` which adds a DALI-backed
    video loader for unsupervised losses.

    Batch sizes are divided by ``cfg.training.num_gpus`` so the effective per-step batch
    size stays constant regardless of GPU count. Context models receive special treatment:
    four frames are added back after dividing to preserve the two-frame context on each
    side of the centre frame.

    Args:
        cfg: Hydra config. Relevant fields: ``cfg.training.num_gpus``,
            ``cfg.training.train_batch_size``, ``cfg.training.val_batch_size``,
            ``cfg.training.test_batch_size``, ``cfg.training.num_workers``,
            ``cfg.training.train_prob``, ``cfg.training.val_prob``,
            ``cfg.training.train_frames``, ``cfg.training.rng_seed_data_pt``,
            ``cfg.model.losses_to_use``, ``cfg.dali``.
        dataset: labeled dataset produced by :func:`get_dataset`.
        video_dir: path to unlabeled video directory; required for semi-supervised training,
            ignored otherwise.

    Returns:
        data module ready to be passed to the PyTorch Lightning ``Trainer``.

    Raises:
        ValidationError: if a context model is requested but
            ``dali.context.train.batch_size < 5 * num_gpus``.
    """

    # Old configs may have num_gpus: 0. We will remove support in a future release.
    if cfg.training.num_gpus == 0:
        warnings.warn(
            'Config contains unsupported value num_gpus: 0. '
            'Update num_gpus to 1 in your config.',
            stacklevel=2,
        )
    cfg.training.num_gpus = max(cfg.training.num_gpus, 1)

    # Divide config batch_size by num_gpus to maintain the same effective batch
    # size in a multi-gpu setting.
    train_batch_size = int(
        np.ceil(cfg.training.train_batch_size / cfg.training.num_gpus)
    )
    val_batch_size = int(np.ceil(cfg.training.val_batch_size / cfg.training.num_gpus))

    from lightning_pose.models.base import check_if_semi_supervised
    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    if not semi_supervised:
        data_module = BaseDataModule(
            dataset=dataset,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.get('num_workers'),
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

        if cfg.model.model_type == 'heatmap_mhcrnn' and context_batch_size < 5:
            raise ValidationError(
                'dali.context.train.batch_size must be >= 5 * num_gpus for '
                'semi-supervised context models. '
                'Found {cfg.dali.context.train.batch_size}'
            )

        dali_config = OmegaConf.merge(
            cfg.dali,
            {
                'base': {'train': {'sequence_length': base_sequence_length}},
                'context': {'train': {'batch_size': context_batch_size}},
            },
        )

        assert video_dir is not None, 'video_dir must be provided for semi-supervised training'
        view_names = cfg.data.get('view_names', None)
        view_names = list(view_names) if view_names is not None else None
        data_module = UnlabeledDataModule(
            dataset=dataset,
            video_paths_list=video_dir,
            view_names=view_names,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.get('num_workers'),
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
            dali_config=dali_config,
            torch_seed=cfg.training.rng_seed_data_pt,
            imgaug=cfg.training.get('imgaug', 'default'),
        )
    return data_module

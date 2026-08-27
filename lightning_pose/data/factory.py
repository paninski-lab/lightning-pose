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

**Adding a new model type** (data-side changes only — see ``models/factory.py`` for the
model-side steps):

1. If the new type can reuse an existing dataset class (e.g. it is a heatmap variant),
   extend the appropriate ``elif`` branch in :func:`get_dataset` to match the new
   ``cfg.model.model_type`` string.  If it needs a new dataset class, define that class
   in ``datasets.py``, import it here, and add a new ``elif`` branch.
2. If the new type needs a different :class:`~lightning_pose.data.datamodules.BaseDataModule`
   subclass, add a branch in :func:`get_data_module`; otherwise no change is needed there.
"""

import copy
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
__all__: list[str] = []


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

    return imgaug_transform(_imgaug_params_dict(cfg))


def _imgaug_params_dict(cfg: DictConfig | ListConfig) -> dict:
    """Resolve ``cfg.training.imgaug`` to a plain transform-parameters dictionary.

    Shared by :func:`get_imgaug_transform` (which builds the single global pipeline) and
    :func:`get_imgaug_transforms_per_dataset` (which builds per-dataset variants of it).

    Args:
        cfg: standard config file; see :func:`get_imgaug_transform` for the accepted
            ``cfg.training.imgaug`` values.

    Returns:
        dictionary mapping imgaug transform names to their parameter dicts.

    Raises:
        TypeError: if ``cfg.training.imgaug`` is not a str, dict, or DictConfig.
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

    return params_dict  # type: ignore[return-value]


def get_imgaug_transforms_per_dataset(
    cfg: DictConfig | ListConfig,
) -> dict[str, iaa.Sequential] | None:
    """Build per-dataset variants of the imgaug pipeline with dataset-specific zoom-out bounds.

    Reads ``cfg.training.imgaug_per_dataset_zoom``, a mapping from dataset name (an entry of
    ``cfg.data.dataset_names``) to the upper bound of the CropAndPad ``percent`` range for that
    dataset, or to a ``[lower, upper]`` pair that also sets the zoom-in (crop) bound. Each listed dataset gets a copy of the base pipeline (``cfg.training.imgaug``) whose
    CropAndPad upper bound is replaced by its own value; the lower bound, probability, and other
    CropAndPad kwargs are inherited from the base pipeline's CropAndPad entry when present,
    otherwise default to ``percent=(-0.15, <bound>)``, ``keep_size=True``, ``p=0.4``. Datasets
    absent from the mapping fall back to the global pipeline at load time.

    The motivation is scale-gap bridging in multi-dataset corpora: each source dataset should be
    zoomed out only as far as needed to reach the smallest apparent scale in the corpus, rather
    than all datasets sharing one (over-)aggressive range.

    Args:
        cfg: standard config file; relevant parameters are ``cfg.training.imgaug_per_dataset_zoom``
            (mapping described above; absent/empty disables the feature),
            ``cfg.training.imgaug`` (the base pipeline), and ``cfg.data.dataset_names``.

    Returns:
        mapping from dataset name to its pipeline, or None when the feature is not configured.

    Raises:
        ValueError: if the mapping is set without ``cfg.data.dataset_names``, references a name
            missing from it, or a zoom bound does not exceed the CropAndPad lower bound.
    """
    zoom_cfg = cfg.training.get('imgaug_per_dataset_zoom', None)
    if not zoom_cfg:
        return None
    zoom = OmegaConf.to_object(zoom_cfg) if isinstance(zoom_cfg, DictConfig) else dict(zoom_cfg)
    dataset_names = cfg.data.get('dataset_names', None)
    if not dataset_names:
        raise ValueError(
            'training.imgaug_per_dataset_zoom requires data.dataset_names to be set'
        )
    unknown = set(zoom) - set(dataset_names)
    if unknown:
        raise ValueError(
            f'training.imgaug_per_dataset_zoom names {sorted(unknown)} not present in '
            f'data.dataset_names {list(dataset_names)}'
        )
    params_dict_base = _imgaug_params_dict(cfg)
    pipelines = {}
    for name, bound in zoom.items():
        params_dict = copy.deepcopy(params_dict_base)
        entry = params_dict.get(
            'CropAndPad', {'p': 0.4, 'kwargs': {'keep_size': True}},
        )
        kwargs = entry.setdefault('kwargs', {})
        percent = kwargs.get('percent', (-0.15, None))
        # a scalar is the zoom-out (upper) bound with the base lower bound; a [lower, upper]
        # pair also sets the zoom-in bound (negative percent = crop = enlarge)
        if isinstance(bound, (list, tuple)):
            if len(bound) != 2:
                raise ValueError(f'imgaug_per_dataset_zoom[{name}] pair must be [lower, upper]')
            lower, upper = float(bound[0]), float(bound[1])
        else:
            lower, upper = float(percent[0]), float(bound)
        if upper <= lower:
            raise ValueError(
                f'imgaug_per_dataset_zoom[{name}]: upper bound {upper} must exceed the '
                f'lower bound {lower}'
            )
        kwargs['percent'] = (lower, upper)
        params_dict['CropAndPad'] = entry
        pipelines[name] = imgaug_transform(params_dict)
    return pipelines


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

    imgaug_hflip = bool(cfg.training.get('imgaug_hflip', False))
    imgaug_transform_per_dataset = get_imgaug_transforms_per_dataset(cfg)

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
                imgaug_hflip=imgaug_hflip,
                dataset_names=cfg.data.get('dataset_names', None),
                imgaug_transform_per_dataset=imgaug_transform_per_dataset,
            )
    elif cfg.model.model_type.find('heatmap') > -1:
        if cfg.data.get('view_names', None) and len(cfg.data.view_names) > 1:
            if imgaug_hflip:
                raise ValueError(
                    'imgaug_hflip is not supported for multiview models'
                )
            if imgaug_transform_per_dataset is not None:
                raise NotImplementedError(
                    'imgaug_per_dataset_zoom is not supported for multiview models'
                )
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
                imgaug_hflip=imgaug_hflip,
                dataset_names=cfg.data.get('dataset_names', None),
                imgaug_transform_per_dataset=imgaug_transform_per_dataset,
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

    # multi-dataset sampling temperature; the string 'inf' (Hydra has no float-inf
    # literal) maps to dataset-uniform supervision shares
    sampling_temperature = cfg.training.get('sampling_temperature', None)
    if isinstance(sampling_temperature, str):
        sampling_temperature = float(sampling_temperature)

    from lightning_pose.models import check_if_semi_supervised
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
            sampling_temperature=sampling_temperature,
        )
    else:
        if sampling_temperature is not None and sampling_temperature != 1:
            raise NotImplementedError(
                'sampling_temperature is not supported for semi-supervised models: the '
                'unlabeled video loader has no dataset identity to route by'
            )
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

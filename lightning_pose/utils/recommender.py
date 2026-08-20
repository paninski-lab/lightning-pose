"""Dataset analysis and config recommendation logic backing ``litpose recommend``.

Given a directory (or single CSV file) of DLC-format labeled frames, this module inspects
the dataset and available hardware to recommend a starting-point lightning-pose config:
model type, backbone, image resize dimensions, batch size, epoch budget, augmentation
pipeline, optimizer, and unsupervised losses.

Pipeline: :func:`analyze_dataset` + :func:`get_gpu_info` gather raw facts, :func:`recommend`
turns those facts into a :class:`ConfigRecommendation` (with a human-readable rationale per
field), and :func:`build_config` / :func:`format_report` render that recommendation as a
complete config YAML or a printable report, respectively.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from lightning_pose.utils.io import parse_label_csv

logger = logging.getLogger(__name__)

# per-side image_resize_dims recommendation: 256 by default; 128 for a short side (< the
# small-side threshold); 384 only for a long side (> the large-side threshold) that also has
# enough labeled frames (> the large-side frame threshold) to justify the extra resolution
_RESIZE_SMALL = 128
_RESIZE_DEFAULT = 256
_RESIZE_LARGE = 384
_RESIZE_SMALL_SIDE_THRESHOLD_PX = 192
_RESIZE_LARGE_SIDE_THRESHOLD_PX = 1024
_RESIZE_LARGE_SIDE_THRESHOLD_FRAMES = 500

# train_batch_size lookup for the base (single-view, fully-supervised, non-context) vits_dino
# heatmap model: {min gpu vram (gb): {max image size tier (px): batch size}}. measured
# empirically (not guessed): forward + backward + AdamW optimizer.step() peak memory was
# swept over several batch sizes at each resolution with the backbone fully unfrozen (the
# worst-case memory state, reached once training passes `unfreezing_epoch`), fit to a linear
# model in batch size (R^2 fit residual < 1% at every point), then extrapolated to each vram
# budget at a 90% safety margin and rounded down to a multiple of 4. context and multiview
# models divide this base number further -- see `_CONTEXT_BATCH_SIZE_DIVISOR` and the
# per-num-views division in `recommend()`.
_BATCH_SIZE_TABLE = {
    24: {256: 232, 384: 100},
    16: {256: 156, 384: 68},
    8: {256: 76, 384: 32},
}
_NO_GPU_BATCH_SIZE = 4

# a context (temporal, e.g. heatmap_mhcrnn) model stacks 5 frames per sample, which multiplies
# per-sample activation memory roughly 5x relative to the base model (fixed weight memory
# aside); divide the base train_batch_size recommendation by this factor for context models
_CONTEXT_BATCH_SIZE_DIVISOR = 5

# semi-supervised training (temporal/pca losses over unlabeled video) loads a DALI batch of
# unlabeled frames alongside each labeled batch; halve the labeled train_batch_size to make
# room for it, then use that halved number as the unlabeled dali sequence length. context
# models process their whole unlabeled sequence window in one pass rather than a 5x-stacked
# per-sample volume, so their memory scales ~1x per frame instead of ~5x and can afford a
# longer sequence -- `_CONTEXT_SEQUENCE_LENGTH_MULTIPLIER` widens it accordingly
_SEMI_SUPERVISED_BATCH_SIZE_DIVISOR = 2
_CONTEXT_SEQUENCE_LENGTH_MULTIPLIER = 4

# final guardrail applied to train_batch_size only (dali sequence lengths are left as
# computed above): a large GPU can recommend a batch size that swallows an entire modest
# dataset in 1-2 iterations/epoch, which starves the lr scheduler and per-step logging. round
# down to a multiple of this...
_BATCH_SIZE_ROUND_TO = 8
# ...never go below this...
_MIN_TRAIN_BATCH_SIZE = 8
# ...and keep shrinking (by _BATCH_SIZE_ROUND_TO) until the train split clears this many
# iterations/epoch, or until _MIN_TRAIN_BATCH_SIZE is hit
_MIN_TRAIN_ITERS_PER_EPOCH = 10
# train_prob default from config_default.yaml / config_default_multiview.yaml
_DEFAULT_TRAIN_PROB = 0.95

_DALI_DEFAULTS = {
    'base': {
        'train': {'sequence_length': 32},
        'predict': {'sequence_length': 96},
    },
    'context': {
        'train': {'batch_size': 16},
        'predict': {'sequence_length': 96},
    },
}

_LOSSES_DEFAULTS = {
    'pca_multiview': {
        'log_weight': 11.0,
        'components_to_keep': 3,
        'epsilon': None,
    },
    'pca_singleview': {
        'log_weight': 11.0,
        'components_to_keep': 0.99,
        'epsilon': None,
    },
    'temporal': {
        'log_weight': 11.0,
        'epsilon': 20.0,
        'prob_threshold': 0.05,
    },
}

_CALLBACKS_DEFAULTS = {
    'anneal_weight': {
        'attr_name': 'total_unsupervised_importance',
        'init_val': 0.0,
        'increase_factor': 0.01,
        'final_val': 1.0,
        'freeze_until_epoch': 60,
    },
}

_HYDRA_DEFAULTS = {
    'run': {'dir': 'outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'},
    'sweep': {
        'dir': 'multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}',
        'subdir': '${hydra.job.num}',
    },
}


@dataclass
class DatasetAnalysis:
    """Raw facts gathered from a lightning-pose dataset directory.

    Attributes:
        dataset_path: the original path passed by the caller (directory or CSV file)
        data_dir: directory containing the label CSV(s); equals ``dataset_path`` when a
            directory was passed, else its parent
        csv_paths: ordered list of label CSV file paths
        view_names: derived per-view names, or ``None`` for a single-view dataset
        n_frames: number of labeled frames (rows) in the first CSV
        num_keypoints: number of keypoints in the first CSV
        keypoint_names: ordered keypoint names from the first CSV
        image_height: pixel height of a sample labeled image
        image_width: pixel width of a sample labeled image
        video_dir: assumed location of unlabeled videos (``data_dir / 'videos'``)
        has_videos: whether ``video_dir`` exists and contains at least one ``.mp4`` file
    """

    dataset_path: Path
    data_dir: Path
    csv_paths: list[Path]
    view_names: list[str] | None
    n_frames: int
    num_keypoints: int
    keypoint_names: list[str]
    image_height: int
    image_width: int
    video_dir: Path
    has_videos: bool


@dataclass
class GpuInfo:
    """Name and VRAM capacity of the local CUDA device used for sizing recommendations."""

    name: str
    vram_gb: float


@dataclass
class ConfigRecommendation:
    """Recommended config field values, with a rationale string per field.

    Attributes:
        model_type: `heatmap` | `heatmap_multiview_transformer`
        backbone: backbone identifier (see `model.backbone` in the default config)
        image_resize_height: recommended `data.image_resize_dims.height`
        image_resize_width: recommended `data.image_resize_dims.width`
        train_batch_size: recommended `training.train_batch_size`
        max_epochs: recommended `training.max_epochs` (and `min_epochs`)
        optimizer: `Adam` | `AdamW`
        imgaug: `dlc` | `dlc-top-down`
        losses_to_use: recommended `model.losses_to_use`
        dali_train_sequence_length: recommended unlabeled-frame batch size for semi-supervised
            training (`dali.base.train.sequence_length`, or `dali.context.train.batch_size`
            for a context model), or `None` when `losses_to_use` is empty
        rationale: field name -> one-line explanation of the recommendation
    """

    model_type: str
    backbone: str
    image_resize_height: int
    image_resize_width: int
    train_batch_size: int
    max_epochs: int
    optimizer: str
    imgaug: str
    losses_to_use: list[str]
    dali_train_sequence_length: int | None = None
    rationale: dict[str, str] = field(default_factory=dict)


def _derive_view_names(csv_paths: list[Path]) -> list[str]:
    """derive short view names from csv filenames by stripping tokens common to all stems

    stems are split on non-alphanumeric delimiters into tokens; leading and trailing tokens
    shared by every stem are dropped, and the remaining tokens are rejoined with '_'. falls
    back to the full stem for every file if this produces an empty or duplicate name.
    """
    stems = [p.stem for p in csv_paths]
    token_lists = [re.split(r'[_\-.\s]+', s) for s in stems]
    min_len = min(len(tokens) for tokens in token_lists)

    n_leading = 0
    for i in range(min_len):
        if len({tokens[i] for tokens in token_lists}) == 1:
            n_leading += 1
        else:
            break

    n_trailing = 0
    for i in range(1, min_len - n_leading + 1):
        if len({tokens[-i] for tokens in token_lists}) == 1:
            n_trailing += 1
        else:
            break

    trimmed = [
        tokens[n_leading: len(tokens) - n_trailing] if n_trailing else tokens[n_leading:]
        for tokens in token_lists
    ]
    view_names = ['_'.join(tokens) for tokens in trimmed]

    if any(name == '' for name in view_names) or len(set(view_names)) != len(view_names):
        return stems

    return view_names


def analyze_dataset(dataset_path: Path) -> DatasetAnalysis:
    """Analyze a lightning-pose dataset directory or CSV file.

    Args:
        dataset_path: a directory containing one or more DLC-format label CSVs (auto-discovered,
            multiple CSVs are treated as a multi-view dataset), or a single label CSV file.

    Returns:
        :class:`DatasetAnalysis` summarizing the dataset.

    Raises:
        FileNotFoundError: if `dataset_path` does not exist, contains no CSV files, or the
            first labeled image referenced by the CSV cannot be found.
    """
    dataset_path = Path(dataset_path)
    if dataset_path.is_file():
        csv_paths = [dataset_path]
        data_dir = dataset_path.parent
    elif dataset_path.is_dir():
        csv_paths = sorted(dataset_path.glob('*.csv'))
        data_dir = dataset_path
    else:
        raise FileNotFoundError(f'dataset path does not exist: {dataset_path}')

    if not csv_paths:
        raise FileNotFoundError(f'no label CSV files found in {dataset_path}')

    view_names = _derive_view_names(csv_paths) if len(csv_paths) > 1 else None

    labeled_data = [parse_label_csv(str(p)) for p in csv_paths]
    keypoint_names = labeled_data[0].keypoint_names
    num_keypoints = len(keypoint_names)
    n_frames = len(labeled_data[0].image_names)

    view_labels = view_names if view_names is not None else [csv_paths[0].stem]
    for view_name, ld, csv_path in zip(view_labels, labeled_data, csv_paths, strict=True):
        if len(ld.keypoint_names) != num_keypoints:
            logger.warning(
                f'view {view_name!r} ({csv_path.name}) has {len(ld.keypoint_names)} keypoints, '
                f'expected {num_keypoints} (from {csv_paths[0].name}); '
                'using keypoint names from the first view'
            )
        if len(ld.image_names) != n_frames:
            logger.warning(
                f'view {view_name!r} ({csv_path.name}) has {len(ld.image_names)} labeled frames, '
                f'expected {n_frames} (from {csv_paths[0].name})'
            )

    sample_image_path = data_dir / labeled_data[0].image_names[0]
    if not sample_image_path.exists():
        raise FileNotFoundError(f'labeled image not found: {sample_image_path}')
    with Image.open(sample_image_path) as img:
        image_width, image_height = img.size

    video_dir = data_dir / 'videos'
    has_videos = video_dir.is_dir() and any(video_dir.glob('*.mp4'))

    return DatasetAnalysis(
        dataset_path=dataset_path,
        data_dir=data_dir,
        csv_paths=csv_paths,
        view_names=view_names,
        n_frames=n_frames,
        num_keypoints=num_keypoints,
        keypoint_names=keypoint_names,
        image_height=image_height,
        image_width=image_width,
        video_dir=video_dir,
        has_videos=has_videos,
    )


def get_gpu_info() -> GpuInfo | None:
    """Return name and VRAM capacity of CUDA device 0, or `None` if no CUDA device is available."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return GpuInfo(name=props.name, vram_gb=props.total_memory / (1024 ** 3))


def _recommend_resize_dim(side_px: int, n_frames: int) -> int:
    """recommend a resize dimension for one side of the image

    256 by default; 128 if the side is shorter than `_RESIZE_SMALL_SIDE_THRESHOLD_PX`; 384
    only if the side is longer than `_RESIZE_LARGE_SIDE_THRESHOLD_PX` and there are more than
    `_RESIZE_LARGE_SIDE_THRESHOLD_FRAMES` labeled frames to justify the extra resolution.
    """
    if side_px < _RESIZE_SMALL_SIDE_THRESHOLD_PX:
        return _RESIZE_SMALL
    if (
        side_px > _RESIZE_LARGE_SIDE_THRESHOLD_PX
        and n_frames > _RESIZE_LARGE_SIDE_THRESHOLD_FRAMES
    ):
        return _RESIZE_LARGE
    return _RESIZE_DEFAULT


def _select_batch_size(gpu: GpuInfo | None, resize_dim: int) -> int:
    """pick a train batch size from the gpu-vram x image-size lookup table"""
    if gpu is None:
        return _NO_GPU_BATCH_SIZE
    size_tier = 256 if resize_dim <= 256 else 384
    for vram_tier in (24, 16, 8):
        if gpu.vram_gb >= vram_tier:
            return _BATCH_SIZE_TABLE[vram_tier][size_tier]
    return _NO_GPU_BATCH_SIZE


def _apply_min_iterations_floor(train_batch_size: int, n_frames: int) -> int:
    """round `train_batch_size` down to a multiple of `_BATCH_SIZE_ROUND_TO`, then keep
    shrinking it by that amount (never below `_MIN_TRAIN_BATCH_SIZE`) until the labeled train
    split clears `_MIN_TRAIN_ITERS_PER_EPOCH` iterations/epoch.
    """
    batch_size = max(
        _MIN_TRAIN_BATCH_SIZE, (train_batch_size // _BATCH_SIZE_ROUND_TO) * _BATCH_SIZE_ROUND_TO
    )
    n_train_frames = n_frames * _DEFAULT_TRAIN_PROB
    while (
        batch_size > _MIN_TRAIN_BATCH_SIZE
        and n_train_frames / batch_size < _MIN_TRAIN_ITERS_PER_EPOCH
    ):
        batch_size -= _BATCH_SIZE_ROUND_TO
    return batch_size


def recommend(
    analysis: DatasetAnalysis,
    gpu: GpuInfo | None,
    top_down_freely_moving: bool = False,
) -> ConfigRecommendation:
    """Derive config recommendations from dataset analysis and GPU info.

    Args:
        analysis: result of :func:`analyze_dataset`
        gpu: result of :func:`get_gpu_info`, or `None` if no GPU is available
        top_down_freely_moving: whether the dataset is a top-down view of a freely moving
            animal; selects the `dlc-top-down` augmentation pipeline and a longer epoch budget

    Returns:
        :class:`ConfigRecommendation` with a rationale string for each recommended field.
    """
    rationale: dict[str, str] = {}
    is_multiview = analysis.view_names is not None

    model_type = 'heatmap_multiview_transformer' if is_multiview else 'heatmap'
    rationale['model_type'] = (
        f'{len(analysis.view_names)} views detected -> multiview transformer'
        if is_multiview
        else 'single view detected -> standard heatmap model '
        '(use heatmap_mhcrnn instead if temporal context is desired)'
    )

    backbone = 'vits_dino'
    rationale['backbone'] = 'vits_dino is a strong default across dataset sizes'

    image_resize_height = _recommend_resize_dim(analysis.image_height, analysis.n_frames)
    image_resize_width = _recommend_resize_dim(analysis.image_width, analysis.n_frames)
    resize_rationale = (
        f'source images are {analysis.image_width}x{analysis.image_height}px '
        f'({analysis.n_frames} labeled frames); {_RESIZE_DEFAULT}px by default per side, '
        f'{_RESIZE_SMALL}px for a side under {_RESIZE_SMALL_SIDE_THRESHOLD_PX}px, '
        f'{_RESIZE_LARGE}px for a side over {_RESIZE_LARGE_SIDE_THRESHOLD_PX}px with more than '
        f'{_RESIZE_LARGE_SIDE_THRESHOLD_FRAMES} labeled frames'
    )
    if backbone.startswith('vit') and image_resize_height != image_resize_width:
        # ViT backbones require square input; the larger of the two independently-computed
        # sides wins so neither side ends up smaller than what its own rule recommended
        square_dim = max(image_resize_height, image_resize_width)
        resize_rationale += (
            f'; height/width computed independently gave {image_resize_width}x'
            f'{image_resize_height}, but vits_dino requires a square input, so both are set '
            f'to {square_dim}'
        )
        image_resize_height = square_dim
        image_resize_width = square_dim
    rationale['image_resize_dims'] = resize_rationale

    is_context = model_type == 'heatmap_mhcrnn'  # never auto-recommended today, kept for
    # correctness in case a caller overrides model_type before re-deriving batch size

    full_batch_size = _select_batch_size(gpu, max(image_resize_height, image_resize_width))
    batch_size_rationale = (
        f'base vits_dino model sized for {gpu.name} ({gpu.vram_gb:.1f} GB) at '
        f'{image_resize_width}x{image_resize_height}px'
        if gpu is not None
        else 'no GPU detected, using a conservative default'
    )
    if is_context:
        full_batch_size = max(1, full_batch_size // _CONTEXT_BATCH_SIZE_DIVISOR)
        batch_size_rationale += (
            f'; divided by {_CONTEXT_BATCH_SIZE_DIVISOR} for the context model, whose '
            'per-sample activation memory is roughly that much higher than the base model'
        )
    elif is_multiview:
        num_views = len(analysis.view_names)
        full_batch_size = max(1, full_batch_size // num_views)
        batch_size_rationale += (
            f'; divided by {num_views} views, since each training step pushes '
            'batch_size * num_views images through the backbone'
        )

    imgaug = 'dlc-top-down' if top_down_freely_moving else 'dlc'
    rationale['imgaug'] = (
        'top-down freely-moving setup -> dlc-top-down '
        '(adds 90/180/270-degree rotations on top of the dlc pipeline)'
        if top_down_freely_moving
        else 'dlc is the standard augmentation pipeline'
    )

    if analysis.n_frames < 50 and top_down_freely_moving:
        max_epochs = 1000
        rationale['max_epochs'] = (
            f'{analysis.n_frames} labeled frames (<50) with dlc-top-down augmentation -> '
            'more epochs needed to converge'
        )
    elif analysis.n_frames < 50 or top_down_freely_moving:
        max_epochs = 500
        rationale['max_epochs'] = (
            f'{analysis.n_frames} labeled frames and/or dlc-top-down augmentation -> '
            'more epochs needed to converge'
        )
    else:
        max_epochs = 300
        rationale['max_epochs'] = f'{analysis.n_frames} labeled frames -> standard epoch budget'

    losses_to_use: list[str] = []
    if analysis.has_videos and not is_multiview:
        min_frames_for_pca = 2 * analysis.num_keypoints
        if analysis.n_frames >= min_frames_for_pca:
            losses_to_use = ['temporal', 'pca_singleview']
            rationale['losses_to_use'] = (
                'unlabeled videos found, and enough labeled frames '
                f'(>= {min_frames_for_pca} = 2 x {analysis.num_keypoints} keypoints) '
                'to estimate a pca_singleview subspace'
            )
        else:
            losses_to_use = ['temporal']
            rationale['losses_to_use'] = (
                'unlabeled videos found, but too few labeled frames to estimate a '
                f'pca_singleview subspace (need >= {min_frames_for_pca}); omitting pca_singleview'
            )
    elif analysis.has_videos and is_multiview:
        rationale['losses_to_use'] = (
            'unlabeled videos found, but multiview unsupervised losses (pca_multiview, '
            'supervised_reprojection_heatmap_mse) require a camera calibration file and are '
            'not auto-recommended; add them manually if applicable'
        )
    else:
        rationale['losses_to_use'] = 'no unlabeled videos found -> fully supervised training'

    dali_train_sequence_length: int | None = None
    if losses_to_use:
        train_batch_size = max(1, full_batch_size // _SEMI_SUPERVISED_BATCH_SIZE_DIVISOR)
        if is_context:
            dali_train_sequence_length = _CONTEXT_SEQUENCE_LENGTH_MULTIPLIER * train_batch_size
        else:
            dali_train_sequence_length = train_batch_size
        batch_size_rationale += (
            f'; halved to {train_batch_size} to leave room for the unlabeled DALI batch '
            'loaded alongside it each training step (dali_train_sequence_length='
            f'{dali_train_sequence_length})'
        )
    else:
        train_batch_size = full_batch_size

    # final guardrail: shrink train_batch_size (never dali_train_sequence_length) so a large
    # GPU doesn't reduce a modest dataset to a couple of iterations per epoch
    floored_batch_size = _apply_min_iterations_floor(train_batch_size, analysis.n_frames)
    if floored_batch_size != train_batch_size:
        batch_size_rationale += (
            f'; rounded to a multiple of {_BATCH_SIZE_ROUND_TO} and reduced to '
            f'{floored_batch_size} (minimum {_MIN_TRAIN_BATCH_SIZE}) so the '
            f'{analysis.n_frames} labeled frames (x {_DEFAULT_TRAIN_PROB} train split) clear '
            f'{_MIN_TRAIN_ITERS_PER_EPOCH} training iterations per epoch'
        )
    train_batch_size = floored_batch_size
    rationale['train_batch_size'] = batch_size_rationale

    optimizer = 'AdamW' if backbone.startswith('vit') else 'Adam'
    rationale['optimizer'] = (
        'AdamW is recommended for ViT backbones'
        if optimizer == 'AdamW'
        else 'Adam is recommended for ResNet backbones'
    )

    return ConfigRecommendation(
        model_type=model_type,
        backbone=backbone,
        image_resize_height=image_resize_height,
        image_resize_width=image_resize_width,
        train_batch_size=train_batch_size,
        max_epochs=max_epochs,
        optimizer=optimizer,
        imgaug=imgaug,
        losses_to_use=losses_to_use,
        dali_train_sequence_length=dali_train_sequence_length,
        rationale=rationale,
    )


def build_config(rec: ConfigRecommendation, analysis: DatasetAnalysis) -> DictConfig:
    """Assemble a complete, ready-to-use lightning-pose config from a recommendation.

    Fields not covered by :func:`recommend` (e.g. `dali`, loss hyperparameters, `hydra`) are
    filled with the same defaults used in `scripts/configs/config_default.yaml`.

    Args:
        rec: recommendation produced by :func:`recommend`
        analysis: dataset analysis the recommendation was derived from

    Returns:
        `DictConfig` matching the structure of `scripts/configs/config_default.yaml`
        (or `config_default_multiview.yaml` for multi-view datasets).
    """
    is_multiview = analysis.view_names is not None

    data: dict = {
        'image_resize_dims': {
            'height': rec.image_resize_height,
            'width': rec.image_resize_width,
        },
        'data_dir': str(analysis.data_dir),
        'video_dir': str(analysis.video_dir),
        'num_keypoints': analysis.num_keypoints,
        'keypoint_names': list(analysis.keypoint_names),
        'mirrored_column_matches': None,
        'columns_for_singleview_pca': None,
    }
    if is_multiview:
        data['csv_file'] = [str(p) for p in analysis.csv_paths]
        data['view_names'] = list(analysis.view_names)
    else:
        data['csv_file'] = str(analysis.csv_paths[0])

    training: dict = {
        'imgaug': rec.imgaug,
        'imgaug_hflip': False,
        'train_batch_size': rec.train_batch_size,
        'val_batch_size': 2 * rec.train_batch_size,
        'test_batch_size': 2 * rec.train_batch_size,
        'train_prob': _DEFAULT_TRAIN_PROB,
        'val_prob': 0.05,
        'train_frames': 1,
        'num_gpus': 1,
        'unfreezing_epoch': 20,
        'min_epochs': rec.max_epochs,
        'max_epochs': rec.max_epochs,
        'log_every_n_steps': 10,
        'check_val_every_n_epoch': 5,
        'ckpt_every_n_epochs': None,
        'early_stopping': False,
        'early_stop_patience': 3,
        'rng_seed_data_pt': 0,
        'rng_seed_model_pt': 0,
        'optimizer': rec.optimizer,
        'optimizer_params': {'learning_rate': 5e-5 if is_multiview else 1e-3},
        'lr_scheduler': 'multisteplr',
        'lr_scheduler_params': {
            'multisteplr': {'milestones': [150, 200, 250], 'gamma': 0.5},
        },
        'uniform_heatmaps_for_nan_keypoints': True,
    }
    if is_multiview:
        training['imgaug_3d'] = True
        training['patch_mask'] = {
            'init_epoch': 40,
            'final_epoch': 300,
            'init_ratio': 0.0,
            'final_ratio': 0.5,
        }

    model = {
        'losses_to_use': list(rec.losses_to_use),
        'backbone': rec.backbone,
        'model_type': rec.model_type,
        'heatmap_loss_type': 'mse',
        'model_name': 'test',
        'checkpoint': None,
    }

    losses = {name: dict(hparams) for name, hparams in _LOSSES_DEFAULTS.items()}
    if is_multiview:
        losses['supervised_reprojection_heatmap_mse'] = {'log_weight': 3.0}

    dali_cfg = copy.deepcopy(_DALI_DEFAULTS)
    if rec.dali_train_sequence_length is not None:
        if rec.model_type == 'heatmap_mhcrnn':
            dali_cfg['context']['train']['batch_size'] = rec.dali_train_sequence_length
        else:
            dali_cfg['base']['train']['sequence_length'] = rec.dali_train_sequence_length

    eval_cfg = {
        'predict_vids_after_training': True,
        'test_videos_directory': '${data.video_dir}',
        'save_vids_after_training': False,
        'colormap': 'cool',
        'confidence_thresh_for_vid': 0.90,
    }

    cfg_dict = {
        'data': data,
        'training': training,
        'model': model,
        'dali': dali_cfg,
        'losses': losses,
        'eval': eval_cfg,
        'callbacks': _CALLBACKS_DEFAULTS,
        'hydra': _HYDRA_DEFAULTS,
    }

    return OmegaConf.create(cfg_dict)


def format_report(
    rec: ConfigRecommendation,
    analysis: DatasetAnalysis,
    gpu: GpuInfo | None,
) -> str:
    """Render a human-readable dataset summary and recommendation report.

    Args:
        rec: recommendation produced by :func:`recommend`
        analysis: dataset analysis the recommendation was derived from
        gpu: result of :func:`get_gpu_info`, or `None` if no GPU is available

    Returns:
        multi-line report string suitable for printing to the console.
    """
    lines = ['Dataset summary', '-' * 16]
    lines.append(f'  labeled frames:   {analysis.n_frames}')
    lines.append(f'  keypoints:        {analysis.num_keypoints}')
    lines.append(f'  image size:       {analysis.image_width}x{analysis.image_height}px')
    if analysis.view_names is not None:
        lines.append(f'  views:            {", ".join(analysis.view_names)}')
    lines.append(
        f'  unlabeled videos: {"found" if analysis.has_videos else "none found"} '
        f'({analysis.video_dir})'
    )
    lines.append(
        f'  gpu:              {gpu.name} ({gpu.vram_gb:.1f} GB)'
        if gpu is not None
        else '  gpu:              none detected'
    )

    lines += ['', 'Recommendations', '-' * 16]
    fields = [
        ('model_type', rec.model_type),
        ('backbone', rec.backbone),
        ('image_resize_dims', f'{rec.image_resize_width}x{rec.image_resize_height}'),
        ('train_batch_size', rec.train_batch_size),
        ('max_epochs', rec.max_epochs),
        ('imgaug', rec.imgaug),
        ('optimizer', rec.optimizer),
        ('losses_to_use', rec.losses_to_use if rec.losses_to_use else '[]'),
    ]
    for name, value in fields:
        lines.append(f'  {name}: {value}')
        if name in rec.rationale:
            lines.append(f'    -> {rec.rationale[name]}')
    if rec.dali_train_sequence_length is not None:
        lines.append(f'  dali_train_sequence_length: {rec.dali_train_sequence_length}')

    return '\n'.join(lines)

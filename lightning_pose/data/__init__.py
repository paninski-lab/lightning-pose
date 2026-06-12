"""Data loading, preprocessing, and augmentation for pose estimation.

**Three-stage factory pipeline** (the main entry points for all callers):

1. :func:`get_imgaug_transform` — build an ``imgaug`` augmentation pipeline from config.
2. :func:`get_dataset` — build a labeled dataset that pairs images with CSV keypoints.
3. :func:`get_data_module` — build a Lightning ``DataModule`` (labeled dataset +
   optional unlabeled video loader for semi-supervised training).

All three dispatch on ``cfg.model.model_type`` to select the right class.

**Module layout**:

*Pipeline stages* — what most callers interact with:

- ``factory.py`` — the three factory functions above.
- ``datasets.py`` — :class:`~lightning_pose.data.datasets.BaseTrackingDataset`,
  :class:`~lightning_pose.data.datasets.HeatmapDataset`,
  :class:`~lightning_pose.data.datasets.MultiviewHeatmapDataset`.
- ``datamodules.py`` — :class:`~lightning_pose.data.datamodules.BaseDataModule`,
  :class:`~lightning_pose.data.datamodules.UnlabeledDataModule`.
- ``augmentations.py`` — imgaug transform construction helpers.
- ``datatypes.py`` — TypedDict batch-dict types used throughout the package
  (e.g. ``HeatmapLabeledBatchDict``, ``MultiviewUnlabeledBatchDict``).

*Coordinate and label utilities* — used inside datasets and model forward passes:

- ``bboxes.py`` — coordinate transforms between frame, normalised-bbox, and model
  pixel spaces; see that module's docstring for the three-space mental model.
- ``heatmaps.py`` — :func:`~lightning_pose.data.heatmaps.generate_heatmaps` and
  :func:`~lightning_pose.data.heatmaps.evaluate_heatmaps_at_location`.
- ``cameras.py`` — camera-parameter utilities for multiview calibration.
- ``utils.py`` — dataset-level utilities: train/val/test splits, frame counting,
  affine-transform undo.

*Video infrastructure* — used only for semi-supervised training:

- ``dali.py`` — GPU-accelerated video loading via NVIDIA DALI
  (:class:`~lightning_pose.data.dali.PrepareDALI`,
  :class:`~lightning_pose.data.dali.LitDaliWrapper`).
- ``extractor.py`` — frame extraction utilities.
"""

# statistics of imagenet dataset on which the resnet was trained
# see https://pytorch.org/vision/stable/models.html
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

from lightning_pose.data.factory import (  # noqa: E402
    get_data_module,
    get_dataset,
    get_imgaug_transform,
)

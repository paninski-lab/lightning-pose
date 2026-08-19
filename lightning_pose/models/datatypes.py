"""Return-type contracts for tracker loss-input methods.

Each ``TypedDict`` here documents the exact keys and tensor shapes produced by one tracker's
``get_loss_inputs_labeled``/``get_loss_inputs_unlabeled`` method (see
:class:`~lightning_pose.models.base.BaseSupervisedTracker` and
:class:`~lightning_pose.models.base.SemiSupervisedTrackerMixin`). These dicts are unpacked with
``**`` into a :class:`~lightning_pose.losses.factory.LossFactory` call, so a given dict may feed
several different losses at once; the contract is therefore named after the tracker that
produces it, not any single loss that consumes it.
"""
from __future__ import annotations

from typing import TypedDict

import torch
from jaxtyping import Float

# to ignore imports for sphinx-autoapidoc
__all__ = [
    "RegressionTrackerLabeledOutputsDict",
    "RegressionTrackerUnlabeledOutputsDict",
    "HeatmapTrackerLabeledOutputsDict",
    "HeatmapTrackerUnlabeledOutputsDict",
    "HeatmapTrackerMHCRNNUnlabeledOutputsDict",
    "HeatmapTrackerMultiviewTransformerLabeledOutputsDict",
]


class RegressionTrackerLabeledOutputsDict(TypedDict):
    """Return type of ``RegressionTracker.get_loss_inputs_labeled()``."""
    keypoints_targ: Float[torch.Tensor, "batch num_targets"]
    keypoints_pred: Float[torch.Tensor, "batch num_targets"]


class RegressionTrackerUnlabeledOutputsDict(TypedDict):
    """Return type of ``SemiSupervisedRegressionTracker.get_loss_inputs_unlabeled()``."""
    keypoints_pred: Float[torch.Tensor, "seq_len num_targets"]


class HeatmapTrackerLabeledOutputsDict(TypedDict):
    """Return type of ``HeatmapTracker.get_loss_inputs_labeled()`` and
    ``HeatmapTrackerMHCRNN.get_loss_inputs_labeled()``."""
    heatmaps_targ: Float[torch.Tensor, "batch num_keypoints heatmap_height heatmap_width"]
    heatmaps_pred: Float[torch.Tensor, "batch num_keypoints heatmap_height heatmap_width"]
    keypoints_targ: Float[torch.Tensor, "batch num_targets"]
    keypoints_pred: Float[torch.Tensor, "batch num_targets"]
    confidences: Float[torch.Tensor, "batch num_keypoints"]


class HeatmapTrackerUnlabeledOutputsDict(TypedDict):
    """Return type of ``SemiSupervisedHeatmapTracker.get_loss_inputs_unlabeled()`` and
    ``SemiSupervisedHeatmapTrackerMultiviewTransformer.get_loss_inputs_unlabeled()``."""
    heatmaps_pred: Float[torch.Tensor, "seq_len num_keypoints heatmap_height heatmap_width"]
    keypoints_pred: Float[torch.Tensor, "seq_len num_targets"]
    keypoints_pred_augmented: Float[torch.Tensor, "seq_len num_targets"]
    confidences: Float[torch.Tensor, "seq_len num_keypoints"]


class HeatmapTrackerMHCRNNUnlabeledOutputsDict(TypedDict):
    """Return type of ``SemiSupervisedHeatmapTrackerMHCRNN.get_loss_inputs_unlabeled()``."""
    heatmaps_pred: Float[torch.Tensor, "seq_len num_keypoints heatmap_height heatmap_width"]
    keypoints_pred: Float[torch.Tensor, "seq_len num_targets"]
    confidences: Float[torch.Tensor, "seq_len num_keypoints"]


class HeatmapTrackerMultiviewTransformerLabeledOutputsDict(HeatmapTrackerLabeledOutputsDict):
    """Return type of ``HeatmapTrackerMultiviewTransformer.get_loss_inputs_labeled()``.

    Adds 3D-projection keys that are only populated when camera calibration data is present
    on the batch; all three are ``None`` otherwise.
    """
    keypoints_targ_3d: Float[torch.Tensor, "batch num_keypoints 3"] | None
    keypoints_pred_3d: Float[torch.Tensor, "batch cam_pairs num_keypoints 3"] | None
    keypoints_pred_2d_reprojected: Float[torch.Tensor, "batch num_views_x_num_keypoints 2"] | None

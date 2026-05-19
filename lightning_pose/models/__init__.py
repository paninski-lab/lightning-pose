"""Pose estimation model classes, re-exported at the package level."""

from typing import Union

from omegaconf import ListConfig

from lightning_pose.models.heatmap_tracker import (
    HeatmapTracker,
    SemiSupervisedHeatmapTracker,
)
from lightning_pose.models.heatmap_tracker_mhcrnn import (
    HeatmapTrackerMHCRNN,
    SemiSupervisedHeatmapTrackerMHCRNN,
)
from lightning_pose.models.heatmap_tracker_multiview import (
    HeatmapTrackerMultiviewTransformer,
    SemiSupervisedHeatmapTrackerMultiviewTransformer,
)
from lightning_pose.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)

# reassign module to make classes appear to belong here
HeatmapTracker.__module__ = "lightning_pose.models"
SemiSupervisedHeatmapTracker.__module__ = "lightning_pose.models"
HeatmapTrackerMHCRNN.__module__ = "lightning_pose.models"
SemiSupervisedHeatmapTrackerMHCRNN.__module__ = "lightning_pose.models"
HeatmapTrackerMultiviewTransformer.__module__ = "lightning_pose.models"
SemiSupervisedHeatmapTrackerMultiviewTransformer.__module__ = "lightning_pose.models"
RegressionTracker.__module__ = "lightning_pose.models"
SemiSupervisedRegressionTracker.__module__ = "lightning_pose.models"

def check_if_semi_supervised(losses_to_use: ListConfig | list | None = None) -> bool:
    """Determine from the losses config whether the model is semi-supervised.

    Args:
        losses_to_use: the cfg entry specifying unsupervised losses to use.

    Returns:
        True if the model is semi-supervised, False otherwise.

    """
    if losses_to_use is None:
        return False
    if len(losses_to_use) == 0:
        return False
    if len(losses_to_use) == 1 and losses_to_use[0] == '':
        return False
    return True


ALLOWED_MODELS = (
    HeatmapTracker
    | SemiSupervisedHeatmapTracker
    | HeatmapTrackerMHCRNN
    | SemiSupervisedHeatmapTrackerMHCRNN
    | HeatmapTrackerMultiviewTransformer
    | SemiSupervisedHeatmapTrackerMultiviewTransformer
    | RegressionTracker
    | SemiSupervisedRegressionTracker
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "check_if_semi_supervised",
    "HeatmapTracker",
    "SemiSupervisedHeatmapTracker",
    "HeatmapTrackerMHCRNN",
    "SemiSupervisedHeatmapTrackerMHCRNN",
    "HeatmapTrackerMultiviewTransformer",
    "SemiSupervisedHeatmapTrackerMultiviewTransformer",
    "RegressionTracker",
    "SemiSupervisedRegressionTracker",
]

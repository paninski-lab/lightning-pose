"""Pose estimation model classes, re-exported at the package level."""

from typing import Literal

from lightning_pose.models.base import check_if_semi_supervised
from lightning_pose.models.factory import get_model
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
from lightning_pose.models.heatmap_tracker_multiview_mhcrnn import (
    HeatmapTrackerMultiviewTransformerMHCRNN,
)
from lightning_pose.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)

# reassign module to make classes appear to belong here
HeatmapTracker.__module__ = 'lightning_pose.models'
SemiSupervisedHeatmapTracker.__module__ = 'lightning_pose.models'
HeatmapTrackerMHCRNN.__module__ = 'lightning_pose.models'
SemiSupervisedHeatmapTrackerMHCRNN.__module__ = 'lightning_pose.models'
HeatmapTrackerMultiviewTransformer.__module__ = 'lightning_pose.models'
SemiSupervisedHeatmapTrackerMultiviewTransformer.__module__ = 'lightning_pose.models'
HeatmapTrackerMultiviewTransformerMHCRNN.__module__ = 'lightning_pose.models'
RegressionTracker.__module__ = 'lightning_pose.models'
SemiSupervisedRegressionTracker.__module__ = 'lightning_pose.models'

ALLOWED_MODEL_TYPES = Literal[
    'regression',
    'heatmap',
    'heatmap_mhcrnn',
    'heatmap_multiview_transformer',
    'heatmap_multiview_transformer_mhcrnn',
]

ALLOWED_MODELS = (
    HeatmapTracker
    | SemiSupervisedHeatmapTracker
    | HeatmapTrackerMHCRNN
    | SemiSupervisedHeatmapTrackerMHCRNN
    | HeatmapTrackerMultiviewTransformer
    | SemiSupervisedHeatmapTrackerMultiviewTransformer
    | HeatmapTrackerMultiviewTransformerMHCRNN
    | RegressionTracker
    | SemiSupervisedRegressionTracker
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    'check_if_semi_supervised',
    'get_model',
    'HeatmapTracker',
    'SemiSupervisedHeatmapTracker',
    'HeatmapTrackerMHCRNN',
    'SemiSupervisedHeatmapTrackerMHCRNN',
    'HeatmapTrackerMultiviewTransformer',
    'SemiSupervisedHeatmapTrackerMultiviewTransformer',
    'HeatmapTrackerMultiviewTransformerMHCRNN',
    'RegressionTracker',
    'SemiSupervisedRegressionTracker',
]

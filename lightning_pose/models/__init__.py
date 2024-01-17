from typing import Union

from lightning_pose.models.heatmap_tracker import HeatmapTracker, SemiSupervisedHeatmapTracker
from lightning_pose.models.heatmap_tracker_mhcrnn import (
    HeatmapTrackerMHCRNN,
    SemiSupervisedHeatmapTrackerMHCRNN,
)
from lightning_pose.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)

ALLOWED_MODELS = Union[
    HeatmapTracker,
    SemiSupervisedHeatmapTracker,
    HeatmapTrackerMHCRNN,
    SemiSupervisedHeatmapTrackerMHCRNN,
    RegressionTracker,
    SemiSupervisedRegressionTracker
]

# to ignore imports for sphix-autoapidoc
__all__ = []

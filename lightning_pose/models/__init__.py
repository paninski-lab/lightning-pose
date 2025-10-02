from typing import Union

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

ALLOWED_MODELS = Union[
    HeatmapTracker,
    SemiSupervisedHeatmapTracker,
    HeatmapTrackerMHCRNN,
    SemiSupervisedHeatmapTrackerMHCRNN,
    HeatmapTrackerMultiviewTransformer,
    SemiSupervisedHeatmapTrackerMultiviewTransformer,
    RegressionTracker,
    SemiSupervisedRegressionTracker,
]

# to ignore imports for sphix-autoapidoc
__all__ = [
    "HeatmapTracker",
    "SemiSupervisedHeatmapTracker",
    "HeatmapTrackerMHCRNN",
    "SemiSupervisedHeatmapTrackerMHCRNN",
    "HeatmapTrackerMultiviewTransformer",
    "SemiSupervisedHeatmapTrackerMultiviewTransformer",
    "RegressionTracker",
    "SemiSupervisedRegressionTracker",
]

from lightning_pose.models.heads.heatmap import HeatmapHead
from lightning_pose.models.heads.heatmap_mhcrnn import HeatmapMHCRNNHead
from lightning_pose.models.heads.regression import LinearRegressionHead

# reassign module to make classes appear to belong here
HeatmapHead.__module__ = "lightning_pose.models.heads"
HeatmapMHCRNNHead.__module__ = "lightning_pose.models.heads"
LinearRegressionHead.__module__ = "lightning_pose.models.heads"

# to ignore imports for sphix-autoapidoc
__all__ = [
    "HeatmapHead",
    "HeatmapMHCRNNHead",
    "LinearRegressionHead",
]

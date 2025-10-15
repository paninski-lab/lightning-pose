from typing import Union

from lightning_pose.models.heads.heatmap import HeatmapHead
from lightning_pose.models.heads.heatmap_mhcrnn import HeatmapMHCRNNHead, HeatmapMHCRNNHeadMultiview
from lightning_pose.models.heads.heatmap_multiview import (
    MultiviewHeatmapCNNHead,
    MultiviewHeatmapCNNMultiHead,
)

from lightning_pose.models.heads.regression import LinearRegressionHead

ALLOWED_MULTIVIEW_HEADS = Union[
    MultiviewHeatmapCNNHead,
]

ALLOWED_MULTIVIEW_MULTIHEADS = Union[
    MultiviewHeatmapCNNMultiHead,
]

# to ignore imports for sphix-autoapidoc
__all__ = []

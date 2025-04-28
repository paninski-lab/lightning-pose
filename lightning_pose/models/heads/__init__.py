from typing import Union

from lightning_pose.models.heads.heatmap import HeatmapHead
from lightning_pose.models.heads.heatmap_mhcrnn import HeatmapMHCRNNHead
from lightning_pose.models.heads.heatmap_multiview import (
    MultiviewHeatmapCNNHead,
    MultiviewHeatmapCNNMultiHead,
)
from lightning_pose.models.heads.heatmap_multiview_transformer import (
    MultiviewFeatureTransformerHead,
)
from lightning_pose.models.heads.regression import LinearRegressionHead

ALLOWED_MULTIVIEW_HEADS = Union[
    MultiviewFeatureTransformerHead,
    MultiviewHeatmapCNNHead,
]

ALLOWED_MULTIVIEW_MULTIHEADS = Union[
    MultiviewHeatmapCNNMultiHead,
]

# to ignore imports for sphix-autoapidoc
__all__ = []

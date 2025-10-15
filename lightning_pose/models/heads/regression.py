"""Heads that produce (x, y) predictions for coordinate regression."""

from torch import nn
from torchtyping import TensorType

# to ignore imports for sphix-autoapidoc
__all__ = []


class LinearRegressionHead(nn.Module):
    """Linear regression head that converts 2D feature maps to a vector of (x, y) coordinates."""

    def __init__(
        self,
        in_channels: int,
        num_targets: int,
    ):
        super().__init__()
        self.linear_layer = nn.Linear(in_channels, num_targets)

    def forward(
        self,
        features: TensorType["batch", "features", "height", "width"]
    ) -> TensorType["batch", "coordinates"]:
        features_reshaped = features.reshape(features.shape[0], features.shape[1])
        coordinates = self.linear_layer(features_reshaped)
        return coordinates

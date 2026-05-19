"""Heads that produce (x, y) predictions for coordinate regression."""

import torch
from jaxtyping import Float
from torch import nn

# to ignore imports for sphix-autoapidoc
__all__ = []


class LinearRegressionHead(nn.Module):
    """Linear regression head that converts 2D feature maps to a vector of (x, y) coordinates."""

    def __init__(
        self,
        in_channels: int,
        num_targets: int,
    ) -> None:
        """Initialize LinearRegressionHead.

        Args:
            in_channels: number of input feature channels.
            num_targets: number of output coordinate values (``2 * num_keypoints``).
        """
        super().__init__()
        self.linear_layer = nn.Linear(in_channels, num_targets)

    def forward(
        self,
        features: Float[torch.Tensor, "batch features height width"]
    ) -> Float[torch.Tensor, "batch coordinates"]:
        """Map feature maps to keypoint coordinate predictions.

        Args:
            features: feature tensor of shape ``(batch, features, height, width)``;
                spatial dimensions are collapsed before the linear layer.

        Returns:
            Predicted coordinates of shape ``(batch, num_targets)``.
        """
        features_reshaped = features.reshape(features.shape[0], features.shape[1])
        coordinates = self.linear_layer(features_reshaped)
        return coordinates

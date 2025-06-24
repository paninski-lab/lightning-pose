"""Load vision encoder from Facebook DINO model."""

import math

import torch
from transformers import ViTModel

# to ignore imports for sphix-autoapidoc
__all__ = [
    "DinoVisionEncoder",
]


class DinoVisionEncoder(torch.nn.Module):
    """Wrapper around DINO Vision Encoder."""

    def __init__(self, model_name: str = "facebook/dino-vitb16"):
        super().__init__()
        # Load the DINO backbone
        self.vision_encoder = ViTModel.from_pretrained(model_name)
        if hasattr(self, "pooler"):
            self.pooler = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vision encoder.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Encoded features
        """

        # Patch embedding
        outputs = self.vision_encoder(
            x,
            return_dict=True,
            output_hidden_states=False,
            output_attentions=False,
            interpolate_pos_encoding=True,
        ).last_hidden_state

        # skip the cls token
        outputs = outputs[:, 1:, ...]  # [N, S, D]
        # change the shape to [N, H, W, D] -> [N, D, H, W]
        N = x.shape[0]
        S = outputs.shape[1]
        H, W = math.isqrt(S), math.isqrt(S)
        outputs = outputs.reshape(N, H, W, -1).permute(0, 3, 1, 2)

        return outputs

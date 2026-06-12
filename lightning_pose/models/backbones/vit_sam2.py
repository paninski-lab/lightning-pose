"""Load vision encoder from Facebook SAM 2 model using HuggingFace."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


class VisionEncoderSam2(nn.Module):
    """Wrapper around HuggingFace's SAM2 Hiera vision backbone.

    Loads the hierarchical Hiera transformer from a SAM2 checkpoint and discards the
    FPN neck. Positional embeddings are bicubic-interpolated inside the backbone's forward
    pass, so arbitrary input resolutions are supported without any init-time resizing.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize VisionEncoderSam2 from a HuggingFace SAM2 checkpoint.

        Args:
            model_name: HuggingFace model identifier, e.g. "facebook/sam2.1-hiera-small".
        """
        super().__init__()

        from transformers import AutoModel
        full_model = AutoModel.from_pretrained(model_name)
        full_model = full_model.cpu()
        # retain only the Hiera backbone; discard the FPN neck and mask decoder
        self.vision_encoder = full_model.vision_encoder.backbone
        del full_model

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run input through the Hiera backbone.

        Args:
            x: input tensor of shape (B, C, H, W)

        Returns:
            spatial feature map of shape (B, embed_dim, H', W') where the spatial
            stride is 32 relative to the input (4× patch embed + 3× 2× pooling stages)
        """
        outputs = self.vision_encoder(pixel_values=x)
        # last_hidden_state is NHWC: (B, H', W', embed_dim_final_stage)
        return outputs.last_hidden_state.permute(0, 3, 1, 2)

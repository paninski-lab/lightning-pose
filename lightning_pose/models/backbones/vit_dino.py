"""Load vision encoder from DINO/DINOv2/DINOv3 models using HuggingFace."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []

_DINOV3_ACCESS_HELP = """
================================================================================
DINOv3 Model Access Required
================================================================================

The DINOv3 models are gated on HuggingFace and require authentication.
Please follow these steps to gain access:

1. CREATE A HUGGINGFACE ACCOUNT (if you don't have one):
   - Go to https://huggingface.co/join
   - Sign up with your email

2. REQUEST ACCESS TO THE MODEL:
   - Visit https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
   - Click "Agree and access repository"
   - Accept the terms of use
   - You should receive immediate access

3. CREATE AN ACCESS TOKEN:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Give it a name (e.g., "lightning-pose-dinov3")
   - Select "Read" permission (sufficient for downloading models)
   - Click "Generate token"
   - COPY THE TOKEN - you won't be able to see it again!

4. INSTALL/UPDATE HUGGINGFACE PACKAGES AND LOGIN FROM THE TERMINAL:
   pip install -U transformers huggingface_hub
   hf auth login

   When prompted, paste your token and press Enter.

5. RE-RUN YOUR CODE

For more information, visit: https://huggingface.co/docs/hub/security-tokens

Note: Both vits_dinov3 and vitb_dinov3 backbones require this authentication.
================================================================================
"""


class VisionEncoderDino(nn.Module):
    """Wrapper around DINOv2/DINOv3 vision encoder.

    Normalises all models to patch size 16 via bicubic interpolation of the patch-embedding
    projection weights when the pre-trained checkpoint uses a different patch size.
    Gated-repo errors from HuggingFace (e.g. DINOv3) are caught and re-raised with
    step-by-step authentication instructions.
    """

    def __init__(self, model_name: str, pretrained_patch_size: int) -> None:
        """Initialize VisionEncoderDINO from a HuggingFace DINO model.

        Args:
            model_name: HuggingFace model identifier (e.g. ``"facebook/dinov2-base"``).
            pretrained_patch_size: native patch size of the pre-trained checkpoint; weights
                are bicubic-interpolated to patch size 16 when this differs from 16.

        Raises:
            RuntimeError: when the model is on a gated HuggingFace repo and the user has
                not authenticated.
        """
        super().__init__()
        from transformers import AutoModel
        try:
            self.vision_encoder = AutoModel.from_pretrained(model_name)
        except OSError as e:
            if 'gated repo' in str(e).lower():
                logger.error(_DINOV3_ACCESS_HELP)
                raise RuntimeError(
                    'Cannot access gated model. Please follow the instructions above to '
                    'authenticate with HuggingFace.'
                ) from e
            raise
        # self.vision_encoder is one of:
        # - transformers.models.dinov2.modeling_dinov2.Dinov2Model
        # - transformers.models.dinov3_vit.modeling_dinov3_vit.Dinov3ViTModel

        if pretrained_patch_size != 16:
            patch_size = 16
            self.patch_size = patch_size
            self._resize_patch_embedding_weights()
            self.vision_encoder.config.patch_size = patch_size
            self.vision_encoder.embeddings.patch_size = patch_size
            self.vision_encoder.embeddings.patch_embeddings.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vision encoder.

        Args:
            x: input tensor of shape (B, C, H, W)

        Returns:
            encoded features of shape (B, D, H', W')
        """
        outputs = self.vision_encoder(
            x,
            output_hidden_states=False,
        ).last_hidden_state

        # v2/v3 each have 1 CLS token; v3 has 4 additional register tokens
        num_prefix = 1 + getattr(self.vision_encoder.config, 'num_register_tokens', 0)
        outputs = outputs[:, num_prefix:, ...]  # (N, S, D)
        N, _, height, width = x.shape
        patch_size = self.vision_encoder.config.patch_size
        H, W = int(height / patch_size), int(width / patch_size)
        return outputs.reshape(N, H, W, -1).permute(0, 3, 1, 2)

    def _resize_patch_embedding_weights(self) -> None:
        """Resize patch-embedding projection to patch size 16 via bicubic interpolation."""
        projection = self.vision_encoder.embeddings.patch_embeddings.projection
        out_channels, in_channels, old_h, old_w = projection.weight.shape
        new_h, new_w = self.patch_size, self.patch_size

        reshaped = projection.weight.data.view(out_channels * in_channels, 1, old_h, old_w)
        resized = torch.nn.functional.interpolate(
            reshaped,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=True,
            antialias=True,
        )
        new_weights = resized.view(out_channels, in_channels, new_h, new_w)

        new_projection = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=new_h, stride=new_h,
            bias=projection.bias is not None,
        )
        new_projection.weight.data = new_weights
        if projection.bias is not None:
            assert new_projection.bias is not None
            new_projection.bias.data = projection.bias.data

        self.vision_encoder.embeddings.patch_embeddings.projection = new_projection

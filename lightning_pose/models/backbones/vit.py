"""Load vision encoder from plain ViT models using HuggingFace."""

import logging
import math

import safetensors.torch
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# to ignore imports for sphix-autoapidoc
__all__ = []


class VisionEncoder(nn.Module):
    """Wrapper around a generic HuggingFace ViT encoder."""

    def __init__(self, model_name: str) -> None:
        """Initialize VisionEncoder by loading a pre-trained ViT from HuggingFace.

        Args:
            model_name: HuggingFace model identifier (e.g. ``"facebook/dino-vitb16"``).
        """
        super().__init__()
        from transformers import ViTModel
        self.vision_encoder = ViTModel.from_pretrained(model_name, add_pooling_layer=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vision encoder.

        Args:
            x: input tensor of shape (B, C, H, W)

        Returns:
            encoded features of shape (B, D, H', W')
        """
        outputs = self.vision_encoder(
            x,
            return_dict=True,
            output_hidden_states=False,
            interpolate_pos_encoding=True,
        ).last_hidden_state

        outputs = outputs[:, 1:, ...]  # skip CLS token → (N, S, D)
        N = x.shape[0]
        S = outputs.shape[1]
        H, W = math.isqrt(S), math.isqrt(S)
        return outputs.reshape(N, H, W, -1).permute(0, 3, 1, 2)


def load_vit_backbone_checkpoint(base: VisionEncoder, checkpoint: str) -> None:
    """Load pre-trained ViT-MAE weights into a VisionEncoder backbone.

    Supports both ``.safetensors`` and standard PyTorch checkpoint formats. Only layers
    whose names and shapes match the encoder's state dict are loaded.

    Args:
        base: the ``VisionEncoder`` instance whose weights will be updated.
        checkpoint: path to the checkpoint file.
    """
    logger.info(f'loading VIT-MAE weights from {checkpoint}')
    if checkpoint.endswith('.safetensors'):
        ckpt_vit_pretrain = safetensors.torch.load_file(checkpoint, device='cpu')
    else:
        try:
            ckpt_vit_pretrain = torch.load(checkpoint, map_location='cpu')
        except Exception as e:
            logger.warning(f'failed to load checkpoint with default settings: {e}')
            logger.warning('attempting to load with weights_only=False...')
            ckpt_vit_pretrain = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt_vit_pretrain:
        ckpt_vit_pretrain = ckpt_vit_pretrain['state_dict']
    vit_mae_state_dict = {}
    for key, value in ckpt_vit_pretrain.items():  # type: ignore[union-attr]
        if key.startswith('vit_mae.'):
            model_key = key.replace('vit_mae.vit.', '')
            if model_key in base.vision_encoder.state_dict():
                if base.vision_encoder.state_dict()[model_key].shape == value.shape:
                    vit_mae_state_dict[model_key] = value
    base.vision_encoder.load_state_dict(vit_mae_state_dict, strict=False)

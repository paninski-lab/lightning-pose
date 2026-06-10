"""Backbone loader for Vision Transformer (ViT) architectures including DINO and SAM."""

from __future__ import annotations

import logging
import math
from typing import Any

import safetensors.torch
import torch

logger = logging.getLogger(__name__)

# to ignore imports for sphix-autoapidoc
__all__ = []


def build_backbone(
    backbone_arch: str,
    image_size: int = 256,
    **kwargs: Any
) -> tuple[torch.nn.Module, int]:
    """Load backbone weights for resnet models.

    Args:
        backbone_arch: which backbone version/weights to use
        image_size: height/width in pixels of images (must be square)

    Returns:
        tuple
            - backbone: pytorch model
            - num_fc_input_features (int): number of input features to fully connected layer

    """

    # load backbone weights
    if backbone_arch == "vits_dino":
        base = VisionEncoder(model_name="facebook/dino-vits16")
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == "vitb_dino":
        base = VisionEncoder(model_name="facebook/dino-vitb16")
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == "vits_dinov2":
        from lightning_pose.models.backbones.vit_dino import VisionEncoderDino
        base = VisionEncoderDino(model_name="facebook/dinov2-small", pretrained_patch_size=14)
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == "vitb_dinov2":
        from lightning_pose.models.backbones.vit_dino import VisionEncoderDino
        base = VisionEncoderDino(model_name="facebook/dinov2-base", pretrained_patch_size=14)
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == "vits_dinov3":
        from lightning_pose.models.backbones.vit_dino import VisionEncoderDino
        base = VisionEncoderDino(
            model_name="facebook/dinov3-vits16-pretrain-lvd1689m", pretrained_patch_size=16,
        )
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == "vitb_dinov3":
        from lightning_pose.models.backbones.vit_dino import VisionEncoderDino
        base = VisionEncoderDino(
            model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", pretrained_patch_size=16,
        )
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif "vitb_imagenet" in backbone_arch:
        base = VisionEncoder(model_name="facebook/vit-mae-base")
        encoder_embed_dim = base.vision_encoder.config.hidden_size
        if kwargs.get("backbone_checkpoint"):
            load_vit_backbone_checkpoint(base, kwargs["backbone_checkpoint"])
    elif backbone_arch == "vitb_sam":
        from lightning_pose.models.backbones.vit_sam import VisionEncoderSam
        base = VisionEncoderSam(
            model_name="facebook/sam-vit-base",
            finetune_img_size=image_size,
        )
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch in ("vitb_sam2", "vits_sam2", "vitt_sam2"):
        from lightning_pose.models.backbones.vit_sam2 import VisionEncoderSam2
        _sam2_model_names = {
            "vitb_sam2": "facebook/sam2.1-hiera-base-plus",
            "vits_sam2": "facebook/sam2.1-hiera-small",
            "vitt_sam2": "facebook/sam2.1-hiera-tiny",
        }
        base = VisionEncoderSam2(model_name=_sam2_model_names[backbone_arch])
        encoder_embed_dim = base.vision_encoder.config.embed_dim_per_stage[-1]
    else:
        raise NotImplementedError(f"{backbone_arch} is not a valid backbone")

    num_fc_input_features = encoder_embed_dim

    return base, num_fc_input_features


def load_vit_backbone_checkpoint(base: VisionEncoder, checkpoint: str) -> None:
    """Load pre-trained ViT-MAE weights into a VisionEncoder backbone.

    Supports both ``.safetensors`` and standard PyTorch checkpoint formats. Only layers
    whose names and shapes match the encoder's state dict are loaded.

    Args:
        base: the ``VisionEncoder`` instance whose weights will be updated.
        checkpoint: path to the checkpoint file.
    """
    logger.info(f'loading VIT-MAE weights from {checkpoint}')
    # support loading safetensors
    if checkpoint.endswith(".safetensors"):
        ckpt_vit_pretrain = safetensors.torch.load_file(checkpoint, device="cpu")
    else:
        # Try loading with default settings first, fallback to weights_only=False if needed
        try:
            ckpt_vit_pretrain = torch.load(checkpoint, map_location="cpu")
        except Exception as e:
            logger.warning(f'failed to load checkpoint with default settings: {e}')
            logger.warning('attempting to load with weights_only=False...')
            ckpt_vit_pretrain = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # extract state dict if checkpoint contains additional info
    if "state_dict" in ckpt_vit_pretrain:
        ckpt_vit_pretrain = ckpt_vit_pretrain["state_dict"]
    # Create a filtered state dict for the VIT-MAE part only
    vit_mae_state_dict = {}
    for key, value in ckpt_vit_pretrain.items():  # type: ignore[union-attr]
        if key.startswith("vit_mae."):
            model_key = key.replace("vit_mae.vit.", "")
            # Check if shapes match before including in state dict
            if model_key in base.vision_encoder.state_dict():
                if base.vision_encoder.state_dict()[model_key].shape == value.shape:
                    vit_mae_state_dict[model_key] = value
    # Load the filtered weights
    base.vision_encoder.load_state_dict(vit_mae_state_dict, strict=False)


class VisionEncoder(torch.nn.Module):
    """Wrapper around generic ViT Encoder."""

    def __init__(self, model_name: str) -> None:
        """Initialize VisionEncoder by loading a pre-trained ViT from HuggingFace.

        Args:
            model_name: HuggingFace model identifier (e.g., ``"facebook/dino-vitb16"``).
        """
        super().__init__()
        from transformers import ViTModel
        self.vision_encoder = ViTModel.from_pretrained(model_name, add_pooling_layer=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vision encoder.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Encoded features
        """

        outputs = self.vision_encoder(
            x,
            return_dict=True,
            output_hidden_states=False,
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

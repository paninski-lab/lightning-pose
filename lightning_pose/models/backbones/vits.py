import math

import torch
from transformers import ViTModel
from typeguard import typechecked

from lightning_pose.models.backbones.vit_sam import SamVisionEncoder

# to ignore imports for sphix-autoapidoc
__all__ = []


@typechecked
def build_backbone(backbone_arch: str, image_size: int = 256, **kwargs):
    """Load backbone weights for resnet models.

    Args:
        backbone_arch: which backbone version/weights to use
        image_size: height/width in pixels of images (must be square)

    Returns:
        tuple
            - backbone: pytorch model
            - num_fc_input_features (int): number of input features to fully connected layer

    """

    # deprecation warnings
    if "vit_h_sam" in backbone_arch:
        backbone_arch = "vitb_sam"
        raise DeprecationWarning('vit_h_sam is now deprecated; reverting to "vitb_sam"')
    elif "vit_b_sam" in backbone_arch:
        backbone_arch = "vitb_sam"
        raise DeprecationWarning('vit_b_sam is now deprecated; reverting to "vitb_sam"')

    # load backbone weights
    if "vits_dino" in backbone_arch:
        base = VisionEncoder(model_name="facebook/dino-vits16")
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif "vitb_dino" in backbone_arch:
        base = VisionEncoder(model_name="facebook/dino-vitb16")
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif "vitb_imagenet" in backbone_arch:
        base = VisionEncoder(model_name="facebook/vit-mae-base")
        encoder_embed_dim = base.vision_encoder.config.hidden_size
        if kwargs.get("backbone_checkpoint"):
            load_vit_backbone_checkpoint(base, kwargs["backbone_checkpoint"])
    elif "vitb_sam" in backbone_arch:
        base = SamVisionEncoder(
            model_name="facebook/sam-vit-base",
            finetune_img_size=image_size,
        )
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    else:
        raise NotImplementedError(f"{backbone_arch} is not a valid backbone")

    num_fc_input_features = encoder_embed_dim

    return base, num_fc_input_features


def load_vit_backbone_checkpoint(base, checkpoint: str):
    print(f"Loading VIT-MAE weights from {checkpoint}")
    # Try loading with default settings first, fallback to weights_only=False if needed
    try:
        ckpt_vit_pretrain = torch.load(checkpoint, map_location="cpu")
    except Exception as e:
        print(f"Warning: Failed to load checkpoint with default settings: {e}")
        print("Attempting to load with weights_only=False...")
        ckpt_vit_pretrain = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # extract state dict if checkpoint contains additional info
    if "state_dict" in ckpt_vit_pretrain:
        ckpt_vit_pretrain = ckpt_vit_pretrain["state_dict"]
    # Create a filtered state dict for the VIT-MAE part only
    vit_mae_state_dict = {}
    for key, value in ckpt_vit_pretrain.items():
        if key.startswith("vit_mae."):
            model_key = key.replace("vit_mae.vit.", "")
            # Skip known problematic layers with size mismatches
            if any(prob in model_key for prob in [
                "position_embeddings",
                "patch_embeddings.projection",  # in case backbone was trained with grayscale imgs
                "decoder_pos_embed",
                "decoder_pred",
            ]):
                continue
            # Check if shapes match before including in state dict
            if model_key in base.vision_encoder.state_dict():
                if base.vision_encoder.state_dict()[model_key].shape == value.shape:
                    vit_mae_state_dict[model_key] = value
    # Load the filtered weights
    base.vision_encoder.load_state_dict(vit_mae_state_dict, strict=False)


class VisionEncoder(torch.nn.Module):
    """Wrapper around ViT Encoder."""

    def __init__(self, model_name):
        super().__init__()
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

import math

import safetensors
import torch
from typeguard import typechecked

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

    # load backbone weights
    if backbone_arch == "vits_dino":
        base = VisionEncoder(model_name="facebook/dino-vits16")
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == "vitb_dino":
        base = VisionEncoder(model_name="facebook/dino-vitb16")
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == "vits_dinov2":
        base = VisionEncoderDino(model_name="facebook/dinov2-small", pretrained_patch_size=14)
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == "vitb_dinov2":
        base = VisionEncoderDino(model_name="facebook/dinov2-base", pretrained_patch_size=14)
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == "vits_dinov3":
        base = VisionEncoderDino(
            model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            pretrained_patch_size=16,
        )
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == "vitb_dinov3":
        base = VisionEncoderDino(
            model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
            pretrained_patch_size=16,
        )
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif "vitb_imagenet" in backbone_arch:
        base = VisionEncoder(model_name="facebook/vit-mae-base")
        encoder_embed_dim = base.vision_encoder.config.hidden_size
        if kwargs.get("backbone_checkpoint"):
            load_vit_backbone_checkpoint(base, kwargs["backbone_checkpoint"])
    elif backbone_arch == "vitb_sam":
        from lightning_pose.models.backbones.vit_sam import SamVisionEncoder
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
    # support loading safetensors
    if checkpoint.endswith(".safetensors"):
        ckpt_vit_pretrain = safetensors.torch.load_file(checkpoint, device="cpu")
    else:
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
            # Check if shapes match before including in state dict
            if model_key in base.vision_encoder.state_dict():
                if base.vision_encoder.state_dict()[model_key].shape == value.shape:
                    vit_mae_state_dict[model_key] = value
    # Load the filtered weights
    base.vision_encoder.load_state_dict(vit_mae_state_dict, strict=False)


class VisionEncoder(torch.nn.Module):
    """Wrapper around generic ViT Encoder."""

    def __init__(self, model_name):
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


class VisionEncoderDino(torch.nn.Module):
    """Wrapper around DINOv2/DINOv3 Encoder."""

    def __init__(self, model_name, pretrained_patch_size):
        super().__init__()
        from transformers import AutoModel
        self.vision_encoder = AutoModel.from_pretrained(model_name)
        # self.vision_encoder is one of:
        # - transformers.models.dinov2.modeling_dinov2.Dinov2Model
        # - transformers.models.dinov3_vit.modeling_dinov3_vit.Dinov3ViTModel

        if pretrained_patch_size != 16:
            # use patch size of 16 for all models
            patch_size = 16
            self.patch_size = patch_size
            self._resize_patch_embedding_weights()
            self.vision_encoder.config.patch_size = patch_size
            self.vision_encoder.embeddings.patch_size = patch_size
            self.vision_encoder.embeddings.patch_embeddings.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vision encoder.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Encoded features
        """

        outputs = self.vision_encoder(
            x,
            output_hidden_states=False,
        ).last_hidden_state

        # v2/v3 each have 1 CLS token, v3 has 4 register tokens
        num_prefix = 1 + getattr(self.vision_encoder.config, "num_register_tokens", 0)
        # skip the cls+register token
        outputs = outputs[:, num_prefix:, ...]  # [N, S, D]
        # change the shape to [N, H, W, D] -> [N, D, H, W]
        N, _, height, width = x.shape
        patch_size = self.vision_encoder.config.patch_size
        H, W = int(height / patch_size), int(width / patch_size)
        outputs = outputs.reshape(N, H, W, -1).permute(0, 3, 1, 2)

        return outputs

    def _resize_patch_embedding_weights(self):

        projection = self.vision_encoder.embeddings.patch_embeddings.projection
        out_channels, in_channels, old_h, old_w = projection.weight.shape
        new_h, new_w = (self.patch_size, self.patch_size)

        # Reshape to (out_channels * in_channels, 1, old_h, old_w) for interpolation
        reshaped = projection.weight.data.view(out_channels * in_channels, 1, old_h, old_w)

        # Use bicubic interpolation
        resized = torch.nn.functional.interpolate(
            reshaped,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=True,
            antialias=True,  # reduces aliasing artifacts
        )

        # Reshape back to original format
        new_weights = resized.view(out_channels, in_channels, new_h, new_w)

        new_projection = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=new_h, stride=new_h,
            bias=projection.bias is not None,
        )
        new_projection.weight.data = new_weights
        if projection.bias is not None:
            new_projection.bias.data = projection.bias.data

        self.vision_encoder.embeddings.patch_embeddings.projection = new_projection

import torch
from typeguard import typechecked

from lightning_pose.models.backbones.vit_mae import ViTVisionEncoder
from lightning_pose.models.backbones.vit_sam import SamVisionEncoderHF

# to ignore imports for sphix-autoapidoc
__all__ = [
    "build_backbone",
]


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
    if "vit_h_sam" in backbone_arch:
        backbone_arch = "vitb_sam"
        raise DeprecationWarning('vit_h_sam is now deprecated; reverting to "vitb_sam"')

    elif "vit_b_sam" in backbone_arch:
        backbone_arch = "vitb_sam"
        raise DeprecationWarning('vit_b_sam is now deprecated; reverting to "vitb_sam"')

    if "vitb_sam" in backbone_arch:

        base = SamVisionEncoderHF(
            model_name="facebook/sam-vit-base",
            finetune_img_size=image_size,
        )
        encoder_embed_dim = 768

    elif "vitb_imagenet" in backbone_arch:

        base = ViTVisionEncoder(
            model_name="facebook/vit-mae-base",
            finetune_img_size=image_size,
        )
        encoder_embed_dim = base.vision_encoder.config.hidden_size

        if kwargs.get("backbone_checkpoint"):
            load_vit_backbone_checkpoint(base, kwargs["backbone_checkpoint"])

    else:
        raise NotImplementedError(f"{backbone_arch} is not a valid backbone")

    num_fc_input_features = encoder_embed_dim

    return base, num_fc_input_features


def load_vit_backbone_checkpoint(base, checkpoint: str):
    print(f"Loading VIT-MAE weights from {checkpoint}")
    ckpt_vit_pretrain = torch.load(checkpoint, map_location="cpu")
    # Create a filtered state dict for the VIT-MAE part only
    vit_mae_state_dict = {}
    for key, value in ckpt_vit_pretrain.items():
        if key.startswith("vit_mae."):
            model_key = key.replace("vit_mae.", "")
            # Skip known problematic layers with size mismatches
            if any(prob in model_key for prob in [
                "position_embeddings",
                "patch_embeddings.projection",
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

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

    else:
        raise NotImplementedError(f"{backbone_arch} is not a valid backbone")

    num_fc_input_features = encoder_embed_dim

    return base, num_fc_input_features

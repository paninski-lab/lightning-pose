"""Factory for constructing backbone networks."""

import logging
from collections import OrderedDict
from typing import Any, Literal, get_args

import torch
import torch.nn as nn
import torchvision.models as tvmodels

logger = logging.getLogger(__name__)

# to ignore imports for sphix-autoapidoc
__all__ = []

ALLOWED_CONVNET_BACKBONES = Literal[
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnet50_animal_apose',
    'resnet50_animal_ap10k',
    'resnet50_human_jhmdb',
    'resnet50_human_res_rle',
    'resnet50_human_top_res',
    'resnet50_human_hand',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
]

ALLOWED_TRANSFORMER_BACKBONES = Literal[
    'vits_dino',
    'vits_dinov2',
    'vits_dinov3',
    'vitb_dino',
    'vitb_dinov2',
    'vitb_dinov3',
    'vitb_imagenet',
    'vitb_sam',
    'vitb_sam2',
    'vits_sam2',
    'vitt_sam2',
]

ALLOWED_TRANSFORMER_BACKBONES_MULTIVIEW = Literal[
    'vits_dino',
    'vits_dinov2',
    'vits_dinov3',
    'vitb_dino',
    'vitb_dinov2',
    'vitb_dinov3',
    'vitb_imagenet',
]

ALLOWED_BACKBONES = ALLOWED_CONVNET_BACKBONES | ALLOWED_TRANSFORMER_BACKBONES

_ALLOWED_BACKBONE_VALUES: frozenset[str] = frozenset(
    get_args(ALLOWED_CONVNET_BACKBONES) + get_args(ALLOWED_TRANSFORMER_BACKBONES)
)


def build_backbone(
    backbone_arch: ALLOWED_BACKBONES,
    pretrained: bool = True,
    model_type: str = 'heatmap',
    image_size: int = 256,
    **kwargs: Any,
) -> tuple[nn.Module, int]:
    """Load a backbone network by architecture name.

    Dispatches to :func:`_build_transformer_backbone` for ViT-family architectures and
    :func:`_build_convnet_backbone` for ResNet/EfficientNet architectures.

    Args:
        backbone_arch: backbone identifier (e.g. ``"resnet50"``, ``"vits_dino"``).
        pretrained: load pretrained weights; used only for convnet backbones.
        model_type: ``"heatmap"`` or ``"regression"``; controls truncation depth for convnets.
        image_size: height/width in pixels; used only for ViT backbones.
        **kwargs: additional arguments forwarded to the backbone loader
            (e.g. ``backbone_checkpoint`` for ``vitb_imagenet``).

    Returns:
        tuple of (backbone module, number of output features)

    Raises:
        ValueError: if ``backbone_arch`` is not in ``ALLOWED_BACKBONES``.
    """
    if backbone_arch not in _ALLOWED_BACKBONE_VALUES:
        raise ValueError(
            f'"{backbone_arch}" is not a valid backbone; '
            f'allowed backbones: {sorted(_ALLOWED_BACKBONE_VALUES)}'
        )
    if backbone_arch.startswith('vit'):
        return _build_transformer_backbone(backbone_arch, image_size=image_size, **kwargs)  # type: ignore[arg-type]
    else:
        return _build_convnet_backbone(backbone_arch, pretrained=pretrained, model_type=model_type)  # type: ignore[arg-type]


def _build_transformer_backbone(
    backbone_arch: ALLOWED_TRANSFORMER_BACKBONES,
    image_size: int = 256,
    **kwargs: Any,
) -> tuple[nn.Module, int]:
    """Load a Vision Transformer backbone.

    Args:
        backbone_arch: ViT backbone identifier.
        image_size: height/width in pixels of input images (must be square).
        **kwargs: additional arguments (e.g. ``backbone_checkpoint``).

    Returns:
        tuple of (backbone module, number of output features)
    """
    from lightning_pose.models.backbones.vit import VisionEncoder, load_vit_backbone_checkpoint

    if backbone_arch == 'vits_dino':
        base = VisionEncoder(model_name='facebook/dino-vits16')
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == 'vitb_dino':
        base = VisionEncoder(model_name='facebook/dino-vitb16')
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == 'vits_dinov2':
        from lightning_pose.models.backbones.vit_dino import VisionEncoderDino
        base = VisionEncoderDino(model_name='facebook/dinov2-small', pretrained_patch_size=14)
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == 'vitb_dinov2':
        from lightning_pose.models.backbones.vit_dino import VisionEncoderDino
        base = VisionEncoderDino(model_name='facebook/dinov2-base', pretrained_patch_size=14)
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == 'vits_dinov3':
        from lightning_pose.models.backbones.vit_dino import VisionEncoderDino
        base = VisionEncoderDino(
            model_name='facebook/dinov3-vits16-pretrain-lvd1689m', pretrained_patch_size=16,
        )
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch == 'vitb_dinov3':
        from lightning_pose.models.backbones.vit_dino import VisionEncoderDino
        base = VisionEncoderDino(
            model_name='facebook/dinov3-vitb16-pretrain-lvd1689m', pretrained_patch_size=16,
        )
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif 'vitb_imagenet' in backbone_arch:
        base = VisionEncoder(model_name='facebook/vit-mae-base')
        encoder_embed_dim = base.vision_encoder.config.hidden_size
        if kwargs.get('backbone_checkpoint'):
            load_vit_backbone_checkpoint(base, kwargs['backbone_checkpoint'])
    elif backbone_arch == 'vitb_sam':
        from lightning_pose.models.backbones.vit_sam import VisionEncoderSam
        base = VisionEncoderSam(model_name='facebook/sam-vit-base', finetune_img_size=image_size)
        encoder_embed_dim = base.vision_encoder.config.hidden_size
    elif backbone_arch in ('vitb_sam2', 'vits_sam2', 'vitt_sam2'):
        from lightning_pose.models.backbones.vit_sam2 import VisionEncoderSam2
        _sam2_model_names = {
            'vitb_sam2': 'facebook/sam2.1-hiera-base-plus',
            'vits_sam2': 'facebook/sam2.1-hiera-small',
            'vitt_sam2': 'facebook/sam2.1-hiera-tiny',
        }
        base = VisionEncoderSam2(model_name=_sam2_model_names[backbone_arch])
        encoder_embed_dim = base.vision_encoder.config.embed_dim_per_stage[-1]
    else:
        raise NotImplementedError(f'"{backbone_arch}" is not a valid transformer backbone')

    return base, encoder_embed_dim


def _build_convnet_backbone(
    backbone_arch: ALLOWED_CONVNET_BACKBONES,
    pretrained: bool = True,
    model_type: str = 'heatmap',
) -> tuple[nn.Module, int]:
    """Load a ResNet or EfficientNet backbone from torchvision.

    Args:
        backbone_arch: convnet backbone identifier.
        pretrained: load pretrained ImageNet weights.
        model_type: ``"heatmap"`` or ``"regression"``; controls which layers are retained.

    Returns:
        tuple of (backbone module, number of output features)
    """
    if 'resnet50_animal' in backbone_arch:
        base = tvmodels.resnet50(weights=None)
        backbone_type = '_'.join(backbone_arch.split('_')[2:])
        if backbone_type == 'apose':
            ckpt_url = 'https://download.openmmlab.com/mmpose/animal/resnet/res50_animalpose_256x256-e1f30bff_20210426.pth'  # noqa: E501
        else:
            ckpt_url = 'https://download.openmmlab.com/mmpose/animal/resnet/res50_ap10k_256x256-35760eb8_20211029.pth'  # noqa: E501
        state_dict = torch.hub.load_state_dict_from_url(ckpt_url)['state_dict']
        new_state_dict = OrderedDict()
        for key in state_dict:
            if 'backbone' in key:
                new_state_dict['.'.join(key.split('.')[1:])] = state_dict[key]
        base.load_state_dict(new_state_dict, strict=False)

    elif 'resnet50_human' in backbone_arch:
        base = tvmodels.resnet50(weights=None)
        backbone_type = '_'.join(backbone_arch.split('_')[2:])
        if backbone_type == 'jhmdb':
            ckpt_url = 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256-c4ec1a0b_20201122.pth'  # noqa: E501
        elif backbone_type == 'res_rle':
            ckpt_url = 'https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_rle-5f92a619_20220504.pth'  # noqa: E501
        elif backbone_type == 'top_res':
            ckpt_url = 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256-418ffc88_20200812.pth'  # noqa: E501
        elif backbone_type == 'hand':
            ckpt_url = 'https://download.openmmlab.com/mmpose/hand/resnet/res50_onehand10k_256x256-739c8639_20210330.pth'  # noqa: E501
        state_dict = torch.hub.load_state_dict_from_url(ckpt_url)['state_dict']
        new_state_dict = OrderedDict()
        for key in state_dict:
            if 'backbone' in key:
                new_state_dict['.'.join(key.split('.')[1:])] = state_dict[key]
        base.load_state_dict(new_state_dict, strict=False)

    else:
        if pretrained:
            if backbone_arch == 'resnet18':
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.IMAGENET1K_V1
            elif backbone_arch == 'resnet34':
                from torchvision.models import ResNet34_Weights
                weights = ResNet34_Weights.IMAGENET1K_V1
            elif backbone_arch == 'resnet50':
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.IMAGENET1K_V2
            elif backbone_arch == 'resnet101':
                from torchvision.models import ResNet101_Weights
                weights = ResNet101_Weights.IMAGENET1K_V2
            elif backbone_arch == 'resnet152':
                from torchvision.models import ResNet152_Weights
                weights = ResNet152_Weights.IMAGENET1K_V2
            elif backbone_arch == 'efficientnet_b0':
                from torchvision.models import EfficientNet_B0_Weights
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            elif backbone_arch == 'efficientnet_b1':
                from torchvision.models import EfficientNet_B1_Weights
                weights = EfficientNet_B1_Weights.IMAGENET1K_V2
            elif backbone_arch == 'efficientnet_b2':
                from torchvision.models import EfficientNet_B2_Weights
                weights = EfficientNet_B2_Weights.IMAGENET1K_V1
            else:
                raise NotImplementedError(f'"{backbone_arch}" is not a valid convnet backbone')
        else:
            weights = None

        base = getattr(tvmodels, backbone_arch)(weights=weights)

    last_layer_ind = -3 if model_type == 'heatmap' else -2
    backbone = grab_layers_sequential(model=base, last_layer_ind=last_layer_ind)

    if 'resnet' in backbone_arch:
        num_fc_input_features = base.fc.in_features
    elif 'eff' in backbone_arch:
        num_fc_input_features = base.classifier[-1].in_features  # type: ignore[index,attr-defined]
    else:
        raise NotImplementedError

    return backbone, num_fc_input_features


def grab_layers_sequential(model: nn.Module, last_layer_ind: int) -> nn.Sequential:
    """Package a prefix of a model's child layers into a Sequential.

    Args:
        model: source model (resnet or efficientnet).
        last_layer_ind: index of the last child layer to include (inclusive).

    Returns:
        sequential model containing the selected layers.
    """
    layers = list(model.children())[:last_layer_ind + 1]
    return nn.Sequential(*layers)

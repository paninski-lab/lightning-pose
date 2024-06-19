from collections import OrderedDict
from typing import Tuple

import torch
import torchvision.models as tvmodels
from typeguard import typechecked

from lightning_pose.models.base import ALLOWED_BACKBONES

# to ignore imports for sphix-autoapidoc
__all__ = [
    "build_backbone",
    "grab_layers_sequential",
]


@typechecked
def build_backbone(
    backbone_arch: ALLOWED_BACKBONES,
    pretrained: bool = True,
    model_type: str = "heatmap",
    **kwargs,
) -> Tuple:
    """Load backbone weights for resnets, efficientnets, and other models from torchvision.

    Args:
        backbone_arch: which backbone version/weights to use
        pretrained: True to load weights pretrained on imagenet
        model_type: "heatmap" or "regression"

    Returns:
        tuple
            - backbone: pytorch model
            - num_fc_input_features (int): number of input features to fully connected layer

    """

    if backbone_arch == "resnet50_contrastive":
        # load resnet50 pretrained using SimCLR on imagenet
        try:
            from pl_bolts.models.self_supervised import SimCLR
        except ImportError:
            raise Exception(
                "lightning-bolts package is not installed.\n"
                "Run `pip install lightning-bolts` "
                "in order to access 'resnet50_contrastive' backbone"
            )
        ckpt_url = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"  # noqa: E501
        simclr = SimCLR.load_from_checkpoint(ckpt_url, strict=False)
        base = simclr.encoder

    elif "resnet50_animal" in backbone_arch:
        base = getattr(tvmodels, "resnet50")(weights=None)
        backbone_type = "_".join(backbone_arch.split("_")[2:])
        if backbone_type == "apose":
            ckpt_url = "https://download.openmmlab.com/mmpose/animal/resnet/res50_animalpose_256x256-e1f30bff_20210426.pth"  # noqa: E501
        else:
            ckpt_url = "https://download.openmmlab.com/mmpose/animal/resnet/res50_ap10k_256x256-35760eb8_20211029.pth"  # noqa: E501

        state_dict = torch.hub.load_state_dict_from_url(ckpt_url)["state_dict"]
        new_state_dict = OrderedDict()
        for key in state_dict:
            if "backbone" in key:
                new_key = ".".join(key.split(".")[1:])
                new_state_dict[new_key] = state_dict[key]
        base.load_state_dict(new_state_dict, strict=False)

    elif "resnet50_human" in backbone_arch:
        base = getattr(tvmodels, "resnet50")(weights=None)
        backbone_type = "_".join(backbone_arch.split("_")[2:])
        if backbone_type == "jhmdb":
            ckpt_url = "https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256-c4ec1a0b_20201122.pth"  # noqa: E501
        elif backbone_type == "res_rle":
            ckpt_url = "https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_rle-5f92a619_20220504.pth"  # noqa: E501
        elif backbone_type == "top_res":
            ckpt_url = "https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256-418ffc88_20200812.pth"  # noqa: E501
        elif backbone_type == "hand":
            ckpt_url = "https://download.openmmlab.com/mmpose/hand/resnet/res50_onehand10k_256x256-739c8639_20210330.pth"  # noqa: E501

        state_dict = torch.hub.load_state_dict_from_url(ckpt_url)["state_dict"]
        new_state_dict = OrderedDict()
        for key in state_dict:
            if "backbone" in key:
                new_key = ".".join(key.split(".")[1:])
                new_state_dict[new_key] = state_dict[key]
        base.load_state_dict(new_state_dict, strict=False)

    else:
        if pretrained:
            if backbone_arch == "resnet18":
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.IMAGENET1K_V1
            elif backbone_arch == "resnet34":
                from torchvision.models import ResNet34_Weights
                weights = ResNet34_Weights.IMAGENET1K_V1
            elif backbone_arch == "resnet50":
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.IMAGENET1K_V2
            elif backbone_arch == "resnet101":
                from torchvision.models import ResNet101_Weights
                weights = ResNet101_Weights.IMAGENET1K_V2
            elif backbone_arch == "resnet152":
                from torchvision.models import ResNet152_Weights
                weights = ResNet152_Weights.IMAGENET1K_V2
            elif backbone_arch == "efficientnet_b0":
                from torchvision.models import EfficientNet_B0_Weights
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            elif backbone_arch == "efficientnet_b1":
                from torchvision.models import EfficientNet_B1_Weights
                weights = EfficientNet_B1_Weights.IMAGENET1K_V2
            elif backbone_arch == "efficientnet_b2":
                from torchvision.models import EfficientNet_B2_Weights
                weights = EfficientNet_B2_Weights.IMAGENET1K_V1
            else:
                raise NotImplementedError(
                    f"{backbone_arch} is not a valid backbone, choose from {ALLOWED_BACKBONES}")
        else:
            weights = None

        # load resnet or efficientnet models from torchvision.models
        base = getattr(tvmodels, backbone_arch)(weights=weights)

    # get truncated version of backbone; don't include final avg pool
    last_layer_ind = -3 if model_type == "heatmap" else -2
    backbone = grab_layers_sequential(model=base, last_layer_ind=last_layer_ind)

    # compute number of input features
    if "resnet" in backbone_arch:
        num_fc_input_features = base.fc.in_features
    elif "eff" in backbone_arch:
        num_fc_input_features = base.classifier[-1].in_features
    else:
        raise NotImplementedError

    return backbone, num_fc_input_features


@typechecked
def grab_layers_sequential(model, last_layer_ind: int) -> torch.nn.Sequential:
    """Package selected number of layers into a torch.nn.Sequential object.

    Args:
        model: original resnet or efficientnet model
        last_layer_ind: final layer to pass data through

    Returns:
        potentially reduced backbone model

    """
    layers = list(model.children())[: last_layer_ind + 1]
    return torch.nn.Sequential(*layers)

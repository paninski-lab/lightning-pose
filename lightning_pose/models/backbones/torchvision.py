import torch
import torchvision.models as tvmodels
from typeguard import typechecked


@typechecked
def build_backbone(backbone_arch: str, pretrained: bool = True, **kwargs):
    """Load backbone weights for resnets, efficientnets, and other models from torchvision.

    Args:
        backbone_arch: which backbone version/weights to use
        pretrained: True to load weights pretrained on imagenet

    Returns:
        tuple
            - backbone: pytorch model
            - mode (str): "2d" or "3d"
            - num_fc_input_features (int): number of input features to fully connected layer

    """

    mode = "2d"

    if "3d" in backbone_arch:
        base = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        mode = "3d"

    elif backbone_arch == "resnet50_contrastive":
        # load resnet50 pretrained using SimCLR on imagenet
        from pl_bolts.models.self_supervised import SimCLR

        ckpt_url = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
        simclr = SimCLR.load_from_checkpoint(ckpt_url, strict=False)
        base = simclr.encoder

    elif "resnet50_animal" in backbone_arch:
        base = getattr(tvmodels, "resnet50")(pretrained=False)
        backbone_type = "_".join(backbone_arch.split("_")[2:])
        if backbone_type == "apose":
            ckpt_url = "https://download.openmmlab.com/mmpose/animal/resnet/res50_animalpose_256x256-e1f30bff_20210426.pth"
        else:
            ckpt_url = "https://download.openmmlab.com/mmpose/animal/resnet/res50_ap10k_256x256-35760eb8_20211029.pth"

        state_dict = torch.hub.load_state_dict_from_url(ckpt_url)["state_dict"]
        new_state_dict = OrderedDict()
        for key in state_dict:
            if "backbone" in key:
                new_key = ".".join(key.split(".")[1:])
                new_state_dict[new_key] = state_dict[key]
        base.load_state_dict(new_state_dict, strict=False)

    elif "resnet50_human" in backbone_arch:
        base = getattr(tvmodels, "resnet50")(pretrained=False)
        backbone_type = "_".join(backbone_arch.split("_")[2:])
        if backbone_type == "jhmdb":
            ckpt_url = "https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256-c4ec1a0b_20201122.pth"
        elif backbone_type == "res_rle":
            ckpt_url = "https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_rle-5f92a619_20220504.pth"
        elif backbone_type == "top_res":
            ckpt_url = "https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256-418ffc88_20200812.pth"

        state_dict = torch.hub.load_state_dict_from_url(ckpt_url)["state_dict"]
        new_state_dict = OrderedDict()
        for key in state_dict:
            if "backbone" in key:
                new_key = ".".join(key.split(".")[1:])
                new_state_dict[new_key] = state_dict[key]
        base.load_state_dict(new_state_dict, strict=False)

    else:
        # load resnet or efficientnet models from torchvision.models
        base = getattr(tvmodels, backbone_arch)(pretrained=pretrained)

    # get truncated version of backbone; don't include final avg pool
    if "3d" in backbone_arch:
        backbone = grab_layers_sequential_3d(model=base, last_layer_ind=-3)
    else:
        backbone = grab_layers_sequential(model=base, last_layer_ind=-3)

    # compute number of input features
    if "resnet" in backbone_arch and "3d" not in backbone_arch:
        num_fc_input_features = base.fc.in_features
    elif "eff" in backbone_arch:
        num_fc_input_features = base.classifier[-1].in_features
    elif "3d" in backbone_arch:
        num_fc_input_features = base.blocks[-1].proj.in_features // 2
    else:
        raise NotImplementedError

    return backbone, mode, num_fc_input_features


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


@typechecked
def grab_layers_sequential_3d(model, last_layer_ind: int) -> torch.nn.Sequential:
    """This is to use a 3d model to extract features"""
    # the AvgPool3d halves the feature maps dims
    layers = list(model.children())[0][:last_layer_ind + 1] + \
             [torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))]
    return torch.nn.Sequential(*layers)

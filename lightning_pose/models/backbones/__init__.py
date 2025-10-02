from typing import Literal

ALLOWED_BACKBONES = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet50_contrastive",  # needs extra install: pip install -e .[extra_models]
    "resnet50_animal_apose",
    "resnet50_animal_ap10k",
    "resnet50_human_jhmdb",
    "resnet50_human_res_rle",
    "resnet50_human_top_res",
    "resnet50_human_hand",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "vits_dino",
    "vitb_dino",
    "vitb_imagenet",
    "vitb_sam",
]

ALLOWED_TRANSFORMER_BACKBONES = Literal[
    "vits_dino",
    "vitb_dino",
    "vitb_imagenet",
    "vitb_sam",
]

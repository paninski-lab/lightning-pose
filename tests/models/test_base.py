"""Test functionality of base model classes."""

import gc

import torch
import torchvision

from lightning_pose.models.base import BaseFeatureExtractor

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2
HEIGHTS = [128, 256, 384]  # standard numbers, not going to bigger images due to memory
WIDTHS = [120, 246, 380]  # similar but not square
RESNET_BACKBONES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
EFFICIENTNET_BACKBONES = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]
VIT_BACKBONES = [
    "vits_dino",
    "vitb_dino",
    "vits_dinov2",
    "vitb_dinov2",
    "vitb_imagenet",
    "vitb_sam",
]


def test_backbones_resnet():

    for ind, backbone in enumerate(RESNET_BACKBONES):
        model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
        resnet_v = int(backbone.replace("resnet", ""))
        if resnet_v <= 34:  # last block is BasicBlock
            assert (
                type(list(model.backbone.children())[-3][-1])
                == torchvision.models.resnet.BasicBlock
            )
        else:  # different arch; BottleneckBlock
            assert (
                type(list(model.backbone.children())[-3][-1])
                == torchvision.models.resnet.Bottleneck
            )
        # remove model from gpu; then cache can be cleared
        del model
        gc.collect()
        torch.cuda.empty_cache()  # remove tensors from gpu


def test_backbones_efficientnet():
    for ind, backbone in enumerate(EFFICIENTNET_BACKBONES):
        model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
        assert (
            type(list(model.backbone.children())[-1][-2][0])
            == torchvision.models.efficientnet.MBConv
        )
        # remove model from gpu; then cache can be cleared
        del model
        gc.collect()
        torch.cuda.empty_cache()  # remove tensors from gpu


def test_backbones_vit():
    for ind, backbone in enumerate(VIT_BACKBONES):
        model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
        if backbone == "vitb_sam":
            from transformers.models.sam.modeling_sam import SamPatchEmbeddings
            assert isinstance(model.backbone.vision_encoder.patch_embed, SamPatchEmbeddings)
        elif backbone in ["vits_dino", "vitb_dino", "vitb_imagenet"]:
            from transformers.models.vit.modeling_vit import ViTEmbeddings
            assert isinstance(model.backbone.vision_encoder.embeddings, ViTEmbeddings)
        elif backbone in ["vits_dinov2", "vitb_dinov2"]:
            from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings
            assert isinstance(model.backbone.vision_encoder.embeddings, Dinov2Embeddings)
        # remove model from gpu; then cache can be cleared
        del model
        gc.collect()
        torch.cuda.empty_cache()  # remove tensors from gpu


def test_representation_shapes_resnet():

    # loop over different backbone versions and make sure that the resulting
    # representation shapes make sense
    shape_list_pre_pool = {
        'resnet18': {
            128: torch.Size([BATCH_SIZE, 512, 4, 4]),
            256: torch.Size([BATCH_SIZE, 512, 8, 8]),
            384: torch.Size([BATCH_SIZE, 512, 12, 12]),
        },
        'resnet34': {
            128: torch.Size([BATCH_SIZE, 512, 4, 4]),
            256: torch.Size([BATCH_SIZE, 512, 8, 8]),
            384: torch.Size([BATCH_SIZE, 512, 12, 12]),
        },
        'resnet50': {
            128: torch.Size([BATCH_SIZE, 2048, 4, 4]),
            256: torch.Size([BATCH_SIZE, 2048, 8, 8]),
            384: torch.Size([BATCH_SIZE, 2048, 12, 12]),
        },
        'resnet101': {
            128: torch.Size([BATCH_SIZE, 2048, 4, 4]),
            256: torch.Size([BATCH_SIZE, 2048, 8, 8]),
            384: torch.Size([BATCH_SIZE, 2048, 12, 12]),
        },
        'resnet152': {
            128: torch.Size([BATCH_SIZE, 2048, 4, 4]),
            256: torch.Size([BATCH_SIZE, 2048, 8, 8]),
            384: torch.Size([BATCH_SIZE, 2048, 12, 12]),
        },
    }

    for backbone in RESNET_BACKBONES:
        if _TORCH_DEVICE == "cuda":
            torch.cuda.empty_cache()
        model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
        for height, width in zip(HEIGHTS, WIDTHS):
            fake_image_batch = torch.rand(
                size=(BATCH_SIZE, 3, height, width),
                device=_TORCH_DEVICE,
            )
            # representation dim depends on both image size and backbone network
            representations = model(fake_image_batch)
            assert representations.shape == shape_list_pre_pool[backbone][height]
            # remove model/data from gpu; then cache can be cleared
            del fake_image_batch
            del representations
        del model

    gc.collect()
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_representation_shapes_efficientnet():

    # loop over different backbone versions and make sure that the resulting
    # representation shapes make sense
    shape_list_pre_pool = {
        'efficientnet_b0': {
            128: torch.Size([BATCH_SIZE, 1280, 4, 4]),
            256: torch.Size([BATCH_SIZE, 1280, 8, 8]),
            384: torch.Size([BATCH_SIZE, 1280, 12, 12]),
        },
        'efficientnet_b1': {
            128: torch.Size([BATCH_SIZE, 1280, 4, 4]),
            256: torch.Size([BATCH_SIZE, 1280, 8, 8]),
            384: torch.Size([BATCH_SIZE, 1280, 12, 12]),
        },
        'efficientnet_b2': {
            128: torch.Size([BATCH_SIZE, 1408, 4, 4]),
            256: torch.Size([BATCH_SIZE, 1408, 8, 8]),
            384: torch.Size([BATCH_SIZE, 1408, 12, 12]),
        },
    }

    for backbone in EFFICIENTNET_BACKBONES:
        if _TORCH_DEVICE == "cuda":
            torch.cuda.empty_cache()
        model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
        for height, width in zip(HEIGHTS, WIDTHS):
            fake_image_batch = torch.rand(
                size=(BATCH_SIZE, 3, height, width),
                device=_TORCH_DEVICE,
            )
            representations = model(fake_image_batch)
            # representation dim depends on both image size and backbone network
            assert representations.shape == shape_list_pre_pool[backbone][height]
            # remove model/data from gpu; then cache can be cleared
            del fake_image_batch
            del representations
        del model

    gc.collect()
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_representation_shapes_vit():

    # loop over different backbone versions and make sure that the resulting
    # representation shapes make sense
    shape_list_pre_pool = {
        "vits_dino": {
            128: torch.Size([BATCH_SIZE, 384, 8, 8]),
            256: torch.Size([BATCH_SIZE, 384, 16, 16]),
            384: torch.Size([BATCH_SIZE, 384, 24, 24]),
        },
        "vitb_dino": {
            128: torch.Size([BATCH_SIZE, 768, 8, 8]),
            256: torch.Size([BATCH_SIZE, 768, 16, 16]),
            384: torch.Size([BATCH_SIZE, 768, 24, 24]),
        },
        "vits_dinov2": {
            128: torch.Size([BATCH_SIZE, 384, 8, 8]),
            256: torch.Size([BATCH_SIZE, 384, 16, 16]),
            384: torch.Size([BATCH_SIZE, 384, 24, 24]),
        },
        "vitb_dinov2": {
            128: torch.Size([BATCH_SIZE, 768, 8, 8]),
            256: torch.Size([BATCH_SIZE, 768, 16, 16]),
            384: torch.Size([BATCH_SIZE, 768, 24, 24]),
        },
        "vitb_imagenet": {
            128: torch.Size([BATCH_SIZE, 768, 8, 8]),
            256: torch.Size([BATCH_SIZE, 768, 16, 16]),
            384: torch.Size([BATCH_SIZE, 768, 24, 24]),
        },
        "vitb_sam": {
            128: torch.Size([BATCH_SIZE, 768, 8, 8]),
            256: torch.Size([BATCH_SIZE, 768, 16, 16]),
            384: torch.Size([BATCH_SIZE, 768, 24, 24]),
        },
    }

    for backbone in VIT_BACKBONES:
        for height in HEIGHTS:
            if _TORCH_DEVICE == "cuda":
                torch.cuda.empty_cache()
            model = BaseFeatureExtractor(backbone=backbone, image_size=height).to(_TORCH_DEVICE)
            fake_image_batch = torch.rand(
                size=(BATCH_SIZE, 3, height, height),
                device=_TORCH_DEVICE,
            )
            # representation dim depends on both image size and backbone network
            representations = model(fake_image_batch)
            assert representations.shape == shape_list_pre_pool[backbone][height]
            # remove model/data from gpu; then cache can be cleared
            del fake_image_batch
            del representations
            del model

    gc.collect()
    torch.cuda.empty_cache()  # remove tensors from gpu

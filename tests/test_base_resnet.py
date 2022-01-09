import os
import torch
import pytest
import pytorch_lightning as pl
import shutil
from typing import Optional
from lightning_pose.models.base import BaseFeatureExtractor
import torchvision

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_BATCH_SIZE = 12
_HEIGHTS = [
    128,
    256,
    384,
]  # standard numbers, not going to bigger images due to memory
_WIDTHS = [120, 246, 380]  # similar but not square
# print(list(zip(_HEIGHTS, _WIDTHS)))

resnet_versions = [18, 34, 50, 101, 152]


def test_backbone():
    """architecture properties when we truncate network at index at last_resnet_layer_to_get"""
    for ind, resnet_v in enumerate(resnet_versions):
        model = BaseFeatureExtractor(
            resnet_version=resnet_v, last_resnet_layer_to_get=-3
        ).to(_TORCH_DEVICE)
        if resnet_v <= 34:  # lest block is BasicBlock
            assert (
                type(list(model.backbone.children())[-3][-1])
                == torchvision.models.resnet.BasicBlock
            )
        else:  # different archi; BottleneckBlock
            assert (
                type(list(model.backbone.children())[-3][-1])
                == torchvision.models.resnet.Bottleneck
            )
        # remove model/data from gpu; then cache can be cleared
        del model
        torch.cuda.empty_cache()  # remove tensors from gpu


def test_representation_shapes_truncated_resnet():
    """loop over different resnet versions and make sure that the
    resulting representation shapes make sense."""
    # assuming you're truncating before average pool; that depends on image shape
    repres_shape_list_truncated_before_avg_pool_small_image = [
        torch.Size([_BATCH_SIZE, 512, 4, 4]),
        torch.Size([_BATCH_SIZE, 512, 4, 4]),
        torch.Size([_BATCH_SIZE, 2048, 4, 4]),
        torch.Size([_BATCH_SIZE, 2048, 4, 4]),
        torch.Size([_BATCH_SIZE, 2048, 4, 4]),
    ]

    repres_shape_list_truncated_before_avg_pool_medium_image = [
        torch.Size([_BATCH_SIZE, 512, 8, 8]),
        torch.Size([_BATCH_SIZE, 512, 8, 8]),
        torch.Size([_BATCH_SIZE, 2048, 8, 8]),
        torch.Size([_BATCH_SIZE, 2048, 8, 8]),
        torch.Size([_BATCH_SIZE, 2048, 8, 8]),
    ]

    repres_shape_list_truncated_before_avg_pool_big_image = [
        torch.Size([_BATCH_SIZE, 512, 12, 12]),
        torch.Size([_BATCH_SIZE, 512, 12, 12]),
        torch.Size([_BATCH_SIZE, 2048, 12, 12]),
        torch.Size([_BATCH_SIZE, 2048, 12, 12]),
        torch.Size([_BATCH_SIZE, 2048, 12, 12]),
    ]
    shape_list_pre_pool = [
        repres_shape_list_truncated_before_avg_pool_small_image,
        repres_shape_list_truncated_before_avg_pool_medium_image,
        repres_shape_list_truncated_before_avg_pool_big_image,
    ]
    for ind_image in range(len(_HEIGHTS)):
        for ind, resnet_v in enumerate(resnet_versions):
            if _TORCH_DEVICE == "cuda":
                torch.cuda.empty_cache()
            fake_image_batch = torch.rand(
                size=(_BATCH_SIZE, 3, _HEIGHTS[ind_image], _WIDTHS[ind_image]),
                device=_TORCH_DEVICE,
            )
            model = BaseFeatureExtractor(
                resnet_version=resnet_v, last_resnet_layer_to_get=-3
            ).to(_TORCH_DEVICE)
            representations = model(fake_image_batch)
            assert representations.shape == shape_list_pre_pool[ind_image][ind]
            # remove model/data from gpu; then cache can be cleared
            del model
            del fake_image_batch
            del representations
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_resnet_versions():
    # no resnet 11
    pytest.raises(TypeError, BaseFeatureExtractor, resnet_version=11)
    # below should run fine
    model = BaseFeatureExtractor(resnet_version=18, last_resnet_layer_to_get=-3)
    # remove model/data from gpu; then cache can be cleared
    del model
    torch.cuda.empty_cache()


def test_representation_shapes_full_resnet():
    # assuming you're taking everything but the resnet's FC layer
    repres_shape_list_all_but_fc = [
        torch.Size([_BATCH_SIZE, 512, 1, 1]),
        torch.Size([_BATCH_SIZE, 512, 1, 1]),
        torch.Size([_BATCH_SIZE, 2048, 1, 1]),
        torch.Size([_BATCH_SIZE, 2048, 1, 1]),
        torch.Size([_BATCH_SIZE, 2048, 1, 1]),
    ]
    for ind_image in range(len(_HEIGHTS)):
        for ind, resnet_v in enumerate(resnet_versions):
            if _TORCH_DEVICE == "cuda":
                torch.cuda.empty_cache()
            fake_image_batch = torch.rand(
                size=(_BATCH_SIZE, 3, _HEIGHTS[ind_image], _WIDTHS[ind_image]),
                device=_TORCH_DEVICE,
            )
            model = BaseFeatureExtractor(
                resnet_version=resnet_v, last_resnet_layer_to_get=-2
            ).to(_TORCH_DEVICE)
            representations = model(fake_image_batch)
            assert representations.shape == repres_shape_list_all_but_fc[ind]
            # remove model/data from gpu; then cache can be cleared
            del model
            del fake_image_batch
            del representations
    torch.cuda.empty_cache()  # remove tensors from gpu

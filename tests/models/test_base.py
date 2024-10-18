"""Test functionality of base model classes."""

import gc

import numpy as np
import pytest
import segment_anything
import torch
import torchvision

from lightning_pose.models.base import (
    BaseFeatureExtractor,
    convert_bbox_coords,
    normalized_to_bbox,
)

# TODO: cleanup
_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2
HEIGHTS = [128, 256, 384]  # standard numbers, not going to bigger images due to memory
WIDTHS = [120, 246, 380]  # similar but not square
RESNET_BACKBONES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
EFFICIENTNET_BACKBONES = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]
VIT_BACKBONES = [
    "vit_b_sam"
]  # "vit_h_sam" very large (2.6GB), takes too long to download/load


def test_normalized_to_bbox():
    # test when keypoints and bboxes are same size
    keypoints = torch.tensor([
        [[0.0, 0.0]],  # xy for 1 keypoint
        [[1.0, 1.0]],
        [[0.5, 0.5]],
    ])

    bboxes = [
        torch.tensor([0, 0, 100, 200]),  # xyhw
        torch.tensor([20, 30, 100, 200]),
    ]
    for bbox in bboxes:
        kps = normalized_to_bbox(keypoints.clone(), bbox.unsqueeze(0).repeat([3, 1]))
        # (0.0, 0.0) should map to top left corner
        assert kps[0, 0, 0] == bbox[0]
        assert kps[0, 0, 1] == bbox[1]
        # (1.0, 1.0) should map to bottom right corner
        assert kps[1, 0, 0] == bbox[3] + bbox[0]
        assert kps[1, 0, 1] == bbox[2] + bbox[1]
        # (0.5, 0.5) should map to top left corner plus half the new height/width
        assert kps[2, 0, 0] == bbox[3] / 2 + bbox[0]
        assert kps[2, 0, 1] == bbox[2] / 2 + bbox[1]

    # test when keypoints come from context model and bboxes have extra entries for edges
    for bbox in bboxes:
        kps = normalized_to_bbox(keypoints.clone(), bbox.unsqueeze(0).repeat([7, 1]))
        # (0.0, 0.0) should map to top left corner
        assert kps[0, 0, 0] == bbox[0]
        assert kps[0, 0, 1] == bbox[1]
        # (1.0, 1.0) should map to bottom right corner
        assert kps[1, 0, 0] == bbox[3] + bbox[0]
        assert kps[1, 0, 1] == bbox[2] + bbox[1]
        # (0.5, 0.5) should map to top left corner plus half the new height/width
        assert kps[2, 0, 0] == bbox[3] / 2 + bbox[0]
        assert kps[2, 0, 1] == bbox[2] / 2 + bbox[1]


def test_convert_bbox_coords(heatmap_data_module, multiview_heatmap_data_module):

    # -------------------------------------
    # test on single view dataset
    # -------------------------------------

    # params
    x_crop = 25
    y_crop = 40

    # get training batch
    batch_dict = next(iter(heatmap_data_module.train_dataloader()))
    orig_converted = convert_bbox_coords(batch_dict, batch_dict['keypoints'])
    old_image_dims = [batch_dict['images'].size(-2), batch_dict['images'].size(-1)]
    old_bbox = batch_dict["bbox"]
    x_pix = x_crop * old_bbox[:, 3] / old_image_dims[1]
    y_pix = y_crop * old_bbox[:, 2] / old_image_dims[0]

    # create a new batch with smaller & cropped images
    new_dict = batch_dict
    new_dict['images'] = new_dict['images'][:, :, y_crop:-y_crop, x_crop:-x_crop]
    new_dict['bbox'][:, 0] = new_dict['bbox'][:, 0] + x_pix
    new_dict['bbox'][:, 1] = new_dict['bbox'][:, 1] + y_pix
    new_dict['bbox'][:, 2] = new_dict['bbox'][:, 2] - 2 * y_pix
    new_dict['bbox'][:, 3] = new_dict['bbox'][:, 3] - 2 * x_pix
    new_dict['keypoints'][:, 0::2] += x_crop  # keypoints x,y shifted in image
    new_dict['keypoints'][:, 1::2] += y_crop
    new_converted = convert_bbox_coords(new_dict, new_dict['keypoints'])

    # orig and new converted coordinates should be the same
    assert torch.allclose(orig_converted, new_converted, equal_nan=True)

    # -------------------------------------
    # test on dummy multi view dataset
    # -------------------------------------
    batch_dict = {
        "images": torch.tensor(np.random.randn(2, 2, 3, 10, 10)),  # batch, views, RGB, h, w
        "predicted_keypoints": torch.tensor([
            [0.0, 0.0, 0.0, 0.0],  # xy, xy (2 keypoints
            [10.0, 10.0, 10.0, 10.0],
        ]),
        "bbox": torch.tensor([
            [5.0, 6.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # xyhw x 2
            [0.0, 0.0, 123.0, 124.0, 0.0, 0.0, 3.0, 4.0],
        ]),
        "num_views": torch.tensor([2, 2]),
    }
    converted = convert_bbox_coords(batch_dict, batch_dict["predicted_keypoints"])
    assert converted[0, 0] == batch_dict["bbox"][0, 0]
    assert converted[0, 1] == batch_dict["bbox"][0, 1]
    assert converted[0, 2] == batch_dict["bbox"][0, 4]
    assert converted[0, 3] == batch_dict["bbox"][0, 5]
    assert converted[1, 0] == batch_dict["bbox"][1, 3]
    assert converted[1, 1] == batch_dict["bbox"][1, 2]
    assert converted[1, 2] == batch_dict["bbox"][1, 7]
    assert converted[1, 3] == batch_dict["bbox"][1, 6]

    # -------------------------------------
    # test on dummy multi view context dataset
    # -------------------------------------
    batch_dict = {
        "images": torch.tensor(np.random.randn(2, 2, 3, 10, 10)),  # batch, views, RGB, h, w
        "predicted_keypoints": torch.tensor([
            [0.0, 0.0, 0.0, 0.0],  # xy, xy (2 keypoints)
            [10.0, 10.0, 10.0, 10.0],
        ]),
        "bbox": torch.tensor([
            [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context, will be removed
            [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context, will be removed
            [5.0, 6.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # xyhw x 2
            [0.0, 0.0, 123.0, 124.0, 0.0, 0.0, 3.0, 4.0],
            [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context, will be removed
            [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context, will be removed
        ]),
        "num_views": torch.tensor([2, 2, 2, 2, 2, 2]),
    }
    converted = convert_bbox_coords(batch_dict, batch_dict["predicted_keypoints"])
    assert converted[0, 0] == batch_dict["bbox"][2, 0]
    assert converted[0, 1] == batch_dict["bbox"][2, 1]
    assert converted[0, 2] == batch_dict["bbox"][2, 4]
    assert converted[0, 3] == batch_dict["bbox"][2, 5]
    assert converted[1, 0] == batch_dict["bbox"][3, 3]
    assert converted[1, 1] == batch_dict["bbox"][3, 2]
    assert converted[1, 2] == batch_dict["bbox"][3, 7]
    assert converted[1, 3] == batch_dict["bbox"][3, 6]

    # -------------------------------------
    # test error on multi view dataset
    # -------------------------------------
    # get training batch
    batch_dict = next(iter(multiview_heatmap_data_module.train_dataloader()))
    # change number of views for one batch element
    batch_dict["num_views"][0] = 16
    # make sure code complains when batch elements have different numbers of views
    with pytest.raises(ValueError):
        convert_bbox_coords(batch_dict, batch_dict['keypoints'])


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
        assert (
            type(list(model.backbone.children())[0])
            == segment_anything.modeling.image_encoder.PatchEmbed
        )
        # remove model from gpu; then cache can be cleared
        del model
        gc.collect()
        torch.cuda.empty_cache()  # remove tensors from gpu


def test_representation_shapes_resnet():

    # loop over different backbone versions and make sure that the resulting
    # representation shapes make sense

    # 128x120
    rep_shape_list_small_image = [
        torch.Size([BATCH_SIZE, 512, 4, 4]),  # resnet18
        torch.Size([BATCH_SIZE, 512, 4, 4]),  # resnet34
        torch.Size([BATCH_SIZE, 2048, 4, 4]),  # resnet50
        torch.Size([BATCH_SIZE, 2048, 4, 4]),  # resnet101
        torch.Size([BATCH_SIZE, 2048, 4, 4]),  # resnet152
    ]
    # 256x246
    rep_shape_list_medium_image = [
        torch.Size([BATCH_SIZE, 512, 8, 8]),
        torch.Size([BATCH_SIZE, 512, 8, 8]),
        torch.Size([BATCH_SIZE, 2048, 8, 8]),
        torch.Size([BATCH_SIZE, 2048, 8, 8]),
        torch.Size([BATCH_SIZE, 2048, 8, 8]),
    ]
    # 384x380
    rep_shape_list_large_image = [
        torch.Size([BATCH_SIZE, 512, 12, 12]),
        torch.Size([BATCH_SIZE, 512, 12, 12]),
        torch.Size([BATCH_SIZE, 2048, 12, 12]),
        torch.Size([BATCH_SIZE, 2048, 12, 12]),
        torch.Size([BATCH_SIZE, 2048, 12, 12]),
    ]
    shape_list_pre_pool = [
        rep_shape_list_small_image,
        rep_shape_list_medium_image,
        rep_shape_list_large_image,
    ]

    for idx_backbone, backbone in enumerate(RESNET_BACKBONES):
        if _TORCH_DEVICE == "cuda":
            torch.cuda.empty_cache()
        model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
        for idx_image in range(len(HEIGHTS)):
            fake_image_batch = torch.rand(
                size=(BATCH_SIZE, 3, HEIGHTS[idx_image], WIDTHS[idx_image]),
                device=_TORCH_DEVICE,
            )
            # representation dim depends on both image size and backbone network
            representations = model(fake_image_batch)
            assert representations.shape == shape_list_pre_pool[idx_image][idx_backbone]
            # remove model/data from gpu; then cache can be cleared
            del fake_image_batch
            del representations
        del model

    gc.collect()
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_representation_shapes_efficientnet():

    # loop over different backbone versions and make sure that the resulting
    # representation shapes make sense

    # 128x120
    rep_shape_list_small_image = [
        torch.Size([BATCH_SIZE, 1280, 4, 4]),  # efficientnet_b0
        torch.Size([BATCH_SIZE, 1280, 4, 4]),  # efficientnet_b1
        torch.Size([BATCH_SIZE, 1408, 4, 4]),  # efficientnet_b2
    ]
    # 256x246
    rep_shape_list_medium_image = [
        torch.Size([BATCH_SIZE, 1280, 8, 8]),
        torch.Size([BATCH_SIZE, 1280, 8, 8]),
        torch.Size([BATCH_SIZE, 1408, 8, 8]),
    ]
    # 384x380
    rep_shape_list_large_image = [
        torch.Size([BATCH_SIZE, 1280, 12, 12]),
        torch.Size([BATCH_SIZE, 1280, 12, 12]),
        torch.Size([BATCH_SIZE, 1408, 12, 12]),
    ]
    shape_list_pre_pool = [
        rep_shape_list_small_image,
        rep_shape_list_medium_image,
        rep_shape_list_large_image,
    ]

    for idx_backbone, backbone in enumerate(EFFICIENTNET_BACKBONES):
        if _TORCH_DEVICE == "cuda":
            torch.cuda.empty_cache()
        model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
        for idx_image in range(len(HEIGHTS)):
            fake_image_batch = torch.rand(
                size=(BATCH_SIZE, 3, HEIGHTS[idx_image], WIDTHS[idx_image]),
                device=_TORCH_DEVICE,
            )
            representations = model(fake_image_batch)
            # representation dim depends on both image size and backbone network
            assert representations.shape == shape_list_pre_pool[idx_image][idx_backbone]
            # remove model/data from gpu; then cache can be cleared
            del fake_image_batch
            del representations
        del model

    gc.collect()
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_representation_shapes_vit():

    # loop over different backbone versions and make sure that the resulting
    # representation shapes make sense

    # 128x128
    rep_shape_list_small_image = [
        torch.Size([BATCH_SIZE, 256, 8, 8]),  # vit_b_sam
    ]
    # 256x256
    rep_shape_list_medium_image = [
        torch.Size([BATCH_SIZE, 256, 16, 16]),
    ]
    # 384x384
    rep_shape_list_large_image = [
        torch.Size([BATCH_SIZE, 256, 24, 24]),
    ]
    shape_list_pre_pool = [
        rep_shape_list_small_image,
        rep_shape_list_medium_image,
        rep_shape_list_large_image,
    ]

    for idx_backbone, backbone in enumerate(VIT_BACKBONES):
        for idx_image in range(len(HEIGHTS)):
            if _TORCH_DEVICE == "cuda":
                torch.cuda.empty_cache()
            model = BaseFeatureExtractor(
                backbone=backbone, image_size=HEIGHTS[idx_image]
            ).to(_TORCH_DEVICE)
            fake_image_batch = torch.rand(
                size=(BATCH_SIZE, 3, HEIGHTS[idx_image], HEIGHTS[idx_image]),
                device=_TORCH_DEVICE,
            )
            # representation dim depends on both image size and backbone network
            representations = model(fake_image_batch)
            assert representations.shape == shape_list_pre_pool[idx_image][idx_backbone]
            # remove model/data from gpu; then cache can be cleared
            del fake_image_batch
            del representations
            del model

    gc.collect()
    torch.cuda.empty_cache()  # remove tensors from gpu

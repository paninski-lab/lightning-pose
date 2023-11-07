"""Test functionality of base model classes."""

import gc

import segment_anything
import torch
import torchvision

from lightning_pose.models.base import BaseFeatureExtractor, convert_bbox_coords

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2
HEIGHTS = [128, 256, 384]  # standard numbers, not going to bigger images due to memory
WIDTHS = [120, 246, 380]  # similar but not square
RESNET_BACKBONES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
EFFICIENTNET_BACKBONES = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]
VIT_BACKBONES = ["vit_b_sam"]  # "vit_h_sam" very large (2.6GB), takes too long to download/load


def test_convert_bbox_coords(heatmap_data_module_combined):
    # params
    x_crop = 25
    y_crop = 40

    # get training batch
    combined_loader = heatmap_data_module_combined.train_dataloader()
    batch = next(iter(combined_loader))
    # batch is tuple as of lightning 2.0.9
    batch = batch[0] if isinstance(batch, tuple) else batch
    batch_dict = batch['labeled']
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

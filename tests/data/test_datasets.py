"""Test basic dataset functionality."""

import torch


def test_base_dataset(cfg, base_dataset):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    num_targets = base_dataset.num_targets

    # check stored object properties
    assert base_dataset.height == im_height
    assert base_dataset.width == im_width

    # check batch properties
    batch = base_dataset[0]
    assert batch["images"].shape == (3, im_height, im_width)
    assert batch["keypoints"].shape == (num_targets,)
    assert type(batch["images"]) is torch.Tensor
    assert type(batch["keypoints"]) is torch.Tensor


def test_heatmap_dataset(cfg, heatmap_dataset):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    num_targets = heatmap_dataset.num_targets

    # check stored object properties
    assert heatmap_dataset.height == im_height
    assert heatmap_dataset.width == im_width

    # check batch properties
    batch = heatmap_dataset[0]
    assert batch["images"].shape == (3, im_height, im_width)
    assert batch["keypoints"].shape == (num_targets,)
    assert batch["heatmaps"].shape[1:] == heatmap_dataset.output_shape
    assert type(batch["images"]) is torch.Tensor
    assert type(batch["keypoints"]) is torch.Tensor


def test_multiview_heatmap_dataset(cfg_multiview, multiview_heatmap_dataset):

    im_height = cfg_multiview.data.image_resize_dims.height
    im_width = cfg_multiview.data.image_resize_dims.width
    num_targets = multiview_heatmap_dataset.num_targets

    # check stored object properties
    assert multiview_heatmap_dataset.height == im_height
    assert multiview_heatmap_dataset.width == im_width

    # check batch properties
    batch = multiview_heatmap_dataset[0]
    assert batch["images"].shape == (len(cfg_multiview.data.csv_file), 3, im_height, im_width)
    assert batch["keypoints"].shape == (num_targets,)
    assert batch["heatmaps"].shape[1:] == multiview_heatmap_dataset.output_shape
    assert type(batch["images"]) is torch.Tensor
    assert type(batch["keypoints"]) is torch.Tensor


def test_heatmap_dataset_context(cfg, heatmap_dataset_context):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    num_targets = heatmap_dataset_context.num_targets

    # check stored object properties
    assert heatmap_dataset_context.height == im_height
    assert heatmap_dataset_context.width == im_width

    # check batch properties
    batch = heatmap_dataset_context[0]
    assert batch["images"].shape == (5, 3, im_height, im_width)
    assert batch["keypoints"].shape == (num_targets,)
    assert batch["heatmaps"].shape[1:] == heatmap_dataset_context.output_shape
    assert type(batch["images"]) is torch.Tensor
    assert type(batch["keypoints"]) is torch.Tensor


def test_multiview_heatmap_dataset_context(cfg_multiview, multiview_heatmap_dataset_context):
    im_height = cfg_multiview.data.image_resize_dims.height
    im_width = cfg_multiview.data.image_resize_dims.width
    num_targets = multiview_heatmap_dataset_context.num_targets

    # check stored object properties
    assert multiview_heatmap_dataset_context.height == im_height
    assert multiview_heatmap_dataset_context.width == im_width

    # check batch properties
    batch = multiview_heatmap_dataset_context[0]
    assert batch["images"].shape == (2, 5, 3, im_height, im_width)
    assert batch["keypoints"].shape == (num_targets,)
    assert batch["heatmaps"].shape[1:] == multiview_heatmap_dataset_context.output_shape
    assert type(batch["images"]) is torch.Tensor
    assert type(batch["keypoints"]) is torch.Tensor


def test_equal_return_sizes(base_dataset, heatmap_dataset):
    # can only assert the batches are the same if not using imgaug pipeline
    assert base_dataset[0]["images"].shape == heatmap_dataset[0]["images"].shape

"""Test basic dataset functionality."""

import pytest
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
    assert type(batch["keypoints"]) == torch.Tensor


def test_heatmap_dataset_dimensions(cfg, heatmap_dataset):

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
    assert type(batch["keypoints"]) == torch.Tensor
    assert batch["heatmaps"].shape[1:] == heatmap_dataset.output_shape


def test_equal_return_sizes(base_dataset, heatmap_dataset):
    assert torch.equal(base_dataset[0]["images"], heatmap_dataset[0]["images"])

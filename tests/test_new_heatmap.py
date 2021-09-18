import os
from pose_est_nets.models.new_heatmap_tracker import HeatmapTracker
import torch
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
from pose_est_nets.utils.wrappers import predict_plot_test_epoch
from pose_est_nets.utils.IO import set_or_open_folder, load_object
from typing import Optional
import torchvision
from pose_est_nets.models.new_heatmap_tracker import HeatmapTracker

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_BATCH_SIZE = 12
_HEIGHT = 256  # TODO: should be different numbers?
_WIDTH = 256


def test_init():
    model = HeatmapTracker(num_targets=34)
    assert model.num_keypoints == 17
    assert model.num_filters_for_upsampling == 512
    assert model.coordinate_scale == 4


def test_create_double_upsampling_layer():
    fake_image_batch = torch.rand(
        size=(_BATCH_SIZE, 3, _HEIGHT, _WIDTH), device=_TORCH_DEVICE
    )
    heatmap_model = HeatmapTracker(num_targets=34).to(_TORCH_DEVICE)
    upsampling_layer = heatmap_model.create_double_upsampling_layer(
        in_channels=512, out_channels=heatmap_model.num_keypoints
    ).to(_TORCH_DEVICE)
    representations = heatmap_model.get_representations(fake_image_batch)
    upsampled = upsampling_layer(representations)
    assert (
        torch.tensor(upsampled.shape[-2:])
        == torch.tensor(representations.shape[-2:]) * 2
    ).all()
    # now another upsampling layer with a different number of in channels
    upsampling_layer_two = heatmap_model.create_double_upsampling_layer(
        in_channels=heatmap_model.num_keypoints,
        out_channels=heatmap_model.num_keypoints,
    ).to(_TORCH_DEVICE)
    twice_upsampled = upsampling_layer_two(upsampled)
    assert (
        torch.tensor(twice_upsampled.shape[-2:])
        == torch.tensor(upsampled.shape[-2:]) * 2
    ).all()

    # test the output
    pix_shuff = torch.nn.PixelShuffle(2)
    pix_shuffled = pix_shuff(representations)
    assert (
        torch.tensor(pix_shuffled[-2:]) == torch.tensor(representations[-2:] * 2)
    ).all()


def test_heatmaps_from_representations():
    fake_image_batch = torch.rand(
        size=(_BATCH_SIZE, 3, _HEIGHT, _WIDTH), device=_TORCH_DEVICE
    )
    heatmap_model = HeatmapTracker(num_targets=34).to(_TORCH_DEVICE)
    representations = heatmap_model.get_representations(fake_image_batch)
    heatmaps = heatmap_model.heatmaps_from_representations(representations)
    assert (
        torch.tensor(heatmaps.shape[-2:])
        == torch.tensor(fake_image_batch.shape[-2:])
        // (2 ** heatmap_model.downsample_factor)
    ).all()

def test_subpixmaxima(): #Finish writing test
    from pose_est_nets.utils.heatmap_tracker_utils import SubPixelMaxima
    #spm = SubPixelMaxima(output)
